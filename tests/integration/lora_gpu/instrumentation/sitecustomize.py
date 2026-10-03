# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Task-7 lifecycle instrumentation for SGLang (non-invasive).

Loaded automatically by `site` in *every* python process of the experiment
when PYTHONPATH contains this directory and LORA_GPU_EVENTS is set.

It wraps (never re-implements) the SGLang code paths that the RFC relies on:
  * sglang.srt.lora.lora_registry.LoRARegistry  register/unregister/acquire/release/wait_for_unload
  * sglang.srt.lora.lora_manager.LoRAManager    load_lora_adapter/unload_lora_adapter
  * sglang.srt.managers.tokenizer_manager.TokenizerManager._validate_and_resolve_lora

Events are appended as single-line JSON to $LORA_GPU_EVENTS (atomic O_APPEND).
Every hook is wrapped in try/except so that instrumentation can never break
the server.
"""

import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from typing import Any


_PATH = os.environ.get("LORA_GPU_EVENTS")


def _emit(ev: Any, **kw: Any) -> None:
    if not _PATH:
        return
    try:
        rec = {"ts": round(time.time(), 6), "mono": round(time.monotonic(), 6), "pid": os.getpid(), "ev": ev}
        rec.update(kw)
        line = json.dumps(rec, default=str) + "\n"
        fd = os.open(_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass


def _counts(registry: Any, ids: Any) -> Any:
    out = {}
    try:
        for i in ids:
            if i is None:
                continue
            c = registry._counters.get(i)
            out[i[:8]] = None if c is None else c.value()
    except Exception:
        pass
    return out


def _patch_registry(mod: Any) -> None:
    R = mod.LoRARegistry

    orig_register = R.register

    async def register(self, lora_ref: Any) -> Any:
        r = await orig_register(self, lora_ref)
        _emit(
            "registry.register",
            name=lora_ref.lora_name,
            path=lora_ref.lora_path,
            id=lora_ref.lora_id[:8],
            pinned=getattr(lora_ref, "pinned", None),
        )
        return r

    R.register = register

    orig_unregister = R.unregister

    async def unregister(self, lora_name: Any) -> Any:
        r = await orig_unregister(self, lora_name)
        _emit(
            "registry.unregister",
            name=lora_name,
            id=(r or "")[:8],
            remaining_names=sorted(self._registry.keys()),
            counts=_counts(self, [r]),
        )
        return r

    R.unregister = unregister

    orig_acquire = R.acquire

    async def acquire(self, lora_name: Any) -> Any:
        r = await orig_acquire(self, lora_name)
        ids = r if isinstance(r, list) else [r]
        _emit("registry.acquire", name=lora_name, ids=[i[:8] for i in ids if i], counts=_counts(self, ids))
        return r

    R.acquire = acquire

    orig_release = R.release

    async def release(self, lora_id: Any) -> Any:
        r = await orig_release(self, lora_id)
        ids = lora_id if isinstance(lora_id, list) else [lora_id]
        _emit("registry.release", ids=[i[:8] for i in ids if i], counts=_counts(self, ids))
        return r

    R.release = release

    orig_wait = R.wait_for_unload

    async def wait_for_unload(self, lora_id: Any) -> Any:
        t0 = time.monotonic()
        _emit("registry.wait_for_unload.enter", id=(lora_id or "")[:8], counts=_counts(self, [lora_id]))
        r = await orig_wait(self, lora_id)
        _emit("registry.wait_for_unload.exit", id=(lora_id or "")[:8], blocked_s=round(time.monotonic() - t0, 4))
        return r

    R.wait_for_unload = wait_for_unload

    # -- eager (sync) internal helper, gives us the uuid4 minted per fresh load
    orig_reg_adapter = R._register_adapter

    def _register_adapter(self, lora_ref: Any) -> Any:
        r = orig_reg_adapter(self, lora_ref)
        _emit("registry.mint_id", name=lora_ref.lora_name, path=lora_ref.lora_path, id=lora_ref.lora_id[:8])
        return r

    R._register_adapter = _register_adapter

    orig_all = R.get_all_adapters

    def get_all_adapters(self) -> Any:
        r = orig_all(self)
        _emit("registry.snapshot", adapters={k: {"id": v.lora_id[:8], "path": v.lora_path} for k, v in r.items()})
        return r

    R.get_all_adapters = get_all_adapters


def _patch_lora_manager(mod: Any) -> None:
    M = mod.LoRAManager

    orig_load = M.load_lora_adapter

    def load_lora_adapter(self, lora_ref: Any) -> Any:
        _emit(
            "manager.load.enter",
            name=lora_ref.lora_name,
            id=lora_ref.lora_id[:8],
            path=getattr(lora_ref, "lora_path", None),
        )
        r = orig_load(self, lora_ref)
        try:
            _emit(
                "manager.load.exit",
                name=lora_ref.lora_name,
                id=lora_ref.lora_id[:8],
                success=getattr(r, "success", None),
                err=getattr(r, "error_message", None),
                loaded_ids=sorted(k[:8] for k in self.loras.keys()),
            )
        except Exception:
            pass
        return r

    M.load_lora_adapter = load_lora_adapter

    orig_unload = M._unload_lora_adapter

    def unload_lora_adapter(self, lora_ref: Any) -> Any:
        t0 = time.monotonic()
        _emit(
            "manager.unload.enter",
            name=lora_ref.lora_name,
            id=lora_ref.lora_id[:8],
            path=getattr(lora_ref, "lora_path", None),
        )
        r = orig_unload(self, lora_ref)
        try:
            _emit(
                "manager.unload.exit",
                name=lora_ref.lora_name,
                id=lora_ref.lora_id[:8],
                success=getattr(r, "success", None),
                physical_unloads=1,
                elapsed_s=round(time.monotonic() - t0, 4),
                loaded_ids=sorted(k[:8] for k in self.loras.keys()),
            )
        except Exception:
            pass
        return r

    M._unload_lora_adapter = unload_lora_adapter


def _patch_tokenizer(mod: Any) -> None:
    TM = mod.TokenizerManager

    orig = TM._validate_and_resolve_lora

    async def _validate_and_resolve_lora(self, obj: Any) -> Any:
        rids = obj.rid if isinstance(obj.rid, list) else [obj.rid]
        _emit("req.resolve.enter", rids=rids, lora=obj.lora_path)
        try:
            out = await orig(self, obj)
        except Exception as e:
            _emit("req.resolve.error", rids=rids, lora=obj.lora_path, err=f"{type(e).__name__}: {e}")
            raise
        _emit("req.resolve.exit", rids=rids, lora=obj.lora_path, lora_id=str(getattr(obj, "lora_id", ""))[:8])
        return out

    TM._validate_and_resolve_lora = _validate_and_resolve_lora

    # terminal bookkeeping in the tokenizer (finish + abort paths)
    orig_remove = TM._remove_req_state

    def _remove_req_state(self, rid: Any, lifecycle_id: Any = None) -> Any:
        r = orig_remove(self, rid, lifecycle_id)
        _emit("req.state_removed", rid=rid)
        return r

    TM._remove_req_state = _remove_req_state

    # log the "failed before reaching the scheduler" cleanup path
    orig_discard = TM._discard_pending_req_states

    def _discard_pending_req_states(self, obj: Any, lifecycle_ids: Any = None) -> Any:
        rids = self._logical_rids(obj)
        _emit(
            "req.discard_pending",
            rids=rids,
            lora=getattr(obj, "lora_path", None),
            lora_id=str(getattr(obj, "lora_id", ""))[:8],
        )
        return orig_discard(self, obj, lifecycle_ids)

    TM._discard_pending_req_states = _discard_pending_req_states

    # KV/prefix cache flushes: any occurrence invalidates the "no flush" claim
    if hasattr(TM, "flush_cache"):
        orig_flush = TM.flush_cache

        async def flush_cache(self, *a: Any, **kw: Any) -> Any:
            _emit("tm.flush_cache.enter", args=str(a)[:200])
            r = await orig_flush(self, *a, **kw)
            _emit("tm.flush_cache.exit")
            return r

        TM.flush_cache = flush_cache


def _patch_scheduler(mod: Any) -> None:
    S = mod.Scheduler
    if hasattr(S, "flush_cache"):
        orig_flush = S.flush_cache

        def flush_cache(self, *a: Any, **kw: Any) -> Any:
            _emit("scheduler.flush_cache.enter", args=str(a)[:200])
            r = orig_flush(self, *a, **kw)
            _emit("scheduler.flush_cache.exit")
            return r

        S.flush_cache = flush_cache


def _patch_activation_aot_fallback(mod: Any) -> None:
    """sm_80 (A100) workaround for this build: the sglang.kernels JIT
    activation kernel segfaults in cudaLaunchKernelExC on the 550.x driver,
    while the AOT sgl_kernel variant works.

    Force the AOT path; semantics are identical.
    """
    if not getattr(mod, "_sgl_silu_and_mul", None):
        return

    def _act_and_mul(jit_fn: Any, sgl_fn: Any, input: Any, out: Any = None) -> Any:
        if out is None:
            out = input.new_empty(*input.shape[:-1], input.shape[-1] // 2)
        sgl_fn(input, out)
        return out

    mod._act_and_mul = _act_and_mul
    _emit("instr.patch", what="activation_aot_fallback")


def _patch_layernorm_aot_fallback(mod: Any) -> None:
    """Same sm_80 workaround for the JIT rmsnorm paths."""
    try:
        mod._jit_rmsnorm_hf_available = False
    except Exception:
        pass
    try:
        # make the fused-add JIT path ineligible
        mod.is_supported_jit_fused_add_rmsnorm_hidden_size = lambda *a, **k: False
    except Exception:
        pass
    _emit("instr.patch", what="layernorm_aot_fallback")


class _WrappedLoader(importlib.abc.Loader):
    def __init__(self, inner: Any, fn: Any) -> None:
        self._inner = inner
        self._fn = fn

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> Any:
        self._inner.exec_module(module)
        try:
            self._fn(module)
        except Exception as e:  # never break the target import
            _emit("instr.error", mod=module.__name__, err=f"{type(e).__name__}: {e}")


_TARGETS = {
    "sglang.srt.lora.lora_registry": _patch_registry,
    "sglang.srt.lora.lora_manager": _patch_lora_manager,
    "sglang.srt.managers.tokenizer_manager": _patch_tokenizer,
    "sglang.srt.managers.scheduler": _patch_scheduler,
}


if os.environ.get("LORA_GPU_AOT") == "1":
    _TARGETS.update(
        {
            "sglang.srt.layers.activation": _patch_activation_aot_fallback,
            "sglang.srt.layers.layernorm": _patch_layernorm_aot_fallback,
        }
    )


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: Any, path: Any = None, target: Any = None) -> Any:
        fn = _TARGETS.get(fullname)
        if fn is None:
            return None
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:
                spec = None
            if spec is not None and spec.loader is not None:
                spec.loader = _WrappedLoader(spec.loader, fn)
                return spec
        return None


if _PATH:
    try:
        # only instrument processes that actually run sglang
        sys.meta_path.insert(0, _Finder())
    except Exception:
        pass
