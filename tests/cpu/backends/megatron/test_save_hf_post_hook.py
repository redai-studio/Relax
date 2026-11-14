# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the --save-hf-post-hook-path injection point.

The hook is called on WORLD rank 0 at the end of `save_hf_model` after all
disk writes complete (BF16/FP8 shards, LoRA adapters, config.json patches).
Contract: sync-return-fast; exceptions logged + swallowed; kwargs are
keyword-only for forward compatibility.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

try:
    from relax.backends.megatron import model as model_mod  # noqa: E402
except (ImportError, AssertionError) as _exc:
    pytest.skip(f"relax.backends.megatron.model unavailable: {_exc}", allow_module_level=True)


def _install_fake_hook_module(monkeypatch, calls, *, raise_exc=None):
    """Register a throwaway module exposing `hook(...)` so load_function
    resolves it."""
    module_name = "relax_test_fake_hooks_" + str(id(calls))
    module = ModuleType(module_name)

    def hook(args, hf_path, rollout_id, *, dtype, is_lora):
        calls.append(
            {
                "args": args,
                "hf_path": hf_path,
                "rollout_id": rollout_id,
                "dtype": dtype,
                "is_lora": is_lora,
            }
        )
        if raise_exc is not None:
            raise raise_exc

    module.hook = hook
    monkeypatch.setitem(sys.modules, module_name, module)
    return f"{module_name}.hook"


class TestInvokeSaveHfPostHook:
    def test_noop_when_hook_path_is_none(self):
        args = SimpleNamespace(save_hf_post_hook_path=None, save_hf_dtype="bf16")
        # Should simply return without raising.
        model_mod._invoke_save_hf_post_hook(args, "/tmp/hf", 3, is_lora=False)

    def test_invokes_hook_with_expected_kwargs(self, monkeypatch):
        calls = []
        path = _install_fake_hook_module(monkeypatch, calls)
        args = SimpleNamespace(save_hf_post_hook_path=path, save_hf_dtype="fp8")

        model_mod._invoke_save_hf_post_hook(args, "/tmp/hf/iter_5", 5, is_lora=True)

        assert len(calls) == 1
        call = calls[0]
        assert call["hf_path"] == "/tmp/hf/iter_5"
        assert call["rollout_id"] == 5
        assert call["dtype"] == "fp8"
        assert call["is_lora"] is True
        assert call["args"] is args

    def test_dtype_defaults_to_bf16_when_absent(self, monkeypatch):
        calls = []
        path = _install_fake_hook_module(monkeypatch, calls)
        args = SimpleNamespace(save_hf_post_hook_path=path)  # no save_hf_dtype attr

        model_mod._invoke_save_hf_post_hook(args, "/tmp/hf", 1, is_lora=False)

        assert calls[0]["dtype"] == "bf16"

    def test_hook_exception_is_swallowed(self, monkeypatch, caplog):
        calls = []
        path = _install_fake_hook_module(monkeypatch, calls, raise_exc=RuntimeError("boom"))
        args = SimpleNamespace(save_hf_post_hook_path=path, save_hf_dtype="bf16")

        # Must not raise; caller (save_hf_model) relies on this.
        model_mod._invoke_save_hf_post_hook(args, "/tmp/hf", 2, is_lora=False)
        assert len(calls) == 1

    def test_hook_receives_string_path_not_pathlike(self, monkeypatch, tmp_path):
        calls = []
        path = _install_fake_hook_module(monkeypatch, calls)
        args = SimpleNamespace(save_hf_post_hook_path=path, save_hf_dtype="bf16")

        model_mod._invoke_save_hf_post_hook(args, tmp_path / "iter_1", 1, is_lora=False)

        assert isinstance(calls[0]["hf_path"], str)


class TestFlushOnFinalSave:
    """`_invoke_save_hf_post_hook(force_sync=True)` calls the module-level
    `flush()` function on the hook module if defined."""

    def test_force_sync_true_calls_flush_when_module_has_it(self, monkeypatch):
        calls = []
        module_name = "relax_test_hook_with_flush_" + str(id(calls))
        module = ModuleType(module_name)

        flush_calls = []

        def _hook(args, hf_path, rollout_id, *, dtype, is_lora):
            calls.append(rollout_id)

        def _flush(timeout_sec: float = 1800.0):
            flush_calls.append(timeout_sec)

        module.hook = _hook
        module.flush = _flush
        monkeypatch.setitem(sys.modules, module_name, module)

        args = SimpleNamespace(save_hf_post_hook_path=f"{module_name}.hook")
        model_mod._invoke_save_hf_post_hook(args, "/tmp/hf", 5, is_lora=False, force_sync=True)

        assert calls == [5]
        assert len(flush_calls) == 1
        assert flush_calls[0] > 0

    def test_force_sync_false_does_not_call_flush(self, monkeypatch):
        calls = []
        module_name = "relax_test_hook_nofsync_" + str(id(calls))
        module = ModuleType(module_name)

        flush_calls = []

        def _hook(args, hf_path, rollout_id, *, dtype, is_lora):
            calls.append(rollout_id)

        def _flush(timeout_sec: float = 1800.0):
            flush_calls.append(timeout_sec)

        module.hook = _hook
        module.flush = _flush
        monkeypatch.setitem(sys.modules, module_name, module)

        args = SimpleNamespace(save_hf_post_hook_path=f"{module_name}.hook")
        model_mod._invoke_save_hf_post_hook(args, "/tmp/hf", 3, is_lora=False, force_sync=False)

        assert flush_calls == []

    def test_force_sync_without_flush_attr_is_noop(self, monkeypatch):
        calls = []
        path = _install_fake_hook_module(monkeypatch, calls)  # no flush attr
        args = SimpleNamespace(save_hf_post_hook_path=path)
        # Must not raise; flush is optional.
        model_mod._invoke_save_hf_post_hook(args, "/tmp/hf", 7, is_lora=False, force_sync=True)
        assert len(calls) == 1

    def test_flush_exception_is_swallowed(self, monkeypatch):
        calls = []
        module_name = "relax_test_hook_flush_boom_" + str(id(calls))
        module = ModuleType(module_name)

        def _hook(args, hf_path, rollout_id, *, dtype, is_lora):
            calls.append(rollout_id)

        def _flush(timeout_sec: float = 1800.0):
            raise RuntimeError("flush boom")

        module.hook = _hook
        module.flush = _flush
        monkeypatch.setitem(sys.modules, module_name, module)

        args = SimpleNamespace(save_hf_post_hook_path=f"{module_name}.hook")
        # Must not propagate the flush error.
        model_mod._invoke_save_hf_post_hook(args, "/tmp/hf", 9, is_lora=False, force_sync=True)
        assert calls == [9]


class TestValidateSaveHfPostHookArgs:
    def test_none_is_noop(self):
        from relax.utils.arguments import validate_save_hf_post_hook_args

        args = SimpleNamespace(save_hf_post_hook_path=None, save_hf=None)
        validate_save_hf_post_hook_args(args)

    def test_requires_save_hf(self):
        from relax.utils.arguments import validate_save_hf_post_hook_args

        args = SimpleNamespace(
            save_hf_post_hook_path="some.module.hook",
            save_hf=None,
        )
        with pytest.raises(ValueError, match="--save-hf"):
            validate_save_hf_post_hook_args(args)

    def test_rejects_unresolvable_path(self):
        from relax.utils.arguments import validate_save_hf_post_hook_args

        args = SimpleNamespace(
            save_hf_post_hook_path="definitely.not.a.real.module.nope",
            save_hf="/tmp/out",
        )
        with pytest.raises((ImportError, AttributeError, ModuleNotFoundError)):
            validate_save_hf_post_hook_args(args)

    def test_accepts_resolvable_path(self, monkeypatch):
        from relax.utils.arguments import validate_save_hf_post_hook_args

        calls = []
        path = _install_fake_hook_module(monkeypatch, calls)
        args = SimpleNamespace(save_hf_post_hook_path=path, save_hf="/tmp/out")
        validate_save_hf_post_hook_args(args)
        # Validation must NOT actually invoke the hook.
        assert calls == []
