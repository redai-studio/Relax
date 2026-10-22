# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Isolate exact-reference FA4 kernels with explicit namespace and arithmetic
API adaptation."""

import hashlib
import importlib
import importlib.abc
import importlib.util
import os
import sys
import sysconfig
from pathlib import Path

import torch


PREFIX = "_relax_fa4_pr2968"


def verify_runtime(reference_root: Path, require_runtime: bool = False) -> dict:
    import cutlass

    assert cutlass.__version__ == "4.6.2"
    prefix = Path(sysconfig.get_paths()["purelib"])
    modules = {
        name: str(module.__file__)
        for name, module in sys.modules.items()
        if (name == "cutlass" or name.startswith("cutlass.")) and getattr(module, "__file__", None)
    }
    assert modules and all(Path(path).resolve().is_relative_to(prefix.resolve()) for path in modules.values())
    mapped = sorted(
        {
            line.rsplit(None, 1)[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if ("_cutlass_ir" in line or "cute_dsl_runtime" in line) and ".so" in line
        }
    )
    ir = [path for path in mapped if "_cutlass_ir" in path]
    runtime = [path for path in mapped if "cute_dsl_runtime" in path]
    assert ir and all(".cu13." in Path(path).name for path in ir)
    assert all("/cu13/" in path for path in runtime)
    assert mapped and all(Path(path).resolve().is_relative_to(prefix.resolve()) for path in mapped)
    if require_runtime:
        assert runtime, "CuTe runtime has not been mapped yet"
    return {
        "cutlass_version": cutlass.__version__,
        "cutlass_file": cutlass.__file__,
        "loaded_cutlass_modules": modules,
        "mapped_cutlass_libraries": mapped,
        "runtime_mapping_required": require_runtime,
    }


def load_exact(reference_root: Path):
    assert not os.environ.get("CUTE_DSL_PTXAS_PATH")
    import cutlass
    import cutlass.cute as cute

    assert cutlass.__version__ == "4.6.2" and hasattr(cute, "FastDivmodDivisorV2")
    prefix = Path(sysconfig.get_paths()["purelib"])
    from .dependency_checks import runtime_manifest

    manifest = runtime_manifest()["fa4_sources"]
    for tree_name, expected in manifest.items():
        tree = reference_root / tree_name
        actual = {
            str(p.relative_to(tree)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in tree.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }
        assert actual == expected and len(actual) == 57
    before = {k: id(v) for k, v in sys.modules.items() if k == "flash_attn.cute" or k.startswith("flash_attn.cute.")}
    module = importlib.import_module(PREFIX + ".cute.interface")
    installed = prefix / PREFIX / "cute"
    for name, digest in manifest["fa4-isolated"].items():
        assert hashlib.sha256((installed / name).read_bytes()).hexdigest() == digest
    assert all(id(sys.modules[k]) == value for k, value in before.items())
    validation = verify_runtime(reference_root)
    return module, {
        "commit": "56e48e3cc3557cfb9f45ec4763c5f971e17cbcbc",
        "pins": 57,
        "namespace": PREFIX,
        "canonical_modules_preserved": len(before),
        "runtime_validation": validation,
    }


_installed = False
_runtime_checked = False
_exact = None


def before_step(
    args: object, rollout_id: int, step_id: int, model: list, optimizer: object, opt_param_scheduler: object
) -> None:
    global _installed, _exact
    if _installed:
        return
    from megatron.core.transformer.attention import SelfAttention

    assert os.environ.get("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "0") == "0"
    reference_root = Path("/opt/relax/qwen35/sources")
    exact, receipt = load_exact(reference_root)
    targets = []
    for chunk in model:
        language = chunk
        while hasattr(language, "module"):
            language = language.module
        language = language.language_model
        targets.extend(m for m in language.modules() if isinstance(m, SelfAttention))
    assert len(targets) == 11
    assert len({id(m.core_attention) for m in targets}) == 11
    for m in targets:
        core = m.core_attention
        assert core.config.context_parallel_size == 1 and core.config.attention_dropout == 0
        assert core.config.kv_channels == 256
    _exact = exact
    _installed = True


@torch.compiler.disable(recursive=False)
def forward(core, query, key, value, attention_mask=None, *pos, packed_seq_params=None, **kw):
    global _runtime_checked
    assert _exact is not None, "Initialize the Qwen35 recipe before attention"
    exact = _exact
    reference_root = Path("/opt/relax/qwen35/sources")
    _scale = getattr(core, "softmax_scale", None)
    if _scale is None:
        _scale = 1.0 / (core.config.kv_channels**0.5)
    assert value.ndim == 3 and value.shape[1:] == (1, 256)
    assert value.dtype == torch.bfloat16
    value = value.contiguous()
    assert attention_mask is None and packed_seq_params is not None
    assert kw.get("attention_bias") is None
    assert query.ndim == key.ndim == value.ndim == 3
    assert value.stride(-1) == 1
    packed = packed_seq_params
    assert packed.qkv_format == "thd"
    assert packed.cu_seqlens_q_padded is None and packed.cu_seqlens_kv_padded is None
    out = exact.flash_attn_varlen_func(
        query.contiguous(),
        key.contiguous(),
        value,
        cu_seqlens_q=packed.cu_seqlens_q,
        cu_seqlens_k=packed.cu_seqlens_kv,
        max_seqlen_q=packed.max_seqlen_q,
        max_seqlen_k=packed.max_seqlen_kv,
        softmax_scale=_scale,
        causal=True,
    )
    if not _runtime_checked:
        verify_runtime(reference_root, require_runtime=True)
        _runtime_checked = True
    if isinstance(out, tuple):
        out = out[0]
    return out.flatten(1)
