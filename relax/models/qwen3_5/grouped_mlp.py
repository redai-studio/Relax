# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Native BF16 three-op MLP installer, independent of model foundation."""

import hashlib
import os
from pathlib import Path
from typing import Any


_INSTALLED = None


def verify_runtime() -> dict:
    import torch
    import transformer_engine_torch as tex

    from relax.models.qwen3_5.dependency_checks import runtime_manifest

    manifest = runtime_manifest()
    library_root = Path(manifest["cublas_library_root"])
    pins = manifest["cublas_library_sha256"]
    expected = {str((library_root / name).resolve()) for name in pins}
    mapped = {
        str(Path(line.split()[-1]).resolve())
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libcublas" in line
    }
    assert mapped == expected, (mapped, expected)
    for name, digest in pins.items():
        assert hashlib.sha256((library_root / name).read_bytes()).hexdigest() == digest
    assert tex.get_cublasLt_version() == 130300
    return {"mapped": sorted(mapped), "te_cublasLt_version": tex.get_cublasLt_version(), "torch": torch.__version__}


def install(args: Any, model: list, optimizer: Any) -> None:
    global _INSTALLED
    identity = tuple(id(chunk) for chunk in model)
    if _INSTALLED is not None:
        assert identity == _INSTALLED
        return
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    assert args.tensor_model_parallel_size in (4,)
    assert args.context_parallel_size == 1
    assert not getattr(args, "use_megatron_fsdp", False)
    assert args.expert_tensor_parallel_size == 1
    assert args.bf16 and not getattr(args, "fp8", None) and not getattr(args, "fp4", None)
    verify_runtime()
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
    os.environ["NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM"] = "1"
    modules = {id(m): m for chunk in model for m in chunk.modules() if isinstance(m, TEGroupedMLP)}
    assert len(modules) == 41, len(modules)
    # Validate every target before mutating any target.
    for module in modules.values():
        assert not module._with_fused_impl and module._fused_ops is None
        assert module._is_fused_impl_supported()
        assert not module.config.use_transformer_engine_op_fuser
        assert not module.config.fp8 and not module.config.fp4
        assert module.config.moe_mlp_glu_interleave_size is None
        assert not module.activation_recompute
        assert not module.linear_fc1.use_bias and not module.linear_fc2.use_bias
        assert module.linear_fc1.fuse_wgrad_accumulation == module.linear_fc2.fuse_wgrad_accumulation
        assert module.linear_fc1.delay_wgrad_compute == module.linear_fc2.delay_wgrad_compute
        if getattr(args, "use_megatron_fsdp", False):
            assert not module.linear_fc1.fuse_wgrad_accumulation
            assert not module.linear_fc1.delay_wgrad_compute
    for module in modules.values():
        module._with_fused_impl = True
    _INSTALLED = identity
