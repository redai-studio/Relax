# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Select the grouped-value GDN implementation for the optimized Qwen
recipe."""

import importlib
import os
from collections.abc import Callable
from pathlib import Path


_native_rule: Callable | None = None


def _guard_kkt_tma(source: Path) -> None:
    """Work around the delivery SM103 KKT TMA store permutation."""
    import tilelang

    module = importlib.import_module("flash_qla.ops.gated_delta_rule.chunk.blackwell.kkt_solve")
    factory = module.tilelang_kkt_solve
    assert tilelang.__version__ == "0.1.12"
    assert factory.pass_configs[tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER] is True


def install_gdn(model: list, source: Path | None = None, *, use_native_fla: bool = False) -> None:
    """Validate the model and select its GDN kernel before the first training
    forward."""
    global _native_rule
    modules = [m for chunk in model for m in chunk.modules() if type(m).__name__ == "GatedDeltaNet"]
    assert len(modules) == 30
    for module in modules:
        assert module.config.context_parallel_size == 1
        assert module.num_value_heads // module.num_key_heads == 2

    if use_native_fla:
        if os.environ.get("FLA_FLASH_QLA") != "0":
            raise RuntimeError("Native FLA requires FLA_FLASH_QLA=0 in the training worker environment")
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        from relax.utils.logging_utils import get_logger

        _native_rule = chunk_gated_delta_rule
        get_logger(__name__).info("Qwen GDN uses native FLA (FLA_FLASH_QLA=0)")
    else:
        if source is None:
            from importlib.util import find_spec

            source = Path(find_spec("flash_qla").origin).parent.parent
        assert (source / "flash_qla/__init__.py").is_file()
        _guard_kkt_tma(source)
        _native_rule = None


def rule(q, k, v, **kw):
    assert kw.pop("cp_context", None) is None
    assert not kw.get("use_qk_l2norm_in_kernel", False)
    if _native_rule is not None:
        return _native_rule(q, k, v, **kw)
    from flash_qla import chunk_gated_delta_rule

    return chunk_gated_delta_rule(q, k, v, auto_cp=True, **kw)
