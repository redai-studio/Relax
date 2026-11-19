# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Install grouped-value FlashQLA and guard the TileLang KKT store
workaround."""

import importlib
from pathlib import Path


def _guard_kkt_tma(source: Path) -> None:
    """Work around the delivery SM103 KKT TMA store permutation."""
    import tilelang

    module = importlib.import_module("flash_qla.ops.gated_delta_rule.chunk.blackwell.kkt_solve")
    factory = module.tilelang_kkt_solve
    assert tilelang.__version__ == "0.1.12"
    assert factory.pass_configs[tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER] is True


def install_gdn(model: list, source: Path) -> None:
    """Use grouped-value FlashQLA with automatic intra-device partitioning."""
    assert (source / "flash_qla/__init__.py").is_file()
    _guard_kkt_tma(source)
    modules = [m for chunk in model for m in chunk.modules() if type(m).__name__ == "GatedDeltaNet"]
    assert len(modules) == 30
    for module in modules:
        assert module.config.context_parallel_size == 1
        assert module.num_value_heads // module.num_key_heads == 2


def rule(q, k, v, **kw):
    from flash_qla import chunk_gated_delta_rule

    assert kw.pop("cp_context", None) is None
    assert not kw.get("use_qk_l2norm_in_kernel", False)
    return chunk_gated_delta_rule(q, k, v, auto_cp=True, **kw)
