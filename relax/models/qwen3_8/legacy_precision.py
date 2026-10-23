# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Validate the baked FLA configuration before restoring Qwen38 preparation
compilation."""

import importlib

from relax.utils.env import Envs


def prep_compile() -> bool:
    """Validate the baked FLA configuration before restoring legacy
    compilation."""
    if not Envs.RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG:
        return False
    kernel = importlib.import_module("fla.ops.common.chunk_o").chunk_fwd_kernel_o
    tuner = kernel.fn
    if (
        type(kernel).__name__ != "Heuristics"
        or type(tuner).__name__ != "CachedAutotuner"
        or type(tuner.fn).__name__ != "JITFunction"
        or len(tuner.configs) != 1
        or tuner.configs[0].kwargs != {"BK": 64, "BV": 64}
        or tuner.configs[0].num_warps != 4
        or tuner.configs[0].num_stages != 3
    ):
        raise RuntimeError("Audited static GDN output configuration was changed")
    return True
