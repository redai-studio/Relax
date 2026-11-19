# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Megatron initialization hook for TileLang JIT in the Qwen multimodal
recipe."""

import os
from argparse import Namespace

from relax.utils.logging_utils import get_logger


def install(args: Namespace) -> None:
    """Isolate the compiler's preload environment before the first JIT
    kernel."""
    from tilelang.contrib import nvcc

    original = getattr(nvcc, "get_nvcc_subprocess_env", None)
    if not callable(original):
        raise RuntimeError("TileLang lacks get_nvcc_subprocess_env; cannot isolate NVCC preloads")

    def compiler_environment() -> dict[str, str]:
        original_env = original()
        env = dict(os.environ if original_env is None else original_env)
        env.pop("LD_PRELOAD", None)
        return env

    nvcc.get_nvcc_subprocess_env = compiler_environment
    get_logger(__name__).info("TileLang NVCC preload isolation enabled")
