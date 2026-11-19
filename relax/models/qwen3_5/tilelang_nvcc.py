# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Megatron initialization hook for CUDA JIT in the Qwen multimodal recipe."""

import filecmp
import os
import shlex
import shutil
from argparse import Namespace
from pathlib import Path

from relax.utils.logging_utils import get_logger


def _configure_deepep_nvcc() -> None:
    """Resolve the compiler on the worker before DeepEP's first dispatch."""
    checkout = Path(__file__).resolve().parents[3]
    wrapper = checkout / "scripts/tools/nvcc-without-preload.sh"
    selected = os.environ.get("EP_JIT_NVCC_COMPILER")
    if selected == str(wrapper) and os.environ.get("RELAX_DEEPEP_NVCC"):
        return
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    default_nvcc = Path(cuda_home) / "bin/nvcc" if cuda_home else shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    compiler = Path(selected or default_nvcc).absolute()
    image_wrapper = checkout / "docker/scripts/nvcc-without-preload.sh"
    if (
        os.access(compiler.resolve().with_name("nvcc.real"), os.X_OK)
        and compiler.is_file()
        and image_wrapper.is_file()
        and filecmp.cmp(compiler, image_wrapper, shallow=False)
        and shlex.quote(str(compiler)) == str(compiler)
    ):
        if selected:
            os.environ["EP_JIT_NVCC_COMPILER"] = str(compiler)
        return
    if not os.access(wrapper, os.X_OK) or shlex.quote(str(wrapper)) != str(wrapper):
        raise RuntimeError(f"DeepEP requires an executable worker wrapper at a shell-safe path: {wrapper}")
    if compiler.resolve() == wrapper.resolve():
        raise ValueError("EP_JIT_NVCC_COMPILER must select the underlying compiler")
    os.environ["RELAX_DEEPEP_NVCC"] = str(compiler)
    os.environ["EP_JIT_NVCC_COMPILER"] = str(wrapper)


def install(args: Namespace) -> None:
    """Isolate the compiler's preload environment before the first JIT
    kernel."""
    from tilelang.contrib import nvcc

    _configure_deepep_nvcc()
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
