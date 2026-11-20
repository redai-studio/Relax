# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Isolate DeepEP and TileLang compiler preloads in the Qwen training
worker."""

import os
import shlex
import shutil
from argparse import Namespace
from pathlib import Path

from relax.utils.logging_utils import get_logger


def _is_image_wrapper(compiler: Path) -> bool:
    # The Blackwell image installs a shell wrapper and renames NVCC to nvcc.real.
    if (
        not os.access(compiler.with_name("nvcc.real"), os.X_OK)
        or not compiler.is_file()
        or not os.access(compiler, os.X_OK)
    ):
        return False
    with compiler.open("rb") as stream:
        header = stream.read(1024)
    return header.startswith(b"#!") and b"unset LD_PRELOAD" in header.splitlines()


def _configure_deepep_nvcc() -> None:
    """Select the isolated compiler in the checkout actually loaded by this
    worker."""
    root = Path(__file__).resolve().parents[3]
    wrapper = root / "scripts/tools/nvcc-without-preload.sh"
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    default = Path(cuda_home) / "bin/nvcc" if cuda_home else shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    compiler = Path(os.environ.get("EP_JIT_NVCC_COMPILER") or default).absolute()
    if compiler.resolve() == wrapper.resolve():
        underlying = os.environ.get("RELAX_DEEPEP_NVCC")
        if not underlying:
            raise ValueError("EP_JIT_NVCC_COMPILER must select the underlying compiler")
        # A bare command uses PATH; a path with a directory belongs to the worker's
        # current checkout. DeepEP later changes cwd before cold compilation.
        compiler = Path(shutil.which(underlying) or underlying).absolute()
        if compiler.resolve() == wrapper.resolve():
            raise ValueError("EP_JIT_NVCC_COMPILER must select the underlying compiler")
    elif (
        compiler.resolve() == Path(default).resolve()
        and shlex.quote(str(compiler)) == str(compiler)
        and _is_image_wrapper(compiler.resolve())
    ):
        os.environ["EP_JIT_NVCC_COMPILER"] = str(compiler)
        return
    # DeepEP passes the wrapper path unquoted to a shell. The wrapper itself quotes the underlying compiler.
    if not os.access(wrapper, os.X_OK) or shlex.quote(str(wrapper)) != str(wrapper):
        raise RuntimeError(f"DeepEP needs an executable NVCC wrapper at a shell-safe worker path: {wrapper}")
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
