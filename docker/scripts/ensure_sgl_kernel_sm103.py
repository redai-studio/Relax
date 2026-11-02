# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Keep or rebuild the bundled SGLang kernel with SM103 support."""

import importlib.metadata as md
import os
import re
import subprocess
import sys
from pathlib import Path


try:
    import tomllib
except ModuleNotFoundError:
    # Python 3.10 CI already receives this backport through pytest.
    import tomli as tomllib


SGLANG_SOURCE = Path("/sgl-workspace/sglang")


def common_ops() -> list[Path]:
    dist = md.distribution("sglang-kernel")
    libs = [
        Path(dist.locate_file(f))
        for f in dist.files or []
        if Path(str(f)).name.startswith("common_ops") and str(f).endswith(".so")
    ]
    # SGLang selects the precise-math sm100 variant for Blackwell. A compatible
    # cubin in the Hopper variant alone does not validate the library Blackwell loads.
    blackwell = [lib for lib in libs if lib.parent.name == "sm100"]
    libs = blackwell or [lib for lib in libs if lib.parent.name == "sgl_kernel"]
    if not libs:
        raise RuntimeError("sglang-kernel Blackwell common_ops library missing")
    return libs


def has_sm103(libs: list[Path]) -> bool:
    # CUDA family-specific sm_100f cubins also cover SM103, unlike sm_100a.
    # Keep exact SM103 support for packages built before the family target.
    # --list-elf can label sm_100f as sm_100; inspect ELF headers instead.
    # Restrict matching to ELF sections, never treat PTX alone as a native cubin.
    return bool(libs) and all(
        re.search(
            r"Fatbin elf code:\s*=+\s*arch = sm_(?:103[af]?|100f)(?=\s|$)",
            subprocess.check_output(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(lib)], text=True),
        )
        for lib in libs
    )


def kernel_source(root: Path, version: str) -> Path:
    # v0.5.20 moved AOT kernels into the Python source tree.
    candidates = (root / "python/sglang/kernels/aot", root / "sgl-kernel")
    for source in candidates:
        if not (source / "CMakeLists.txt").is_file() or not (source / "pyproject.toml").is_file():
            continue
        project = tomllib.loads((source / "pyproject.toml").read_text())["project"]
        if project.get("name") != "sglang-kernel" or project.get("version") != version:
            raise RuntimeError(f"Kernel source {source} does not match installed sglang-kernel=={version}")
        return source
    raise FileNotFoundError(f"SGLang kernel build sources missing; checked: {', '.join(map(str, candidates))}")


def add_sm103_target(cmake: Path) -> None:
    content = cmake.read_text()
    if "compute_103" not in content:
        anchor = '"-gencode=arch=compute_120a,code=sm_120a"'
        if anchor not in content:
            raise RuntimeError("sglang-kernel build flags changed; inspect upstream CMake")
        content = content.replace(anchor, anchor + '\n        "-gencode=arch=compute_103a,code=sm_103a"', 1)
        cmake.write_text(content)


def main() -> None:
    if os.environ["GPU_ARCH"] != "blackwell":
        return

    # Pin the installed distribution before downstream dependency resolution.
    with Path(os.environ["PIP_CONSTRAINT"]).open("a") as constraints:
        constraints.write(f"sglang-kernel=={md.version('sglang-kernel')}\n")

    if has_sm103(common_ops()):
        sys.stdout.write("sglang-kernel carries SM103-compatible cubins (SM103/SM100f); keeping base build\n")
    else:
        before = md.version("sglang-kernel")
        source = kernel_source(SGLANG_SOURCE, before)
        add_sm103_target(source / "CMakeLists.txt")
        env = dict(
            os.environ,
            CUDA_HOME="/usr/local/cuda",
            MAX_JOBS="32",
            CMAKE_BUILD_PARALLEL_LEVEL="32",
            CMAKE_POLICY_VERSION_MINIMUM="3.5",
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "scikit-build-core", "cmake", "ninja"],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                ".",
                "--no-build-isolation",
                "--force-reinstall",
                "--no-deps",
                "--no-cache-dir",
            ],
            cwd=source,
            env=env,
            check=True,
        )
        assert md.version("sglang-kernel") == before, "Unexpected sgl-kernel version change"
        assert has_sm103(common_ops()), "Rebuilt sgl-kernel still lacks SM103"


if __name__ == "__main__":
    main()
