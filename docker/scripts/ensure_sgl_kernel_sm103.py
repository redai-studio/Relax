# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Keep or rebuild the bundled SGLang kernel with SM103 support."""

import importlib.metadata as md
import os
import re
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if os.environ["GPU_ARCH"] != "b300":
        raise SystemExit(0)

    # This base uses the renamed sglang-kernel distribution (import: sgl_kernel).
    # Pin its installed version before any downstream dependency resolution.
    with Path(os.environ["PIP_CONSTRAINT"]).open("a") as constraints:
        constraints.write(f"sglang-kernel=={md.version('sglang-kernel')}\n")

    def common_ops() -> list[Path]:
        dist = md.distribution("sglang-kernel")
        libs = [
            Path(dist.locate_file(f))
            for f in dist.files
            if Path(str(f)).name.startswith("common_ops") and str(f).endswith(".so")
        ]
        assert libs, "sgl-kernel common_ops library missing"
        return libs

    def has_sm103(libs: list[Path]) -> bool:
        return any(
            re.search(
                r"sm_103(?:a|f)?(?:[^0-9]|$)",
                subprocess.check_output(["/usr/local/cuda/bin/cuobjdump", "--list-elf", str(lib)], text=True),
            )
            for lib in libs
        )

    if has_sm103(common_ops()):
        sys.stdout.write("sgl-kernel already carries SM103; keeping base build\n")
    else:
        before = md.version("sglang-kernel")
        source = Path("/sgl-workspace/sglang/sgl-kernel")
        cmake = source / "CMakeLists.txt"
        content = cmake.read_text()
        if "compute_103" not in content:
            anchor = '"-gencode=arch=compute_120a,code=sm_120a"'
            assert anchor in content, "sgl-kernel build flags changed; inspect upstream CMake"
            content = content.replace(anchor, anchor + '\n        "-gencode=arch=compute_103a,code=sm_103a"', 1)
            cmake.write_text(content)
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
