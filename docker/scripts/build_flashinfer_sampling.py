# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Build the FlashInfer AOT sampling kernel for B300."""

import os
import pathlib
import re
import shutil
import subprocess
import sys


def main() -> None:
    if os.environ.get("GPU_ARCH") != "b300":
        sys.stdout.write(f"skip flashinfer sampling sm_103a rebuild (GPU_ARCH={os.environ.get('GPU_ARCH')})\n")
        raise SystemExit(0)

    from flashinfer.jit.sampling import gen_sampling_module

    aot = pathlib.Path("/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sampling/sampling.so")
    if not aot.parent.is_dir():
        raise SystemExit(
            f"flashinfer AOT sampling dir not found: {aot.parent}. flashinfer layout changed; update this build step."
        )
    if aot.exists():  # remove stale prebuilt so it no longer shadows the JIT build
        aot.unlink()

    spec = gen_sampling_module()
    spec.build(verbose=True)  # compiles for FLASHINFER_CUDA_ARCH_LIST, no GPU required
    built = spec.jit_library_path
    if not built.exists():
        raise SystemExit(f"flashinfer sampling build produced no artifact at {built}")

    elfs = subprocess.check_output(["/usr/local/cuda/bin/cuobjdump", "--list-elf", str(built)], text=True)
    assert re.search(r"sm_103(?:a|f)?(?:[^0-9]|$)", elfs), "FlashInfer sampling lacks SM103"
    shutil.copy(built, aot)  # reinstall into AOT slot -> loaded directly at runtime
    sys.stdout.write(f"installed flashinfer sampling.so -> {aot}\n")


if __name__ == "__main__":
    main()
