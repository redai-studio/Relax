# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Link the memory saver preload library for the installed CUDA version."""

import sysconfig
from pathlib import Path


def main() -> None:
    import torch

    major = torch.version.cuda.split(".")[0]
    assert major == "13", torch.version.cuda
    root = Path(sysconfig.get_paths()["platlib"])
    source = root / f"torch_memory_saver_hook_mode_preload_cu{major}.abi3.so"
    target = root / "torch_memory_saver_hook_mode_preload.abi3.so"
    assert source.is_file(), f"Missing memory saver CUDA {major} library: {source}"
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source.name)


if __name__ == "__main__":
    main()
