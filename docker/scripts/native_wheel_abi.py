# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Reject native wheels built against a different TRAIN_IMAGE ABI."""

import argparse
import json
import platform
import sysconfig
from pathlib import Path


def runtime_abi() -> dict:
    import torch

    return {
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": bool(torch.compiled_with_cxx11_abi()),
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
    }


def record(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runtime_abi(), indent=2) + "\n")


def check(path: Path) -> None:
    expected = json.loads(path.read_text())
    actual = runtime_abi()
    if expected != actual:
        raise RuntimeError(f"Native wheel build/runtime ABI mismatch: build={expected}, runtime={actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "check"))
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.action == "record":
        record(args.manifest)
    else:
        check(args.manifest)


if __name__ == "__main__":
    main()
