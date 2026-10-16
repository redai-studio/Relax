# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check GPU imports inside the built image on a host with a GPU driver."""

import sys
from importlib import import_module


def main() -> None:
    import torch

    import_module("transformer_engine.pytorch")
    import_module("sgl_kernel")
    assert torch.cuda.is_available(), "GPU/driver unavailable"
    sys.stdout.write(f"GPU imports OK: {torch.cuda.get_device_name()} {torch.cuda.get_device_capability()}\n")


if __name__ == "__main__":
    main()
