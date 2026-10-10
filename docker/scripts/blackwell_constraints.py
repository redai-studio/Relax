# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Preserve the CUDA 13 base packages and write build constraints."""

import importlib.metadata as md
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    import torch

    assert torch.version.cuda == "13.0", (torch.__version__, torch.version.cuda)
    assert "release 13.0" in subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"], text=True)
    pins = []
    for name in (
        "torch",
        "torchvision",
        "torchaudio",
        "triton",
        "sglang",
        "sgl-kernel",
        "flashinfer-python",
        "nvidia-cudnn-cu13",
        "nvidia-cudnn-frontend",
    ):
        try:
            pins.append(f"{name}=={md.version(name)}")
        except md.PackageNotFoundError:
            if name in ("torch", "triton", "sglang"):
                raise
    pins += [
        f"transformer-engine=={os.environ['TE_VERSION']}",
        f"transformer-engine-cu13=={os.environ['TE_VERSION']}",
        f"transformer-engine-torch=={os.environ['TE_VERSION']}",
        "numpy==2.2.6",
        "pyarrow==20.0.0",
        "huggingface-hub==1.10.0",
        "packaging==25.0",
        "protobuf>=6.33.5,<7",
        "pillow==11.3.0",
        "flash-attn==2.7.4.post1",
        "flash-linear-attention==0.4.1",
        f"tilelang=={os.environ['TILELANG_VERSION']}",
    ]
    Path("/etc/relax-cu130-constraints.txt").write_text("\n".join(pins) + "\n")
    sys.stdout.write(f"Preserving base Torch: {torch.__version__} CUDA: {torch.version.cuda}\n")


if __name__ == "__main__":
    main()
