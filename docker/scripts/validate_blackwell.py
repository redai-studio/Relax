# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Validate CPU-loadable build components and record CUDA 13 package
versions."""

import importlib.metadata as md
import json
import os
import sys
from importlib import import_module
from pathlib import Path


def main() -> None:
    import torch

    import_module("flash_attn")
    import_module("apex")
    import_module("fake_int4_quant_cuda")
    import_module("numpy")
    import_module("pyarrow")

    assert torch.version.cuda == "13.0", torch.version.cuda
    assert md.version("transformer-engine") == "2.18.0"
    assert md.version("transformer-engine-cu13") == "2.18.0"
    for package in ("transformer-engine-cu13", "transformer-engine-torch", "sglang-kernel"):
        dist = md.distribution(package)
        libs = [Path(dist.locate_file(f)) for f in dist.files if str(f).endswith(".so")]
        assert libs and all(p.is_file() for p in libs), (package, libs)
    cudnn_version = torch.backends.cudnn.version()
    assert cudnn_version is not None, "cuDNN is unavailable"
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "megatron_bridge_commit": os.environ["MEGATRON_BRIDGE_COMMIT"],
        "cudnn_loaded": cudnn_version,
        "gpu_imports_validated": False,
        "packages": {
            n: md.version(n)
            for n in (
                "transformer-engine",
                "transformer-engine-cu13",
                "transformer-engine-torch",
                "nvidia-cudnn-cu13",
                "nvidia-cudnn-frontend",
                "flash-attn",
                "fla-core",
                "flash-linear-attention",
                "tilelang",
                "sglang",
                "sglang-kernel",
            )
        },
    }
    Path("/etc/relax-cu130-build.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.stdout.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
