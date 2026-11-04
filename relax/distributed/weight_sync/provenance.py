# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Small runtime identity record for disk-sync acceptance artifacts."""

import hashlib
import json
import os
import sys
from pathlib import Path


def runtime_identity() -> dict:
    modules = {}
    for name in ("relax", "torch", "ray", "sglang", "megatron.core", "megatron.bridge"):
        module = sys.modules.get(name)
        filename = getattr(module, "__file__", None)
        if filename is not None:
            path = Path(filename).resolve()
            modules[name] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "version": str(getattr(module, "__version__", "source")),
            }
    source_root = Path(__file__).resolve().parents[3]
    marker = source_root / ".source-complete.json"
    return {
        "pid": os.getpid(),
        "executable": sys.executable,
        "argv": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source": json.loads(marker.read_text()) if marker.exists() else {"path": str(source_root)},
        "modules": modules,
    }
