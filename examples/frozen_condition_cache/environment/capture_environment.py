"""Record actual software/hardware without allocating CUDA tensors or
training."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def installed_distributions() -> list[dict[str, str]]:
    """Record transitive versions only, never installation URLs or
    credentials."""
    return sorted(
        ({"name": item.metadata["Name"], "version": item.version} for item in importlib.metadata.distributions()),
        key=lambda item: (item["name"].lower(), item["version"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    names = [
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "peft",
        "accelerate",
        "huggingface-hub",
        "numpy",
        "pillow",
        "pyarrow",
        "safetensors",
    ]
    packages = {name: importlib.metadata.version(name) for name in names}
    for module in ["torch", "torchvision", "diffusers", "transformers", "peft", "pyarrow"]:
        importlib.import_module(module)
    import torch

    if torch.__version__.split("+")[0] != "2.9.0":
        raise RuntimeError("Unexpected Torch version; refuse an unrecorded environment")
    if packages["diffusers"] != "0.37.0":
        raise RuntimeError("Unexpected Diffusers version")
    direct = importlib.metadata.distribution("diffusers").read_text("direct_url.json")
    if (
        not direct
        or json.loads(direct).get("vcs_info", {}).get("commit_id") != "d791c5c024014784df4a8dac2601e19fb4d300fc"
    ):
        raise RuntimeError("Diffusers must be installed at the locked official commit")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    result = {
        "schema_version": 1,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "installed_distributions": installed_distributions(),
        "torch_cuda_runtime": torch.version.cuda,
        "diffusers_source_commit": json.loads(direct)["vcs_info"]["commit_id"],
        "gpu_query": gpu.stdout.strip(),
        "gpu_query_error": gpu.stderr.strip(),
        "gpu_query_returncode": gpu.returncode,
        "imports_passed": True,
        "gpu_tensor_tested": False,
        "model_loaded": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
