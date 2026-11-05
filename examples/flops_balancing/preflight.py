# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Read-only GPU/environment report; never install drivers or start a job."""

import argparse
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-gpus", type=int, default=0)
    args = parser.parse_args()
    versions = {}
    for package in (
        "torch",
        "transformers",
        "ray",
        "TransferQueue",
        "megatron-core",
        "transformer-engine",
        "flash-attn",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cuda = torch.cuda.is_available()
    devices = [
        {
            "name": torch.cuda.get_device_name(index),
            "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            "capability": list(torch.cuda.get_device_capability(index)),
        }
        for index in range(torch.cuda.device_count())
    ]
    smi = {}
    if shutil.which("nvidia-smi"):
        for option in (["-L"], ["topo", "-m"]):
            result = subprocess.run(["nvidia-smi", *option], text=True, capture_output=True, timeout=30)
            smi[" ".join(option)] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    report = dict(
        versions=versions,
        torch_cuda=torch.version.cuda,
        cuda_available=cuda,
        devices=devices,
        nvidia_smi=smi,
        disk_free_bytes=shutil.disk_usage(args.output.parent).free,
        gpu_requirement_met=cuda and len(devices) >= args.require_gpus,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_gpus and not report["gpu_requirement_met"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
