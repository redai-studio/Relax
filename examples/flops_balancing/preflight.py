# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Read-only GPU/environment report; never install drivers or start a job."""

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-gpus", type=int, default=0)
    parser.add_argument("--require-host-memory-gib", type=float, default=0)
    parser.add_argument("--require-disk-free-gib", type=float, default=0)
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
    memory_limits = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory_limits["host_physical"] = int(line.split()[1]) * 1024
    for filename in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        path = Path(filename)
        if path.exists():
            value = path.read_text().strip()
            if value.isdigit() and 0 < int(value) < 2**60:
                memory_limits[filename] = int(value)
    effective_memory = min(memory_limits.values()) if memory_limits else None
    disk_free = shutil.disk_usage(args.output.parent).free
    report = dict(
        versions=versions,
        torch_cuda=torch.version.cuda,
        cuda_available=cuda,
        devices=devices,
        nvidia_smi=smi,
        memory_limits_bytes=memory_limits,
        effective_memory_bytes=effective_memory,
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        disk_free_bytes=disk_free,
        gpu_requirement_met=cuda and len(devices) >= args.require_gpus,
        memory_requirement_met=not args.require_host_memory_gib
        or (effective_memory is not None and effective_memory >= args.require_host_memory_gib * 2**30),
        disk_requirement_met=disk_free >= args.require_disk_free_gib * 2**30,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if (
        (args.require_gpus and not report["gpu_requirement_met"])
        or not report["memory_requirement_met"]
        or not report["disk_requirement_met"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
