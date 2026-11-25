# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import hashlib
import os
import socket
from pathlib import Path
from uuid import UUID

import torch


def validate_worker_identity(endpoint: dict) -> None:
    """Require an actual CUDA worker incarnation, not its Ray parent."""
    fields = {"boot_id", "hostname", "pid", "gpu_uuid", "device_index", "generation"}
    if (
        not isinstance(endpoint, dict)
        or set(endpoint) != fields
        or any(not isinstance(endpoint[name], str) or not endpoint[name] for name in fields - {"pid", "device_index"})
        or type(endpoint["pid"]) is not int
        or endpoint["pid"] <= 0
        or type(endpoint["device_index"]) is not int
        or endpoint["device_index"] < 0
    ):
        raise ValueError("Invalid CUDA worker identity")
    try:
        UUID(endpoint["boot_id"])
    except ValueError as error:
        raise ValueError("CUDA worker requires a physical host boot identity") from error


def worker_identity(generation: str, device: torch.device) -> dict:
    """Capture cold identity in the process owning the CUDA allocation.

    CUDA ordinals and container hostnames are not physical placement evidence.
    Linux boot identity distinguishes hosts, while UUID binds the device even
    when CUDA_VISIBLE_DEVICES renumbers it. Do not cache this across restarts.
    """
    if device.type != "cuda" or device.index is None or device.index != torch.cuda.current_device():
        raise ValueError("Worker identity requires its current CUDA device")
    uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    if uuid is None:
        raise RuntimeError("CUDA device properties do not expose a device UUID")
    result = dict(
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        hostname=socket.gethostname(),
        pid=os.getpid(),
        gpu_uuid=str(uuid),
        device_index=device.index,
        generation=generation,
    )
    validate_worker_identity(result)
    return result


def snapshot_digests(weights: dict[str, torch.Tensor]) -> dict[str, str]:
    """Digest logical tensor bytes for cold source and receiver validation."""
    return {
        name: hashlib.sha256(
            tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()
        for name, tensor in weights.items()
    }
