# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""ADTS weight switching for U16-packed Teacher replicas."""

import json
from pathlib import Path


LOADER = "relax.backends.sglang.adts_teacher_loader.adts_teacher_loader"


def package_teacher_ids(directory: str, routes: dict[str, str], tp_size: int = 1) -> tuple[dict[str, str], str]:
    """Map routes to package IDs and select the initial route."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    bank_tp = manifest.get("tp_size", 1)
    if bank_tp != tp_size:
        raise ValueError(f"TP{tp_size} Teacher requires a TP{tp_size} bank, got TP{bank_tp} at {directory}")
    teachers = manifest["teachers"]
    checkpoint_ids = {
        (root / record["checkpoint_path"]).resolve(strict=True): name for name, record in teachers.items()
    }
    if len(checkpoint_ids) != len(teachers):
        raise ValueError("Multiple package Teachers refer to the same checkpoint")

    route_to_id = {}
    for route, path in routes.items():
        checkpoint = Path(path).resolve(strict=True)
        if checkpoint not in checkpoint_ids:
            raise ValueError(f"No package Teacher for checkpoint {checkpoint}")
        route_to_id[route] = checkpoint_ids[checkpoint]

    if set(route_to_id.values()) != teachers.keys():
        raise ValueError("Routes must cover all package Teachers")
    initial_id = manifest["initial_teacher"]
    if initial_id not in teachers:
        raise ValueError(f"Unknown initial Teacher {initial_id!r}")
    # Aliases share weights. Pick the same initial route regardless of their order.
    initial_route = min(route for route, teacher_id in route_to_id.items() if teacher_id == initial_id)
    return route_to_id, initial_route


def control_payload(operation: str, target: str, tp_size: int, *, weight_version: str) -> dict:
    """Prepare takes a package directory. Switch and restore take a Teacher ID."""
    import torch
    from sglang.srt.utils import MultiprocessingSerializer

    marker = [(f"relax_adts:{operation}:{target}", torch.empty(0, dtype=torch.uint8))]
    return {
        "serialized_named_tensors": [MultiprocessingSerializer.serialize(marker, output_str=True)] * tp_size,
        "load_format": LOADER,
        "weight_version": weight_version,
        "flush_cache": True,
        "disable_draft_model": True,
    }
