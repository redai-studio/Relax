# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Control messages and checkpoint routing for shared Teacher U16 packages."""

import json
from pathlib import Path


LOADER = "relax.backends.sglang.adp_teacher_loader.adp_teacher_loader"


def delta_tp_size(args) -> int:
    """Every replica loads the same bank, sharded across its own TP workers."""
    budget = (getattr(args, "resource", None) or {}).get("teacher", (1, 1))[1]
    tp_size = getattr(args, "teacher_num_gpus_per_engine", None)
    if tp_size is None:
        tp_size = budget
    if tp_size <= 0 or budget % tp_size:
        raise ValueError("--opd-teacher-delta-dir requires a positive Teacher GPU budget divisible by replica TP.")
    return tp_size


def package_teacher_ids(directory: str, routes: dict[str, str], tp_size: int = 1) -> dict[str, str]:
    """Resolve each logical Teacher to one physical checkpoint in the bank."""
    root = Path(directory).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("version"), manifest.get("runtime"), manifest.get("tp_size")) != (1, "sglang-dense-v1", tp_size):
        raise ValueError(f"Shared Teacher delta loading requires an ADP1 TP{tp_size} manifest")
    ids = {}
    teachers = manifest["teachers"]
    for route, path in routes.items():
        checkpoint = Path(path).resolve(strict=True)
        matches = [
            name
            for name, record in teachers.items()
            if (
                (root / record["checkpoint_path"]).resolve() == checkpoint
                if record.get("checkpoint_path")
                else name == checkpoint.name
            )
        ]
        if len(matches) != 1:
            raise ValueError(f"No unique package Teacher for checkpoint {checkpoint}")
        ids[route] = matches[0]
    if not ids or set(ids.values()) != set(teachers) or next(iter(ids.values())) != manifest.get("initial_teacher"):
        raise ValueError("Routes must cover the package Teachers and start with its initial Teacher")
    return ids


def control_payload(operation: str, teacher_id: str, tp_size: int = 1) -> dict:
    """Send only a control marker; compressed bytes stay on the worker."""
    import torch
    from sglang.srt.utils import MultiprocessingSerializer

    marker = [(f"relax_adp:{operation}:{teacher_id}", torch.empty(0, dtype=torch.uint8))]
    return {
        "serialized_named_tensors": [MultiprocessingSerializer.serialize(marker, output_str=True)] * tp_size,
        "load_format": LOADER,
        "weight_version": teacher_id,
        "flush_cache": True,
        "disable_draft_model": True,
    }
