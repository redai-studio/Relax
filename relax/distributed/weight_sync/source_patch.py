# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded replacement frames for an immutable source shard."""

import hashlib
from pathlib import Path

from .codec import MAX_CHUNK_BYTES, encode
from .store import digest
from .traffic import file_write, object_all_gather_bytes


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(MAX_CHUNK_BYTES):
            value.update(block)
    return value.hexdigest()


def plan_replacements(
    current: Path, base: Path, width: int, destination: Path, *, expected_current_hash: str | None = None
) -> dict:
    """Return COPY/PATCH/FULL using index/new-value records, never numeric
    deltas.

    A dense chunk selects the complete converter for this source tensor.
    Partial frame files remain private to the candidate's staging directory.
    """
    if width not in (1, 2, 4, 8) or current.stat().st_size % width:
        raise ValueError("Invalid source element width")
    if not base.is_file() or base.stat().st_size != current.stat().st_size:
        return {"route": "FULL", "frames": [], "payload_bytes": 0}
    frames, offset, size = [], 0, 0
    current_hash = hashlib.sha256() if expected_current_hash is not None else None
    with current.open("rb") as source, base.open("rb") as previous:
        while block := source.read(MAX_CHUNK_BYTES):
            if current_hash is not None:
                current_hash.update(block)
            kind, payload = encode(block, previous.read(len(block)), width)
            if kind == "raw":
                return {"route": "FULL", "frames": [], "payload_bytes": 0}
            if payload:
                # REUSE and an immediate FULL fallback have no frame files.
                # Avoid creating their per-tensor scratch directory trees.
                if not frames:
                    destination.mkdir(parents=True, exist_ok=True)
                frame = {
                    "offset": offset // width,
                    "size": len(payload),
                    "sha256": digest(payload),
                    "sequence": len(frames),
                }
                path = destination / f"{frame['sequence']}.bin"
                path.write_bytes(payload)
                file_write(path, len(payload))
                frames.append(frame)
                size += len(payload)
            offset += len(block)
    if current_hash is not None and current_hash.hexdigest() != expected_current_hash:
        raise ValueError("Source candidate content differs from the frozen snapshot")
    return {"route": "PATCH" if frames else "REUSE", "frames": frames, "payload_bytes": size}


def replacement_cost(owners, plans, proposals):
    """Model successful PATCH payload and its control exchanges separately.

    Plan exchange uses Gloo's padded object payload. Base validation and the
    group sync and final patch status each require one agreement; each frame
    adds a read agreement and a remote frame adds an allocation agreement.
    Actual link counters remain the authority for deployment-wide acceptance.
    """
    remote = sum(plan["payload_bytes"] for owner, plan in zip(owners, plans, strict=True) if owner != 0)
    rounds = 3 + sum(len(plan["frames"]) * (1 + (owner != 0)) for owner, plan in zip(owners, plans, strict=True))
    control = object_all_gather_bytes(proposals) + rounds * object_all_gather_bytes([None] * len(proposals))
    return {"payload_bytes": remote, "control_bytes": control}
