# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Lossless BF16 memory buckets; no checkpoint files or floating-point
deltas."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class MemoryBucket:
    names: tuple[tuple[str, tuple[int, ...]], ...]
    elements: int
    changed: int
    encoding: str
    payload: bytes
    snapshot: Any
    dtype: str = "bfloat16"

    def metadata(self) -> dict:
        metadata = {
            "names": self.names,
            "elements": self.elements,
            "changed": self.changed,
            "encoding": self.encoding,
            "dtype": self.dtype,
            "bytes": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }
        _validate_bucket_structure(metadata, self.payload, max_bytes=len(self.payload))
        return metadata


def _encode_bucket(snapshot: Any, previous: Any | None, *, full: bool) -> tuple[int, str, bytes]:
    """Encode an owned BF16/FP32 snapshot using absolute raw-bit values.

    Callers own capture, layout validation and scratch budgeting. A missing
    previous snapshot selects FULL without measuring changes; otherwise FULL
    still reports the actual changed count.
    """
    import torch

    count, width = snapshot.numel(), snapshot.element_size()
    if previous is None:
        changed, encoding, encoded = count, "full", snapshot.view(torch.uint8)
    else:
        words = torch.int16 if width == 2 else torch.int32
        positions = torch.nonzero(snapshot.view(words) != previous.view(words), as_tuple=False).reshape(-1)
        changed = positions.numel()
        if full or changed * (4 + width) >= count * width:
            encoding, encoded = "full", snapshot.view(torch.uint8)
        else:
            encoding = "sparse"
            encoded = torch.cat(
                [positions.to(torch.int32).view(torch.uint8), snapshot.view(words)[positions].view(torch.uint8)]
            )
    return changed, encoding, encoded.cpu().numpy().tobytes()


class MemoryDeltaEncoder:
    """One outstanding version. Advance the GPU baseline only after all ACKs.

    This first path budgets baseline, clones, packing, and worst-case dense
    nonzero/index scratch conservatively at 10x logical BF16 bytes (excluding
    framework allocator workspace). It is for bounded Dense validation.
    """

    def __init__(self, *, bucket_bytes: int = 512 << 20, snapshot_budget_bytes: int = 16 << 30) -> None:
        if bucket_bytes < 2 or snapshot_budget_bytes < 2:
            raise ValueError("Memory budgets must be positive BF16 capacities")
        self.bucket_bytes = bucket_bytes
        self.snapshot_budget_bytes = snapshot_budget_bytes
        self.version = -1
        self.schema: str | None = None
        self._baseline: list[Any] = []
        self._pending: list[MemoryBucket] | None = None
        self._pending_schema: str | None = None

    def prepare(self, named_tensors: Iterable[tuple[str, Any]], *, full: bool = False) -> list[MemoryBucket]:
        import torch

        if self._pending is not None:
            raise RuntimeError("The previous memory version is not committed or aborted")
        groups, current, size, total, seen = [], [], 0, 0, set()
        for name, tensor in named_tensors:
            if not name or name in seen or tensor.dtype != torch.bfloat16 or tensor.layout != torch.strided:
                raise ValueError("Memory delta requires unique names and dense BF16 tensors")
            seen.add(name)
            count = tensor.numel()
            if count == 0 or count >= 2**31:
                raise ValueError("Each tensor must contain 1..2^31-1 BF16 elements")
            nbytes = count * 2
            total += nbytes
            if 10 * total > self.snapshot_budget_bytes:
                raise ValueError("GPU snapshots and worst-case diff scratch exceed the configured budget")
            if current and size + nbytes > self.bucket_bytes:
                groups.append(current)
                current, size = [], 0
            # HF conversion iterators may reuse a conversion/gather buffer at
            # their next yield. Freeze each value before advancing the iterator.
            current.append((name, tensor.detach().clone()))
            size += nbytes
        if current:
            groups.append(current)
        if not groups:
            raise ValueError("Cannot synchronize an empty model")
        layout = [[(name, list(t.shape)) for name, t in group] for group in groups]
        schema = hashlib.sha256(json.dumps(layout, separators=(",", ":")).encode()).hexdigest()
        if self.schema is not None and schema != self.schema:
            raise ValueError("Memory schema changed; create a new stream and FULL baseline")
        buckets = []
        for index, group in enumerate(groups):
            # cat owns its storage: the optimizer cannot mutate a pending version.
            snapshot = torch.cat([tensor.reshape(-1) for _, tensor in group])
            count = snapshot.numel()
            if count >= 2**31:
                raise ValueError("A memory bucket must fit signed int32 positions")
            previous = self._baseline[index] if self._baseline else None
            changed, encoding, payload = _encode_bucket(snapshot, previous, full=full)
            buckets.append(
                MemoryBucket(
                    tuple((name, tuple(t.shape)) for name, t in group), count, changed, encoding, payload, snapshot
                )
            )
        self._pending, self._pending_schema = buckets, schema
        return buckets

    @property
    def pending_schema(self) -> str:
        if self._pending_schema is None:
            raise RuntimeError("No pending memory version")
        return self._pending_schema

    def commit(self, acknowledged: set[str], expected: set[str]) -> None:
        if self._pending is None or not expected or acknowledged != expected:
            raise RuntimeError("Every fixed inference member must ACK before committing")
        self._baseline = [bucket.snapshot for bucket in self._pending]
        self.schema = self._pending_schema
        self.version += 1
        self._pending = self._pending_schema = None

    def abort(self) -> None:
        self._pending = self._pending_schema = None


def validate_bucket(metadata: dict, payload: bytes, *, max_bytes: int) -> None:
    """Validate the complete byte frame before exposing it to the GPU
    loader."""
    _validate_bucket_structure(metadata, payload, max_bytes=max_bytes)
    if hashlib.sha256(payload).hexdigest() != metadata.get("sha256"):
        raise ValueError("Memory bucket checksum mismatch")


def _validate_bucket_structure(metadata: dict, payload: bytes, *, max_bytes: int) -> None:
    import math

    for key in ("elements", "changed", "bytes"):
        if type(metadata.get(key)) is not int or metadata[key] < 0:
            raise ValueError(f"Invalid memory bucket {key}")
    count, changed = metadata["elements"], metadata["changed"]
    if not 0 < count < 2**31 or changed > count:
        raise ValueError("Invalid memory bucket element count")
    encoding = metadata.get("encoding")
    width = {"bfloat16": 2, "float32": 4}.get(metadata.get("dtype", "bfloat16"))
    if width is None:
        raise ValueError("Unsupported memory bucket dtype")
    size = count * width if encoding == "full" else changed * (4 + width) if encoding == "sparse" else -1
    if size != len(payload) or size != metadata["bytes"] or size > max_bytes:
        raise ValueError("Invalid memory bucket payload size")
    names, total = set(), 0
    for name, shape in metadata["names"]:
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Invalid memory tensor name")
        if not isinstance(shape, (tuple, list)) or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("Invalid memory tensor shape")
        names.add(name)
        total += math.prod(shape)
    if total != count:
        raise ValueError("Memory tensor shapes do not cover the payload")
    if encoding == "sparse" and changed:
        import numpy as np

        positions = np.frombuffer(payload, dtype="<i4", count=changed)
        if positions[0] < 0 or positions[-1] >= count or np.any(positions[1:] <= positions[:-1]):
            raise ValueError("Sparse positions must be unique, sorted and in bounds")
