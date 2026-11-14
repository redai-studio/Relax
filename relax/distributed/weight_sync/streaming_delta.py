# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded incremental capture for sender-owned BF16/FP32 shards."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from time import monotonic
from typing import Any, Iterable, Iterator

from .memory_delta import MemoryBucket, _encode_bucket


def steady_state_snapshot_bytes(buckets: Iterable[MemoryBucket], *, packing_bytes: int = 0) -> int:
    """Conservative next-delta admission from a rank's captured directory.

    Reserve old baseline plus new snapshots, the largest diff scratch or
    capture/concat clone, and the adapter's largest live slab plus shard.
    Payload length is irrelevant: even an empty delta owns a full snapshot.
    Converter/library workspace remains a separate admission estimate.
    """
    if type(packing_bytes) is not int or packing_bytes < 0:
        raise ValueError("Packing bytes must be a nonnegative integer")
    snapshots, scratch, clone = 0, 0, 0
    for bucket in buckets:
        width = {"bfloat16": 2, "float32": 4}.get(bucket.dtype)
        if width is None or type(bucket.elements) is not int or not 0 < bucket.elements < 2**31:
            raise ValueError("Steady-state admission requires bounded BF16/FP32 buckets")
        size = bucket.elements * width
        snapshots += size
        clone = max(clone, size)
        scratch = max(scratch, bucket.elements * (17 + 2 * width))
    return 2 * snapshots + max(scratch, clone) + packing_bytes


class SnapshotBudget:
    """One budget shared by every encoder on a training rank.

    Counts live baseline/pending/packing storage and conservative diff scratch.
    The caller separately budgets the training model, conversion buffers and
    allocator/library workspace from the measured native run.
    """

    def __init__(self, limit_bytes: int) -> None:
        if type(limit_bytes) is not int or limit_bytes <= 0:
            raise ValueError("Snapshot budget must be positive")
        self.limit_bytes = limit_bytes
        self.used_bytes = 0
        self.peak_bytes = 0

    def reserve(self, size: int) -> None:
        if size < 0 or self.used_bytes + size > self.limit_bytes:
            raise ValueError("Sender snapshots and diff scratch exceed the shared rank budget")
        self.used_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.used_bytes)

    def release(self, size: int) -> None:
        if not 0 <= size <= self.used_bytes:
            raise RuntimeError("Invalid snapshot budget release")
        self.used_bytes -= size

    @contextmanager
    def temporary(self, size: int) -> Iterator[None]:
        self.reserve(size)
        try:
            yield
        finally:
            self.release(size)


class StreamingDeltaEncoder:
    """Freeze one bucket before advancing a reusable conversion iterator.

    A successful finish does not advance the baseline. Prepare all commit
    references before the distributed commit gate, then commit by swapping
    references. A process failure still invalidates the entire group/stream.
    """

    def __init__(self, budget: SnapshotBudget, *, bucket_bytes: int) -> None:
        if type(bucket_bytes) is not int or bucket_bytes <= 0:
            raise ValueError("Bucket capacity must be positive")
        self.budget, self.bucket_bytes = budget, bucket_bytes
        self.version = -1
        self.schema: str | None = None
        self._baseline: list[tuple[tuple, str, Any]] = []
        self._baseline_bytes = 0
        self._pending: list[MemoryBucket] | None = None
        self._pending_snapshots: list[tuple[tuple, str, Any]] = []
        self._pending_bytes = 0
        self._pending_schema: str | None = None
        self._current: list[tuple[str, Any]] = []
        self._current_bytes = 0
        self._seen: set[str] = set()
        self._full = False
        self._compare = True
        self._finished = False
        self._next_baseline: list | None = None
        self.clone_enqueue_s = self.pack_diff_to_host_s = 0.0

    def begin(self, *, full: bool = False, compare_changes: bool = True) -> None:
        if self._pending is not None:
            raise RuntimeError("A shard capture is already outstanding")
        if not compare_changes and not full:
            raise ValueError("Skipping the comparison requires FULL encoding")
        self._pending = []
        self._pending_snapshots = []
        self._seen = set()
        self._full, self._finished = full, False
        self._compare = compare_changes
        self.clone_enqueue_s = self.pack_diff_to_host_s = 0.0

    def add(self, name: str, tensor: Any) -> None:
        import torch

        if self._pending is None or self._finished:
            raise RuntimeError("Begin an unfinished shard capture before adding tensors")
        if (
            not isinstance(name, str)
            or not name
            or name in self._seen
            or tensor.dtype not in (torch.bfloat16, torch.float32)
            or tensor.layout != torch.strided
            or not 0 < tensor.numel() < 2**31
        ):
            raise ValueError("Shards require unique BF16/FP32 names with bounded dense tensors")
        size = tensor.numel() * tensor.element_size()
        if self._current and (
            self._current_bytes + size > self.bucket_bytes or self._current[0][1].dtype != tensor.dtype
        ):
            self._flush()
        started = monotonic()
        self.budget.reserve(size)
        try:
            frozen = tensor.detach().clone(memory_format=torch.contiguous_format)
        except Exception:
            self.budget.release(size)
            raise
        self._current.append((name, frozen))
        self._current_bytes += size
        self._seen.add(name)
        self.clone_enqueue_s += monotonic() - started

    def _flush(self) -> None:
        import torch

        if not self._current:
            return
        started = monotonic()
        names = tuple((name, tuple(t.shape)) for name, t in self._current)
        dtype = self._current[0][1].dtype
        dtype_name = str(dtype).removeprefix("torch.")
        index = len(self._pending)
        previous = None
        if self.schema is not None:
            if index >= len(self._baseline) or self._baseline[index][:2] != (names, dtype_name):
                raise ValueError("Shard layout changed; create a new stream and FULL baseline")
            previous = self._baseline[index][2]
        size = self._current_bytes
        self.budget.reserve(size)
        try:
            snapshot = torch.cat([tensor.reshape(-1) for _, tensor in self._current])
            self._current.clear()
            self._current_bytes = 0
            self.budget.release(size)
            count, width = snapshot.numel(), snapshot.element_size()
            if count >= 2**31:
                raise ValueError("A shard bucket must fit signed int32 positions")
            if not self._compare:
                previous = None
            # bool comparison + int64 nonzero + int32 cast + gathered
            # values + concatenated sparse bytes, bounded by bucket size.
            scratch_bytes = count * (17 + 2 * width) if previous is not None else 0
            with self.budget.temporary(scratch_bytes):
                changed, encoding, payload = _encode_bucket(snapshot, previous, full=self._full)
            self._pending.append(MemoryBucket(names, count, changed, encoding, payload, None, dtype_name))
            self._pending_snapshots.append((names, dtype_name, snapshot))
            self._pending_bytes += size
        except Exception:
            self.budget.release(size)
            raise
        self.pack_diff_to_host_s += monotonic() - started

    def finish(self) -> list[MemoryBucket]:
        if self._pending is None or self._finished:
            raise RuntimeError("No unfinished shard capture")
        self._flush()
        if not self._pending:
            raise ValueError("Cannot publish an empty shard stream")
        layout = [(bucket.names, bucket.dtype) for bucket in self._pending]
        schema = hashlib.sha256(json.dumps(layout, separators=(",", ":")).encode()).hexdigest()
        if self.schema is not None and schema != self.schema:
            raise ValueError("Shard layout changed; create a new stream and FULL baseline")
        self._pending_schema, self._finished = schema, True
        return list(self._pending)

    @property
    def pending_schema(self) -> str:
        if not self._finished or self._pending_schema is None:
            raise RuntimeError("Finish the shard capture before publishing")
        return self._pending_schema

    def prepare_commit(self, acknowledged: set[str], expected: set[str]) -> None:
        if not self._finished or not expected or acknowledged != expected:
            raise RuntimeError("Every fixed shard member must ACK before committing")
        self._next_baseline = list(self._pending_snapshots)

    def commit(self) -> None:
        if self._next_baseline is None:
            raise RuntimeError("Prepare commit references before the collective commit gate")
        self._baseline, self._next_baseline = self._next_baseline, None
        self.budget.release(self._baseline_bytes)
        self._baseline_bytes = self._pending_bytes
        self.schema = self._pending_schema
        self.version += 1
        self._pending = None
        self._pending_snapshots = []
        self._pending_bytes = 0
        self._pending_schema = None
        self._finished = False

    def abort(self) -> None:
        self._current.clear()
        self._pending = self._next_baseline = None
        self._pending_snapshots = []
        self.budget.release(self._current_bytes + self._pending_bytes)
        self._current_bytes = self._pending_bytes = 0
        self._pending_schema = None
        self._finished = False

    def close(self) -> None:
        self.abort()
        self._baseline.clear()
        self.budget.release(self._baseline_bytes)
        self._baseline_bytes = 0
