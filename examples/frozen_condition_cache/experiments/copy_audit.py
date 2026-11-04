"""Correctness-only accounting of dispatcher-visible logical tensor copies.

The default observer reads metadata only. An opt-in bounded correctness audit
also hashes CPU source values of condition-scoped H2D copies; it never reads a
CUDA tensor back to hash it. Neither mode estimates physical PCIe traffic.
Views, unobserved backend-internal copies, other threads, collectives, and
compiler-generated copies are outside its scope. Never use it in performance.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import torch
from torch.utils._python_dispatch import TorchDispatchMode


_ACTIVE_AUDIT: ContextVar[LogicalCopyAudit | None] = ContextVar("logical_copy_audit", default=None)
_COPY_PHASE: ContextVar[str | None] = ContextVar("condition_copy_phase", default=None)
_COPY_OPERATORS = frozenset(("aten::_to_copy", "aten::clone", "aten::copy_"))


@contextmanager
def condition_copy_phase(name: str) -> Iterator[None]:
    """Identify a real adapter encoder/cache call, without altering its
    inputs."""
    if name not in {"frozen_encoder", "cache_read", "cache_write"}:
        raise ValueError("Unknown fixed condition-copy phase")
    token = _COPY_PHASE.set(name)
    try:
        yield
    finally:
        _COPY_PHASE.reset(token)


def merge_repeated_transfer_evidence(audits: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge retained fingerprints from the SAME fixed workload's real runs.

    This can identify repeated online A1/A2 transfers across process
    boundaries. It does not extrapolate unobserved traffic or make it a steady-
    state metric.
    """
    entries: dict[str, dict[str, Any]] = {}
    for audit in audits:
        for row in audit.get("fingerprints", []):
            current = entries.setdefault(row["sha256"], {**row, "calls": 0})
            if current["destination_bytes_per_call"] != row["destination_bytes_per_call"]:
                raise ValueError("Inconsistent transfer fingerprint encoding")
            current["calls"] += row["calls"]
    return {
        "scope": "retained source-content-identical condition transfers across the supplied fixed runs; partial logical bytes",
        "runs": len(audits),
        "unique_fingerprints": len(entries),
        "repeated_transfer_calls": sum(max(0, row["calls"] - 1) for row in entries.values()),
        "repeated_destination_bytes": sum(
            max(0, row["calls"] - 1) * row["destination_bytes_per_call"] for row in entries.values()
        ),
        "complete_traffic_measurement": False,
    }


@contextmanager
def suspend_logical_copy_audit() -> Iterator[None]:
    """Exclude numerical-evidence copies, without changing their execution."""
    audit = _ACTIVE_AUDIT.get()
    if audit is None:
        yield
        return
    audit._suspended += 1
    try:
        yield
    finally:
        audit._suspended -= 1


def _plain_dense_tensor(value: Any) -> bool:
    # Counting a wrapper's global shape (e.g. DTensor) as a local transfer would
    # overstate volume and may double-count its redispatched local operation.
    # Only dispatcher-visible concrete tensor operations are measured here.
    return type(value) in (torch.Tensor, torch.nn.Parameter) and value.layout == torch.strided and not value.is_meta


def _same_elements(source: torch.Tensor, destination: torch.Tensor) -> bool:
    """A self-copy, including an identical overlapping view, writes no data."""
    return (
        source.device == destination.device
        and source.dtype == destination.dtype
        and source.shape == destination.shape
        and source.stride() == destination.stride()
        and source.data_ptr() == destination.data_ptr()
    )


class LogicalCopyAudit(TorchDispatchMode):
    """Observe one eager correctness update; retain counters, never tensors.

    ``logical_bytes`` counts destination payload bytes once per successful
    copy. Separate source-equivalent and destination byte counts describe dtype
    casts and broadcast copy_; neither is a measured memory-bus transaction
    count.
    """

    def __init__(
        self,
        *,
        collect_repeated_transfers: bool = False,
        max_fingerprint_source_bytes: int = 32 << 20,
        max_total_hashed_bytes: int = 512 << 20,
        max_fingerprints: int = 1024,
        fingerprint_directions: tuple[tuple[str, str], ...] = (("cpu", "cuda"),),
    ) -> None:
        super().__init__()
        self._suspended = 0
        self._token = None
        self._entered = False
        self._operations: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(
            lambda: {
                "calls": 0,
                "logical_bytes": 0,
                "source_equivalent_bytes": 0,
                "destination_bytes": 0,
            }
        )
        self._skipped = {
            "wrapper_or_unsupported_tensor": 0,
            "empty_or_self_copy": 0,
            "suspended_copy_calls": 0,
        }
        self._completed = False
        self._collect_repeated = collect_repeated_transfers
        self._fingerprint_source_limit = max_fingerprint_source_bytes
        self._hash_limit = max_total_hashed_bytes
        self._entry_limit = max_fingerprints
        if min(max_fingerprint_source_bytes, max_total_hashed_bytes, max_fingerprints) < 0:
            raise ValueError("Fingerprint limits must be nonnegative")
        self._fingerprint_directions = frozenset(fingerprint_directions)
        self._fingerprints: dict[str, dict[str, Any]] = {}
        self._hashed_bytes = 0
        self._fingerprint_skips: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "destination_bytes": 0})

    def _fingerprint_copy(self, source: torch.Tensor, destination: torch.Tensor) -> None:
        phase = _COPY_PHASE.get()
        size = destination.numel() * destination.element_size()

        def skip(reason: str) -> None:
            self._fingerprint_skips[reason]["calls"] += 1
            self._fingerprint_skips[reason]["destination_bytes"] += size

        if phase is None:
            skip("outside_condition_phase")
            return
        direction = (source.device.type, destination.device.type)
        if source.device.type != "cpu" or direction not in self._fingerprint_directions:
            skip("outside_declared_cpu_source_directions")
            return
        source_bytes = source.numel() * source.element_size()
        if source_bytes > self._fingerprint_source_limit:
            skip("per_source_byte_limit")
            return
        if self._hashed_bytes + source_bytes > self._hash_limit:
            skip("total_hashed_byte_limit")
            return
        metadata = {
            "direction": f"{source.device}->{destination.device}",
            "source_dtype": str(source.dtype),
            "source_shape": list(source.shape),
            "destination_dtype": str(destination.dtype),
            "destination_shape": list(destination.shape),
        }
        # __torch_dispatch__ has popped this mode. These CPU-only diagnostic
        # reads are excluded from observed training copies and consume no RNG.
        try:
            payload = source.detach().contiguous().view(torch.uint8).numpy().tobytes()
            digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode() + b"\0" + payload).hexdigest()
        except (RuntimeError, TypeError, ValueError):
            skip("unsupported_cpu_payload")
            return
        self._hashed_bytes += source_bytes
        if digest not in self._fingerprints:
            if len(self._fingerprints) >= self._entry_limit:
                skip("fingerprint_entry_limit")
                return
            self._fingerprints[digest] = {
                "sha256": digest,
                **metadata,
                "source_payload_bytes": source_bytes,
                "destination_bytes_per_call": size,
                "calls": 0,
                "phases": {},
            }
        row = self._fingerprints[digest]
        row["calls"] += 1
        row["phases"][phase] = row["phases"].get(phase, 0) + 1

    def __enter__(self) -> LogicalCopyAudit:
        if self._entered or _ACTIVE_AUDIT.get() is not None:
            raise RuntimeError("Use one fresh LogicalCopyAudit for one correctness update")
        super().__enter__()
        self._entered = True
        self._token = _ACTIVE_AUDIT.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            super().__exit__(exc_type, exc_value, traceback)
        finally:
            _ACTIVE_AUDIT.reset(self._token)
            self._token = None
            self._completed = exc_type is None

    def __torch_dispatch__(self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: Any = None) -> Any:
        kwargs = {} if kwargs is None else kwargs
        # Calling the identical operator while this mode is popped preserves
        # native autograd / autocast / subclass dispatch and exception behavior.
        result = func(*args, **kwargs)
        name = func._schema.name
        if name not in _COPY_OPERATORS:
            return result
        if self._suspended:
            self._skipped["suspended_copy_calls"] += 1
            return result
        source = args[0] if args else kwargs.get("self")
        if name == "aten::copy_":
            source = args[1] if len(args) > 1 else kwargs.get("src")
        destination = result
        if not _plain_dense_tensor(source) or not _plain_dense_tensor(destination):
            self._skipped["wrapper_or_unsupported_tensor"] += 1
            return result
        count = destination.numel()
        if not count or _same_elements(source, destination):
            self._skipped["empty_or_self_copy"] += 1
            return result
        direction = f"{source.device}->{destination.device}"
        key = (str(func), direction, str(source.dtype), str(destination.dtype))
        row = self._operations[key]
        row["calls"] += 1
        row["source_equivalent_bytes"] += count * source.element_size()
        row["destination_bytes"] += count * destination.element_size()
        row["logical_bytes"] += count * destination.element_size()
        if self._collect_repeated:
            self._fingerprint_copy(source, destination)
        return result

    def result(self) -> dict[str, Any]:
        rows = [
            {
                "operator": op,
                "direction": direction,
                "source_dtype": source,
                "destination_dtype": target,
                **counts,
            }
            for (op, direction, source, target), counts in sorted(self._operations.items())
        ]
        directions: dict[str, int] = defaultdict(int)
        for row in rows:
            directions[row["direction"]] += row["logical_bytes"]
        result = {
            "schema": 1,
            "completed_without_exception": self._completed,
            "scope": "current-thread dispatcher-visible concrete aten._to_copy/clone/copy_; correctness only",
            "metric": "logical destination payload bytes, not physical PCIe or memory-bus traffic",
            "excludes": [
                "views and identity/self copies",
                "numerical-evidence copies inside suspend_logical_copy_audit",
                "wrapper-level tensors, background threads, collectives and backend-internal/compiled copies",
            ],
            "calls": sum(row["calls"] for row in rows),
            "logical_bytes": sum(row["logical_bytes"] for row in rows),
            "logical_bytes_by_direction": dict(directions),
            "operations": rows,
            "skipped": dict(self._skipped),
        }
        if self._collect_repeated:
            fingerprints = [self._fingerprints[key] for key in sorted(self._fingerprints)]
            result["repeated_transfers"] = {
                "enabled": True,
                "scope": "actual condition-scoped copies with byte-identical CPU source plus identical shape/dtype transfer encoding",
                "metric": "logical destination bytes after the first identical observed source transfer; not physical PCIe bytes",
                "directions": [list(direction) for direction in sorted(self._fingerprint_directions)],
                "cuda_value_readbacks_for_audit": 0,
                "hashed_cpu_source_bytes": self._hashed_bytes,
                "limits": {
                    "per_source_bytes": self._fingerprint_source_limit,
                    "total_hashed_bytes": self._hash_limit,
                    "fingerprint_entries": self._entry_limit,
                },
                "fingerprints": fingerprints,
                "repeated_transfer_calls": sum(max(0, row["calls"] - 1) for row in fingerprints),
                "repeated_destination_bytes": sum(
                    max(0, row["calls"] - 1) * row["destination_bytes_per_call"] for row in fingerprints
                ),
                "unfingerprinted": dict(self._fingerprint_skips),
                "complete_traffic_measurement": False,
                "diagnostic_overhead": "CPU source hashing in correctness only; never installed in measured performance",
            }
        return result
