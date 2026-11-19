# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded raw-byte oracle for sharded SGLang GPU parameters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


_CHUNK_BYTES = 16 * 1024 * 1024
_SUPPORTED_DTYPES = {torch.bfloat16, torch.float32}


class AuditState(str, Enum):
    READY = "READY"
    CAPTURED = "CAPTURED"
    REFERENCE_PENDING = "REFERENCE_PENDING"
    POISONED = "POISONED"
    CLOSED = "CLOSED"


@dataclass
class _StorageEntry:
    key: tuple[str, int]
    start: int
    end: int
    names: list[str]
    parameters: list[torch.nn.Parameter]
    snapshot: torch.Tensor


def _storage_key(tensor: torch.Tensor) -> tuple[str, int]:
    return str(tensor.device), int(tensor.untyped_storage()._cdata)


def _byte_bounds(tensor: torch.Tensor) -> tuple[int, int]:
    if not tensor.is_contiguous():
        raise ValueError("Audit parameters must be contiguous")
    start = tensor.storage_offset() * tensor.element_size()
    return start, start + tensor.numel() * tensor.element_size()


def _fence(devices: set[torch.device]) -> None:
    for device in sorted(devices, key=str):
        if device.type == "cuda":
            torch.cuda.synchronize(device)


class AuditSnapshot:
    """Reusable per-model CPU snapshot and fail-closed raw GPU oracle.

    ``reserve`` and ``release`` update a caller-owned per-node memory lease;
    ``allocated`` reports its current allocation. The snapshot itself is
    limited by ``lease_bytes`` and retained until explicit ``close()``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lease_bytes: int,
        reserve: Callable[[int], None] | None = None,
        allocated: Callable[[], None] | None = None,
        release: Callable[[int], None] | None = None,
    ) -> None:
        if type(lease_bytes) is not int or lease_bytes < 0:
            raise ValueError("lease_bytes must be a nonnegative integer")
        self._model = model
        self._lease_bytes = lease_bytes
        self._reserve = reserve
        self._allocated = allocated
        self._release = release
        if any(callback is not None for callback in (reserve, allocated, release)) and not all(
            callback is not None for callback in (reserve, allocated, release)
        ):
            raise ValueError("reserve, allocated, and release callbacks must be provided together")
        self._entries: list[_StorageEntry] = []
        self._empty_parameters: list[tuple[str, torch.nn.Parameter]] = []
        self._state = AuditState.READY
        self._update_id: str | None = None
        self._bytes = 0
        self._build(model)
        self._signatures = self._current_signatures()
        scratch_bytes = min(_CHUNK_BYTES, max(1, self._bytes))
        # One D2H chunk plus its CPU comparison mask. argmax locates the
        # first different byte without a potentially full int64 nonzero array.
        self._reserved_bytes = self._bytes + 2 * scratch_bytes
        if self._reserved_bytes > lease_bytes:
            raise MemoryError(f"Audit snapshot requires {self._bytes} bytes, lease allows {lease_bytes}")
        if reserve is not None:
            reserve(self._reserved_bytes)
        try:
            for entry in self._entries:
                entry.snapshot = torch.empty(entry.end - entry.start, dtype=torch.uint8, device="cpu")
                entry.snapshot.zero_()  # Touch the allocation before the audit begins.
            self._scratch = torch.empty(scratch_bytes, dtype=torch.uint8, device="cpu")
            self._scratch.zero_()
            if allocated is not None:
                allocated()
        except BaseException:
            self._entries.clear()
            if release is not None:
                release(self._reserved_bytes)
            raise

    @property
    def state(self) -> AuditState:
        return self._state

    @property
    def snapshot_bytes(self) -> int:
        return self._bytes

    def _build(self, model: torch.nn.Module) -> None:
        named = list(model.named_parameters(remove_duplicate=False))
        buffers = list(model.named_buffers(remove_duplicate=False))
        persistent = set(model.state_dict().keys())
        persistent_buffers = [name for name, _ in buffers if name in persistent]
        if persistent_buffers:
            raise ValueError(f"Persistent buffers are unsupported by the GPU audit: {persistent_buffers}")

        grouped: dict[tuple[str, int], list[tuple[int, int, str, torch.nn.Parameter]]] = {}
        for name, parameter in named:
            if not isinstance(parameter, torch.nn.Parameter):
                raise ValueError(f"Expected a parameter at {name}")
            if parameter.dtype not in _SUPPORTED_DTYPES:
                raise ValueError(f"Unsupported audit dtype at {name}: {parameter.dtype}")
            if not parameter.is_contiguous():
                raise ValueError(f"Non-contiguous audit parameter: {name}")
            start, end = _byte_bounds(parameter)
            if start == end:
                self._empty_parameters.append((name, parameter))
                continue
            grouped.setdefault(_storage_key(parameter), []).append((start, end, name, parameter))

        for key, intervals in grouped.items():
            intervals.sort(key=lambda item: (item[0], item[1], item[2]))
            unique: list[tuple[int, int, list[str], list[torch.nn.Parameter]]] = []
            for start, end, name, parameter in intervals:
                if unique and start < unique[-1][1]:
                    previous = unique[-1]
                    if (start, end) != (previous[0], previous[1]):
                        raise ValueError(f"Partially overlapping parameter storage: {previous[2]} and {name}")
                    if parameter.dtype != previous[3][0].dtype or parameter.shape != previous[3][0].shape:
                        raise ValueError("Exact storage aliases must also have the same dtype and shape")
                    previous[2].append(name)
                    previous[3].append(parameter)
                else:
                    unique.append((start, end, [name], [parameter]))
            for start, end, names, parameters in unique:
                byte_count = end - start
                snapshot = torch.empty(0, dtype=torch.uint8, device="cpu")
                self._entries.append(_StorageEntry(key, start, end, names, parameters, snapshot))
                self._bytes += byte_count
        self._entries.sort(key=lambda entry: (entry.key, entry.start, entry.end, entry.names[0]))

    def inventory(self) -> dict[str, Any]:
        """Return parameter and nonpersistent-buffer layout without tensor
        data."""
        persistent = set(self._model.state_dict().keys())
        parameters = []
        for entry in self._entries:
            first = entry.parameters[0]
            parameters.append(
                {
                    "names": sorted(set(entry.names)),
                    "shape": list(first.shape),
                    "dtype": str(first.dtype),
                    "device": str(first.device),
                    "bytes": entry.end - entry.start,
                    "storage_offset_bytes": entry.start,
                    "alias_count": len(set(entry.names)),
                }
            )
        for name, parameter in self._empty_parameters:
            parameters.append(
                {
                    "names": [name],
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "device": str(parameter.device),
                    "bytes": 0,
                    "storage_offset_bytes": parameter.storage_offset() * parameter.element_size(),
                    "alias_count": 0,
                }
            )
        buffers = [
            {
                "name": name,
                "shape": list(buffer.shape),
                "dtype": str(buffer.dtype),
                "device": str(buffer.device),
                "persistent": name in persistent,
            }
            for name, buffer in self._model.named_buffers(remove_duplicate=False)
        ]
        return {"parameters": parameters, "buffers": buffers, "snapshot_bytes": self._bytes}

    def capture(self, update_id: str) -> None:
        self._ensure_open()
        if not update_id:
            raise ValueError("update_id must be nonempty")
        if self._state in {AuditState.POISONED, AuditState.REFERENCE_PENDING}:
            raise RuntimeError("Cannot capture while a reference load is pending")
        try:
            self._check_storage()
            self._fence_all()
            for entry in self._entries:
                source = self._entry_bytes(entry)
                entry.snapshot.copy_(source, non_blocking=False)
            self._fence_all()
        except BaseException:
            self._state = AuditState.POISONED
            raise
        self._update_id = update_id
        self._state = AuditState.CAPTURED

    def compare(self, update_id: str) -> dict[str, Any]:
        self._ensure_open()
        self._check_update(update_id)
        if self._state not in {AuditState.CAPTURED, AuditState.REFERENCE_PENDING}:
            raise RuntimeError(f"Cannot compare snapshot in state {self._state.value}")
        compared = 0
        mismatched = 0
        first_difference: dict[str, Any] | None = None
        try:
            self._check_storage()
            self._fence_all()
            for entry in self._entries:
                actual = self._entry_bytes(entry)
                for offset in range(0, actual.numel(), _CHUNK_BYTES):
                    length = min(_CHUNK_BYTES, actual.numel() - offset)
                    staging = self._scratch[:length]
                    staging.copy_(actual[offset : offset + length], non_blocking=False)
                    expected = entry.snapshot[offset : offset + length]
                    unequal = staging != expected
                    count = int(torch.count_nonzero(unequal).item())
                    compared += length
                    mismatched += count
                    if count and first_difference is None:
                        index = int(unequal.view(torch.uint8).argmax().item())
                        first_difference = {
                            "names": sorted(set(entry.names)),
                            "byte_offset": offset + index,
                            "expected": int(expected[index]),
                            "actual": int(staging[index]),
                        }
                    del unequal
            self._fence_all()
        except BaseException:
            self._state = AuditState.POISONED
            raise
        if mismatched:
            self._state = AuditState.POISONED
            raise AssertionError(
                f"Raw GPU audit mismatch for {update_id}: {mismatched}/{compared} bytes; "
                f"first_difference={first_difference}"
            )
        if self._state == AuditState.REFERENCE_PENDING:
            self._state = AuditState.CAPTURED
        return {
            "update_id": update_id,
            "compared_bytes": compared,
            "mismatched_bytes": 0,
            "unique_storage_intervals": len(self._entries),
            "tensor_count": sum(len(entry.names) for entry in self._entries) + len(self._empty_parameters),
            "parameter_names": sum(len(set(entry.names)) for entry in self._entries) + len(self._empty_parameters),
            "aliases": sum(max(0, len(set(entry.names)) - 1) for entry in self._entries),
            "dtypes": sorted(
                {str(parameter.dtype) for entry in self._entries for parameter in entry.parameters}
                | {str(parameter.dtype) for _, parameter in self._empty_parameters}
            ),
        }

    def complement(self, update_id: str) -> None:
        """Poison live parameters after proving they match the captured
        delta."""
        self._ensure_open()
        self._check_update(update_id)
        if self._state != AuditState.CAPTURED:
            raise RuntimeError(f"Cannot complement snapshot in state {self._state.value}")
        self.compare(update_id)
        try:
            self._fence_all()
            with torch.no_grad():
                for entry in self._entries:
                    parameter = entry.parameters[0]
                    bit_dtype = torch.int16 if parameter.dtype == torch.bfloat16 else torch.int32
                    parameter.view(bit_dtype).bitwise_not_()
            self._fence_all()
        except BaseException:
            self._state = AuditState.POISONED
            raise
        self._state = AuditState.REFERENCE_PENDING

    def close(self) -> None:
        if self._state == AuditState.CLOSED:
            return
        if self._release is not None:
            self._release(self._reserved_bytes)
        self._entries.clear()
        self._scratch = torch.empty(0, dtype=torch.uint8)
        self._state = AuditState.CLOSED

    def _entry_bytes(self, entry: _StorageEntry) -> torch.Tensor:
        parameter = entry.parameters[0]
        return parameter.view(torch.uint8).reshape(-1)

    def _current_signatures(self) -> dict:
        return {
            name: (_storage_key(p), p.data_ptr(), p.storage_offset(), tuple(p.shape), tuple(p.stride()), p.dtype)
            for name, p in self._model.named_parameters(remove_duplicate=False)
        }

    def _check_storage(self) -> None:
        if self._current_signatures() != self._signatures:
            raise ValueError("Live model parameter storage or schema changed during the audit")

    def _fence_all(self) -> None:
        _fence({parameter.device for entry in self._entries for parameter in entry.parameters})

    def _check_update(self, update_id: str) -> None:
        if self._update_id != update_id:
            raise ValueError(f"Snapshot update mismatch: captured={self._update_id}, requested={update_id}")

    def _ensure_open(self) -> None:
        if self._state == AuditState.CLOSED:
            raise RuntimeError("AuditSnapshot is closed")


AUDIT_LOADER = "task_delta_weight_sync.sharded_gpu_audit.load_audit"


def load_audit(model: Any, weights: Any) -> None:
    """Explicit, paused-worker audit controls; no snapshot is written to
    disk."""
    import json
    import os
    import re
    import socket
    import time
    from pathlib import Path

    from relax.distributed.weight_sync.provenance import runtime_identity
    from relax.distributed.weight_sync.store import atomic_write
    from task_delta_weight_sync.audit_cpu_budget import AuditCpuBudget, host_memory_budget

    weights = list(weights)
    if len(weights) != 1 or weights[0][1].numel() != 0 or len(weights[0][0]) > 65536:
        raise ValueError("Audit controls require one bounded empty tensor")
    control = json.loads(weights[0][0])
    command, member, update_id = control["command"], control["member"], control["update_id"]
    if command not in {
        "prepare",
        "capture",
        "compare",
        "complement",
        "fence",
        "close",
        "arm_failure",
        "failure_state",
        "disarm_failure",
        "install_transfer",
        "allocator_reset",
        "allocator_metrics",
    }:
        raise ValueError("Unknown GPU audit command")
    observing = command in {"allocator_reset", "allocator_metrics"}
    if (control.get("paused") is not True and not observing) or any(
        not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.:-]{1,180}", value) is None
        for value in (member, update_id, control["phase"])
    ):
        raise ValueError("Audit requires a paused group and bounded identities")
    directory = Path(control["directory"]).resolve()
    task_run = Path(os.environ["TMPDIR"]).resolve().parent
    if not directory.is_relative_to(task_run):
        raise ValueError("Audit metadata must stay inside the task run directory")
    destination = directory / member
    report = dict(
        command=command,
        member=member,
        update_id=update_id,
        phase=control["phase"],
        time=time.time(),
        pid=os.getpid(),
        hostname=socket.gethostname(),
        tp_rank=control["tp_rank"],
    )
    snapshot = getattr(model, "_relax_cpu_audit_snapshot", None)
    receiver = getattr(model, "_relax_sharded_memory_receiver", None)
    if receiver is not None and hasattr(receiver, "last_metrics"):
        report["last_tcp_receiver_stages"] = receiver.last_metrics
    try:
        if observing:
            from task_delta_weight_sync.sync_metrics import allocator_metrics

            report.update(allocator_metrics(model, reset=command == "allocator_reset"))
        elif command == "fence":
            _fence({parameter.device for parameter in model.parameters()})
            return  # No audit filesystem IO inside a measured completion fence.
        elif command in {"arm_failure", "failure_state", "disarm_failure"}:
            from task_delta_weight_sync.recovery_probe import worker_control

            report.update(worker_control(model, control))
        elif command == "install_transfer":
            from task_delta_weight_sync.prepared_nccl import install_worker

            install_worker()
        elif command == "prepare":
            if snapshot is not None:
                raise ValueError("GPU audit snapshot is already prepared")
            atomic_write(destination / "host-memory.json", json.dumps(host_memory_budget(), indent=2).encode())
            budget = AuditCpuBudget(directory, member, limit_bytes=control["node_budget_bytes"])
            snapshot = AuditSnapshot(
                model,
                lease_bytes=control["node_budget_bytes"],
                reserve=budget.reserve,
                allocated=budget.allocated,
                release=lambda size: budget.release(),
            )
            model._relax_cpu_audit_snapshot = snapshot
            atomic_write(destination / "inventory.json", json.dumps(snapshot.inventory(), indent=2).encode())
            atomic_write(destination / "runtime.json", json.dumps(runtime_identity(), indent=2).encode())
        else:
            if snapshot is None:
                raise ValueError("Prepare the CPU audit snapshot before issuing data commands")
            if command == "capture":
                snapshot.capture(update_id)
            elif command == "compare":
                report.update(snapshot.compare(update_id))
            elif command == "complement":
                snapshot.complement(update_id)
            elif command == "close":
                if snapshot.state in {AuditState.REFERENCE_PENDING, AuditState.POISONED}:
                    raise RuntimeError("Preserve the failed audit snapshot until the group is discarded")
                snapshot.close()
        report.update(success=True, state=snapshot.state.value if snapshot is not None else None)
    except BaseException as exc:
        report.update(success=False, error=f"{type(exc).__name__}: {exc}")
        atomic_write(destination / f"{update_id}-{control['phase']}.json", json.dumps(report, indent=2).encode())
        raise
    atomic_write(destination / f"{update_id}-{control['phase']}.json", json.dumps(report, indent=2).encode())
