# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Node-wide CPU reservations for retained, in-memory GPU audit snapshots."""

from __future__ import annotations

import fcntl
import json
import re
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


def host_memory_budget(proc: Path = Path("/proc")) -> dict:
    """Read node availability and every visible ancestor cgroup constraint.

    Mount roots may expose only a subtree. Record that boundary rather than
    claiming invisible ancestors were checked. Missing accounting fails closed.
    """
    fields = dict(line.split(":", 1) for line in (proc / "meminfo").read_text().splitlines())
    available = int(fields["MemAvailable"].split()[0]) * 1024
    memberships = [line.split(":", 2) for line in (proc / "self/cgroup").read_text().splitlines()]
    constraints = []
    found = False
    for line in (proc / "self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        mount, filesystem = left.split(), right.split()
        kind = filesystem[0]
        if kind not in {"cgroup", "cgroup2"} or (kind == "cgroup" and "memory" not in filesystem[2].split(",")):
            continue
        paths = [
            path
            for _, controllers, path in memberships
            if (kind == "cgroup2" and not controllers) or (kind == "cgroup" and "memory" in controllers.split(","))
        ]
        if not paths:
            continue

        # mountinfo uses octal escaping for whitespace and backslashes.
        def unescape(value: str) -> str:
            return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)

        root, point = Path(unescape(mount[3])), Path(unescape(mount[4]))
        try:
            relative = Path(paths[0]).relative_to(root)
        except ValueError:
            continue  # An unrelated bind mount is not this process's hierarchy.
        directory = point / relative
        found = True
        limit_name, current_name = (
            ("memory.max", "memory.current")
            if kind == "cgroup2"
            else ("memory.limit_in_bytes", "memory.usage_in_bytes")
        )
        while True:
            limit_path = directory / limit_name
            # The v2 hierarchy root has accounting but no configurable limit.
            limit_text = (
                "max"
                if kind == "cgroup2" and directory == point and not limit_path.exists()
                else limit_path.read_text().strip()
            )
            current = int((directory / current_name).read_text().strip())
            limit = None if limit_text == "max" else int(limit_text)
            if kind == "cgroup" and limit is not None and limit >= 1 << 60:
                limit = None  # Linux v1 PAGE_COUNTER_MAX, not a usable quota.
            remaining = None if limit is None else max(0, limit - current)
            constraints.append(
                dict(
                    path=str(directory),
                    mount_root=str(root),
                    kind=kind,
                    limit_bytes=limit,
                    current_bytes=current,
                    remaining_bytes=remaining,
                )
            )
            if remaining is not None:
                available = min(available, remaining)
            if directory == point:
                break
            directory = directory.parent
    if not found:
        raise RuntimeError("No visible memory cgroup accounting; cannot validate audit capacity")
    return dict(
        effective_available_bytes=available,
        node_available_bytes=int(fields["MemAvailable"].split()[0]) * 1024,
        cgroups=constraints,
    )


def available_host_bytes() -> int:
    return host_memory_budget()["effective_available_bytes"]


class AuditCpuBudget:
    """Coordinate workers on one physical node before allocating CPU tensors.

    Allocated reservations count toward the configured total. Pending ones also
    count against MemAvailable, which already excludes resident buffers. Keep a
    fresh directory for each run; failed workers leave forensic leases.
    """

    def __init__(
        self,
        directory: Path,
        member: str,
        *,
        limit_bytes: int,
        margin_bytes: int = 64 << 30,
        available: Callable[[], int] = available_host_bytes,
    ) -> None:
        if not directory.is_absolute() or not member or type(limit_bytes) is not int or limit_bytes <= 0:
            raise ValueError("A task-owned absolute audit directory, member and positive budget are required")
        if type(margin_bytes) is not int or margin_bytes < 0:
            raise ValueError("Audit memory margin must be nonnegative")
        self.directory, self.member = directory, member
        self.limit_bytes, self.margin_bytes, self.available = limit_bytes, margin_bytes, available
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / ("cpu-budget-" + socket.gethostname() + ".json")
        self.lock = self.path.with_suffix(".lock")

    @contextmanager
    def _locked(self) -> Iterator[dict]:
        with self.lock.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else dict(limit_bytes=self.limit_bytes, leases={})
            )
            if state["limit_bytes"] != self.limit_bytes:
                raise ValueError("Node audit CPU budget changed within a run")
            yield state
            temporary = self.path.with_suffix(".partial")
            temporary.write_text(json.dumps(state, sort_keys=True))
            temporary.replace(self.path)

    def reserve(self, size: int) -> None:
        if type(size) is not int or size <= 0:
            raise ValueError("Audit snapshot reservation must be positive")
        with self._locked() as state:
            leases = state["leases"]
            if self.member in leases:
                raise RuntimeError("Audit member already owns a CPU reservation")
            if size + sum(item["bytes"] for item in leases.values()) > self.limit_bytes:
                raise ValueError("All workers' snapshots exceed the shared node CPU budget")
            pending = sum(item["bytes"] for item in leases.values() if not item["allocated"])
            if size + pending + self.margin_bytes > self.available():
                raise ValueError("Insufficient host MemAvailable for pending audit snapshots and margin")
            leases[self.member] = dict(bytes=size, allocated=False)

    def allocated(self) -> None:
        with self._locked() as state:
            state["leases"][self.member]["allocated"] = True

    def release(self) -> None:
        with self._locked() as state:
            state["leases"].pop(self.member, None)
