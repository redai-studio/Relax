#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Optional Linux cgroup-v1 RAM budget for one supervised Mini SWE session.

The supervisor stays outside the private leaf. The worker enters before exec,
so container setup, tools, tests and their descendants share one aggregate cap.
Enabling this on an unsupported or non-delegated host fails before serving
work.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import uuid
from pathlib import Path


def memory_parent() -> Path:
    membership = next(
        (
            line.split(":", 2)[2]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if "memory" in line.split(":", 2)[1].split(",")
        ),
        None,
    )
    if membership is None:
        raise RuntimeError("Session memory isolation requires a delegated cgroup-v1 memory controller")
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        filesystem, _source, options = right.split()[:3]
        if filesystem != "cgroup" or "memory" not in options.split(","):
            continue
        fields = left.split()
        root, mount = (re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value) for value in fields[3:5])
        try:
            relative = Path(membership).relative_to(root)
        except ValueError:
            continue
        parent = Path(mount) / relative
        if (parent / "memory.use_hierarchy").read_text().strip() != "1":
            raise RuntimeError("Session memory isolation requires hierarchical accounting")
        return parent
    raise RuntimeError("Cannot resolve this process's memory cgroup within its mounted hierarchy")


class SessionMemory:
    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def create(cls, limit_mib: int) -> SessionMemory:
        if limit_mib <= 0:
            raise ValueError("Session memory limit must be positive")
        path = memory_parent() / f"mswe-{os.getpid()}-{uuid.uuid4().hex}"
        path.mkdir()
        try:
            limit = limit_mib * 1024 * 1024
            (path / "memory.limit_in_bytes").write_text(str(limit))
            # Disallow swapping beyond the RAM budget, even on hosts with swap.
            (path / "memory.memsw.limit_in_bytes").write_text(str(limit))
            for name in ("memory.limit_in_bytes", "memory.memsw.limit_in_bytes"):
                if int((path / name).read_text()) != limit:
                    raise RuntimeError(f"Session memory limit readback failed: {name}")
            return cls(path)
        except BaseException:
            path.rmdir()
            raise

    def oom_kills(self) -> int:
        oom = dict(line.split() for line in (self.path / "memory.oom_control").read_text().splitlines())
        return int(oom["oom_kill"])

    def snapshot(self) -> dict:
        # failcnt and exit 137 alone are not proof that the memory controller killed a process.
        return {
            "cgroup": str(self.path),
            "limit_bytes": int((self.path / "memory.limit_in_bytes").read_text()),
            "peak_bytes": int((self.path / "memory.max_usage_in_bytes").read_text()),
            "oom_kills": self.oom_kills(),
            "limit_hits": int((self.path / "memory.failcnt").read_text()),
        }

    def remove_if_empty(self) -> bool:
        if not self.path.exists():
            return True
        if (self.path / "cgroup.procs").read_text().strip():
            return False
        self.path.rmdir()
        return True

    def wrap(self, command: list[str]) -> list[str]:
        return [sys.executable, str(Path(__file__).resolve()), str(self.path), *command]


def validate_memory_limit(limit_mib: int, *, supervised: bool) -> None:
    if limit_mib < 0 or (limit_mib and not supervised):
        raise ValueError("AGENT_SERVER_SESSION_MEMORY_MIB must be nonnegative and requires supervised sessions")
    if limit_mib:
        group = SessionMemory.create(limit_mib)
        try:
            group.snapshot()
        finally:
            if not group.remove_if_empty():
                raise RuntimeError("Session memory startup probe unexpectedly contains processes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cgroup", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("A worker command is required")
    (args.cgroup / "cgroup.procs").write_text(str(os.getpid()))
    os.execvpe(args.command[0], args.command, os.environ)
