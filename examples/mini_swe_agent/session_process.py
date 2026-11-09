#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Linux per-session process ownership, independent of cgroup version.

The supervisor stays alive until its worker, named instance and all adopted
descendants have settled. Only ECHILD is a positive child-release proof.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from session_memory import SessionMemory


def enable_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")


def reap_exited_children(worker_pid: int) -> None:
    """Reap daemonized exits while the worker is still verifying instance stop.

    Apptainer treats a zombie instance PID as still present. Waiting until the
    worker exits would make its cleanup falsely fail. Never steal the Popen
    worker's status and never terminate live children here.
    """
    for child in Path(f"/proc/self/task/{os.getpid()}/children").read_text().split():
        pid = int(child)
        if pid == worker_pid:
            continue
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass


def reap_children(deadline: float) -> bool:
    """Terminate only our direct, adopted children and wait for ECHILD."""
    while time.monotonic() < deadline:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return True
        if pid:
            continue
        # WNOHANG=0 means at least one child is still alive, not clean.
        children = Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()
        for child in children:
            try:
                # Parentage is the authority; PID reuse cannot adopt an unrelated process.
                status = Path(f"/proc/{child}/status").read_text().splitlines()
                ppid = next(int(line.split()[1]) for line in status if line.startswith("PPid:"))
                if ppid == os.getpid():
                    os.kill(int(child), signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.02)
    return False


def instance_absent(spec: dict, deadline: float, *, stop: bool) -> bool:
    executable, global_args = spec["apptainer_args"]
    name = spec["instance_name"]

    def command(argv: list[str]) -> subprocess.CompletedProcess:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Instance cleanup deadline elapsed")
        return subprocess.run(
            argv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=min(remaining, 12.0),
            check=False,
        )

    if stop:
        command([executable, *global_args, "instance", "stop", "--timeout", "10", name])
    result = command([executable, *global_args, "instance", "list"])
    if result.returncode:
        raise RuntimeError("Cannot verify the session's Apptainer instance")
    return not any(line.split() and line.split()[0] == name for line in result.stdout.splitlines()[1:])


def supervise(spec_path: Path, report_path: Path, *, worker_command: list[str] | None = None) -> None:
    cancelled_at: float | None = None
    parent_pid = os.getppid()

    def cancel(_signum, _frame):
        nonlocal cancelled_at
        if cancelled_at is None:
            cancelled_at = time.monotonic()

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    enable_subreaper()
    spec = json.loads(spec_path.read_text())
    parent_pid = spec.get("server_pid", parent_pid)
    if os.getppid() != parent_pid:
        cancel(0, None)
    result_path = report_path.with_suffix(".worker.json")
    command = worker_command or [
        sys.executable,
        str(Path(__file__).with_name("agent_server.py")),
        "--session-worker",
        str(spec_path),
        str(result_path),
    ]
    memory = SessionMemory.create(spec["memory_limit_mib"]) if spec.get("memory_limit_mib", 0) else None
    memory_evidence: dict = {}
    try:
        worker = subprocess.Popen(memory.wrap(command) if memory else command, start_new_session=True)
    except BaseException:
        if memory:
            memory.remove_if_empty()
        raise
    sent_term = False
    supervisor_error = ""
    next_memory_poll = 0.0
    try:
        while worker.poll() is None:
            reap_exited_children(worker.pid)
            if memory and time.monotonic() >= next_memory_poll:
                next_memory_poll = time.monotonic() + 0.2
                if memory.oom_kills():
                    cancel(0, None)
            if os.getppid() != parent_pid:
                cancel(0, None)
            if cancelled_at is not None:
                try:
                    if not sent_term:
                        os.killpg(worker.pid, signal.SIGTERM)
                        sent_term = True
                    if time.monotonic() - cancelled_at >= 12.0:
                        os.killpg(worker.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(0.02)
    except BaseException as exc:
        supervisor_error = f"{type(exc).__name__}: {exc}"
    finally:
        if worker.poll() is None:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        worker.wait()
    deadline = (cancelled_at + 35.0) if cancelled_at is not None else time.monotonic() + 25.0
    error = supervisor_error
    absent = False
    try:
        absent = instance_absent(spec, deadline, stop=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    children_reaped = reap_children(deadline)
    if children_reaped:
        try:
            absent = instance_absent(spec, deadline, stop=False)
        except Exception as exc:
            absent = False
            error = f"{type(exc).__name__}: {exc}"
        # The verification subprocess was also wait()ed, but verify the invariant again.
        children_reaped = reap_children(deadline)
    result = {"info": {}, "error": f"Session worker exited with code {worker.returncode} without a valid result"}
    try:
        candidate = json.loads(result_path.read_text())
        if not isinstance(candidate, dict) or not isinstance(candidate.get("info"), dict):
            raise ValueError("Worker result has invalid info")
        if not isinstance(candidate.get("error"), str):
            raise ValueError("Worker result has invalid error")
        result = candidate
    except (OSError, ValueError) as exc:
        result["error"] += f": {type(exc).__name__}: {exc}"
    if not result["error"]:
        if worker.returncode != 0:
            result["error"] = f"Session worker exited with code {worker.returncode}"
        elif cancelled_at is not None:
            result["error"] = "Session supervisor cancelled the worker"
        elif supervisor_error:
            result["error"] = supervisor_error
    memory_released = memory is None
    if memory:
        try:
            memory_evidence = memory.snapshot()
            memory_released = memory.remove_if_empty()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        if memory_evidence.get("oom_kills", 0):
            # A descendant can be killed while its parent writes a success result.
            # Treat the whole session as failed, preserving group replacement.
            result["error"] = f"SessionMemoryLimitExceeded: {memory_evidence}"
    report = {
        **result,
        "nonce": spec["release_nonce"],
        "client_session_id": spec["record"]["session_id"],
        "resources_released": absent and children_reaped and memory_released,
        "children_reaped": children_reaped,
        "instance_absent": absent,
        "cleanup_error": error,
        "memory": {**memory_evidence, "released": memory_released},
    }
    temporary = report_path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(report))
        temporary.replace(report_path)
    finally:
        # A deadline or failed result write is not permission to orphan descendants.
        while not children_reaped:
            children_reaped = reap_children(time.monotonic() + 5.0)
        while memory and not memory_released:
            memory_released = memory.remove_if_empty()
            if not memory_released:
                time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("report", type=Path)
    arguments = parser.parse_args()
    supervise(arguments.spec, arguments.report)
