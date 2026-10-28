# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exit evidence for managed SGLang process sessions on Linux.

The managed launcher and its workers inherit one session/process group.
External engines and launchers that detach workers from that session are not
covered. Probes never signal processes or interpret an unreachable node as
clean.
"""

import os
from pathlib import Path
from typing import Any, Callable


def _local_identity() -> dict[str, str | int]:
    return {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": os.stat("/proc/self/ns/pid").st_ino,
    }


def _process_stat(pid: int) -> tuple[int, int]:
    # comm may contain spaces and parentheses; fields after its final ')' start
    # with state (field 3). Session is field 6; starttime is field 22.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[3]), int(fields[19])


def run_in_backend_session(target: Callable, args: tuple, ready: Any) -> None:
    """Establish the session before the backend can spawn any workers."""
    try:
        os.setsid()
        pid = os.getpid()
        session_id, start_time = _process_stat(pid)
        ready.send({**_local_identity(), "session_id": session_id, "pid": pid, "start_time": start_time})
    finally:
        ready.close()
    target(*args)


def backend_session_exited(identity: dict[str, Any]) -> bool:
    """Confirm absence, including orphaned workers; uncertainty stays
    fenced."""
    try:
        if not isinstance(identity, dict):
            return False
        if any(identity.get(key) != value for key, value in _local_identity().items()):
            return False
        pid, session_id, start_time = (identity.get(key) for key in ("pid", "session_id", "start_time"))
        if any(type(value) is not int or value <= 0 for value in (pid, session_id, start_time)):
            return False
        if pid != session_id:
            return False
        try:
            # A reused PID is not evidence about the recorded generation.
            if _process_stat(pid) != (session_id, start_time):
                return False
            return False
        except FileNotFoundError:
            pass
        try:
            # Kernel check covers workers forked during /proc enumeration in the
            # inherited group, even when their original parent has disappeared.
            os.killpg(session_id, 0)
            return False
        except ProcessLookupError:
            pass
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                member_session, _ = _process_stat(int(entry.name))
            except FileNotFoundError:
                continue
            if member_session == session_id:
                return False
        return True
    except (OSError, ValueError, IndexError):
        return False


def _probe_backend_session(node_id: str, identity: dict[str, Any]) -> bool:
    import ray

    return ray.get_runtime_context().get_node_id() == node_id and backend_session_exited(identity)


def confirm_backend_session_exited(node_id: str, identity: dict[str, Any]) -> bool:
    """Probe the original node without consuming a GPU or a PG bundle."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    if not node_id or not isinstance(identity, dict):
        return False
    ref = None
    try:
        task = ray.remote(num_cpus=0, max_retries=0)(_probe_backend_session)
        ref = task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)).remote(
            node_id, identity
        )
        return ray.get(ref, timeout=10.0) is True
    except Exception:
        if ref is not None:
            try:
                ray.cancel(ref, force=True)
            except Exception:
                pass
        return False
