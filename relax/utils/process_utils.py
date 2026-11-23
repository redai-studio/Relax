# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Process exit checks shared by local shutdown and dead-actor recovery."""

import os
import time

import psutil


def process_is_live(process: psutil.Process) -> bool:
    """Zombies have exited and no longer own CUDA contexts or executable
    work."""
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def server_session_exited(session_id: int, pid_namespace: int | None = None) -> bool:
    """Confirm exit on the server's node, independently of its Ray actor.

    Servers start a new session before launching backend children. Reparenting
    does not change session membership. A reused session id conservatively
    prevents recovery; this check never signals another process.
    """
    if pid_namespace is not None and os.stat("/proc/self/ns/pid").st_ino != pid_namespace:
        return False
    deadline = time.monotonic() + 5
    empty_scans = 0
    while True:
        live = False
        for process in psutil.process_iter():
            try:
                if os.getsid(process.pid) == session_id and process_is_live(process):
                    live = True
                    break
            except ProcessLookupError:
                continue
        empty_scans = 0 if live else empty_scans + 1
        # /proc enumeration is a snapshot: a parent may fork and exit during
        # a scan. Re-enumerate after observing an empty session.
        if empty_scans >= 2:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def check_server_session_exit(process_info: dict[str, int | str], *, timeout: float) -> bool:
    """Check an engine session from its original Ray node and PID namespace."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    node_id = process_info.get("node_id")
    session_id = process_info.get("session_id")
    namespace = process_info.get("pid_namespace")
    if (
        not isinstance(node_id, str)
        or not node_id
        or type(session_id) is not int
        or session_id <= 0
        or type(namespace) is not int
        or namespace <= 0
    ):
        return False
    probe = ray.remote(num_cpus=0)(server_session_exited).options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
    )
    return ray.get(probe.remote(session_id, namespace), timeout=timeout)
