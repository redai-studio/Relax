# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Confirm exit of a server and its workers before returning GPU ownership."""

from typing import Any, Callable


class RouterCleanupError(RuntimeError):
    """Routing cleanup failed after the server tree confirmed exit."""

    released = True


def capture_server_identity(process: Any, *, node_id: str, port: int) -> dict:
    """Keep worker identities outside the actor, including reparented
    workers."""
    import psutil

    parent = psutil.Process(process.pid)
    processes = [parent, *parent.children(recursive=True)]
    return {
        "node_id": node_id,
        "server_pid": parent.pid,
        "port": port,
        "processes": [{"pid": p.pid, "create_time": p.create_time()} for p in processes],
    }


def cleanup_server_identity(identity: dict, timeout: float = 10.0) -> None:
    """Node-local exit proof; never signal a PID belonging to a new process."""
    import psutil

    records = identity.get("processes")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Missing server process-tree identity")
    processes = {}
    for record in records:
        try:
            process = psutil.Process(record["pid"])
            if process.create_time() != record["create_time"]:
                continue
            processes[process.pid] = process
            # Include children created after the original initialization receipt.
            for child in process.children(recursive=True):
                processes[child.pid] = child
        except psutil.NoSuchProcess:
            continue
    for process in reversed(list(processes.values())):
        try:
            process.kill()  # psutil also guards against PID reuse after lookup.
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(processes.values()), timeout=timeout)
    live = [p.pid for p in alive if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
    if live:
        raise RuntimeError(f"Server worker exit was not confirmed: {live}")


def terminate_server(process: Any, kill_tree: Callable[[int], None], timeout: float = 10.0) -> None:
    import psutil

    if process is None or process.pid is None:
        return
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess as error:
        # Once the parent is gone, its children may have been reparented. A
        # dead Ray actor or server PID is not evidence that they released GPU.
        raise RuntimeError("Server exited before its worker tree could be accounted for") from error
    kill_tree(process.pid)
    process.join(timeout=timeout)
    _, alive = psutil.wait_procs([*children, parent], timeout=timeout)
    live = [p.pid for p in alive if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
    if process.is_alive() or live:
        raise RuntimeError(f"Server worker exit was not confirmed: {live}")
