# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Confirm exit of a server and its workers before returning GPU ownership."""

from typing import Any, Callable


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
