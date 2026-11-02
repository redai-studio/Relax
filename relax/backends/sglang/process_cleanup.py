# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Confirm exit of a server and its workers before returning GPU ownership."""

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable


SERVER_INCARNATION_ENV = "RELAX_SGLANG_SERVER_INCARNATION"


def _spawn_cpu_executor(_processor: Any) -> ProcessPoolExecutor:
    # SGLang 0.5.21 otherwise explicitly forks for cpu/cuda_ipc transport.
    # Spawn also avoids inheriting the tokenizer's CUDA IPC pool into workers.
    return ProcessPoolExecutor(
        mp_context=multiprocessing.get_context("spawn"),
        max_workers=int(os.environ.get("SGLANG_CPU_WORKERS", os.cpu_count())),
    )


def inherit_server_incarnation() -> None:
    """Install the generation only inside the managed server spawn child."""
    token = getattr(multiprocessing.current_process(), "_relax_server_incarnation", None)
    if token is None:
        return
    os.environ[SERVER_INCARNATION_ENV] = token
    # SGLang workers call setproctitle; its default overwrites /proc/environ.
    # Let their exec environment retain the membership marker after renaming.
    os.environ["SPT_NOENV"] = "1"
    # /proc/environ sees exec-time state. A fork of this first server would
    # miss the token set above, so keep its explicit multimodal pool on spawn.
    from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

    BaseMultimodalProcessor._create_cpu_executor = _spawn_cpu_executor


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
        **(
            {"incarnation": process._relax_server_incarnation}
            if getattr(process, "_relax_server_incarnation", None) is not None
            else {}
        ),
    }


def cleanup_server_identity(identity: dict, timeout: float = 10.0) -> None:
    """Node-local exit proof; never signal a PID belonging to a new process."""
    import psutil

    records = identity.get("processes")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Missing server process-tree identity")
    token = identity.get("incarnation")
    deadline = time.monotonic() + timeout

    def members() -> list:
        processes = {}
        for record in records:
            try:
                process = psutil.Process(record["pid"])
                if process.create_time() != record["create_time"]:
                    continue
                processes[process.pid] = process
                for child in process.children(recursive=True):
                    processes[child.pid] = child
            except psutil.NoSuchProcess:
                continue
        if token:
            # Membership survives both an early snapshot and reparenting.
            for process in psutil.process_iter():
                try:
                    if process.environ().get(SERVER_INCARNATION_ENV) == token:
                        processes[process.pid] = process
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
        return list(processes.values())

    def live(processes: list) -> list:
        result = []
        for process in processes:
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    result.append(process)
            except psutil.NoSuchProcess:
                pass
        return result

    processes = members()
    while True:
        for process in reversed(processes):
            try:
                process.kill()  # psutil also guards against PID reuse after lookup.
            except psutil.NoSuchProcess:
                pass
        remaining = max(0.0, deadline - time.monotonic())
        _, alive = psutil.wait_procs(live(processes), timeout=min(0.1, remaining) if token else remaining)
        # A worker may have spawned between enumeration and killing its parent.
        # Re-scan in the same budget after stopping the known members.
        # An unreadable later scan cannot revoke a known member's still-live
        # identity. Retain it until exit is confirmed, even after reparenting.
        processes = list({p.pid: p for p in live([*alive, *members()])}.values()) if token else live(alive)
        if not processes:
            return
        if not token or time.monotonic() >= deadline:
            raise RuntimeError(f"Server worker exit was not confirmed: {[p.pid for p in processes]}")


def terminate_server(
    process: Any, kill_tree: Callable[[int], None], timeout: float = 10.0, *, identity: dict | None = None
) -> None:
    import psutil

    if process is None or process.pid is None:
        return
    if identity is not None:
        # A saved tree includes workers reparented after the server died. Its
        # creation times also prevent signalling a reused parent PID.
        cleanup_server_identity(identity, timeout=timeout)
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
