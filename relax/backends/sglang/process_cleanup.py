# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Confirm exit of a server and its workers before returning GPU ownership."""

import multiprocessing
import os
import signal
import time
from typing import Any, Callable

from relax.utils.logging_utils import get_logger


SERVER_INCARNATION_ENV = "RELAX_SGLANG_SERVER_INCARNATION"
logger = get_logger(__name__)


def inherit_server_incarnation() -> None:
    """Install the generation only inside the managed server spawn child."""
    token = getattr(multiprocessing.current_process(), "_relax_server_incarnation", None)
    if token is None:
        return
    os.environ[SERVER_INCARNATION_ENV] = token
    # SGLang workers call setproctitle; its default overwrites /proc/environ.
    # Let their exec environment retain the membership marker after renaming.
    os.environ["SPT_NOENV"] = "1"
    # The server's runtime token enters spawn/exec descendants' /proc/environ.
    # CPU preprocessors forked after this assignment may lack that exec-time
    # marker; preserve SGLang's pool mode. They cannot use the parent's
    # initialized CUDA context and hold no CUDA context of their own, so this
    # does not weaken GPU exit proof. Live-parent traversal still cleans them.


class RouterCleanupError(RuntimeError):
    """Routing cleanup failed after the server tree confirmed exit."""

    released = True


def capture_server_identity(process: Any, *, node_id: str, port: int, timeout: float = 5.0) -> dict:
    """Keep worker identities outside the actor, including reparented
    workers."""
    import psutil

    parent = psutil.Process(process.pid)
    processes = [parent, *parent.children(recursive=True)]
    identity = {
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
    if identity.get("incarnation"):
        # Followers return before the child enters its target. Never capture
        # the inherited Ray actor PGID; only this same server can be leader.
        deadline = time.monotonic() + timeout
        while True:
            try:
                pgid = os.getpgid(parent.pid)
                if not parent.is_running() or parent.status() == psutil.STATUS_ZOMBIE:
                    break
                if pgid == parent.pid:
                    identity["pgid"] = pgid
                    return identity
            except (psutil.NoSuchProcess, ProcessLookupError):
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.01, remaining))
        logger.warning(
            "Server pid=%s node=%s did not establish its own process group within %.2fs; "
            "falling back to PID/token evidence",
            parent.pid,
            node_id,
            timeout,
        )
    return identity


def cleanup_server_identity(
    identity: dict, timeout: float = 10.0, *, _reap_server: Callable[[], None] | None = None
) -> None:
    """Node-local exit proof; never signal a PID belonging to a new process."""
    import psutil

    records = identity.get("processes")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Missing server process-tree identity")
    token = identity.get("incarnation")
    pgid = identity.get("pgid")
    if pgid is not None and (
        type(pgid) is not int or pgid <= 1 or pgid != identity.get("server_pid") or pgid in (os.getpgrp(), os.getpid())
    ):
        raise RuntimeError(f"Unsafe server process group: {pgid!r}")
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

    def group_exists() -> bool:
        if pgid is None:
            return False
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        # EPERM and zombies are not exit proof. Only ESRCH means empty.
        return True

    def owns_group(processes: list) -> bool:
        # Revalidate each signal, including group membership: a token-bearing
        # escapee cannot authorize killing a different generation's reused PGID.
        for record in records:
            if record["pid"] != pgid:
                continue
            try:
                parent = psutil.Process(record["pid"])
                if parent.create_time() == record["create_time"] and os.getpgid(parent.pid) == pgid:
                    return True
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
        if token:
            for process in processes:
                try:
                    if (
                        process.is_running()
                        and process.environ().get(SERVER_INCARNATION_ENV) == token
                        and os.getpgid(process.pid) == pgid
                    ):
                        return True
                except (psutil.AccessDenied, psutil.NoSuchProcess, ProcessLookupError):
                    pass
        return False

    processes = members()
    while True:
        if group_exists() and owns_group(processes):
            try:
                # Unlike a PID snapshot, group SIGKILL also covers descendants
                # born while enumerating. SGLang does not leave its group.
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in reversed(processes):
            try:
                process.kill()  # psutil also guards against PID reuse after lookup.
            except psutil.NoSuchProcess:
                pass
        if _reap_server is not None:
            _reap_server()
        remaining = max(0.0, deadline - time.monotonic())
        waiting = [p for p in live(processes) if pgid is None or p.pid != identity["server_pid"]]
        # The actor's multiprocessing.Process must reap its own server. A
        # psutil waitpid here would steal its exit status and prevent join.
        _, alive = psutil.wait_procs(waiting, timeout=min(0.1, remaining) if token or pgid else remaining)
        if _reap_server is not None:
            _reap_server()
        # A worker may have spawned between enumeration and killing its parent.
        # Re-scan in the same budget after stopping the known members.
        # An unreadable later scan cannot revoke a known member's still-live
        # identity. Retain it until exit is confirmed, even after reparenting.
        processes = list({p.pid: p for p in live([*alive, *members()])}.values()) if token or pgid else live(alive)
        if not processes and not group_exists():
            return
        if not (token or pgid) or time.monotonic() >= deadline:
            raise RuntimeError(
                f"Server worker exit was not confirmed: {[p.pid for p in processes]}; process group={pgid} "
                "(requires a current generation member to signal and ESRCH to confirm exit)"
            )
        if not processes:
            # A group containing only zombies, or without a readable anchor,
            # remains fenced while its parent/init gets time to reap it.
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))


def terminate_server(
    process: Any, kill_tree: Callable[[int], None], timeout: float = 10.0, *, identity: dict | None = None
) -> None:
    import psutil

    if process is None or process.pid is None:
        return
    if identity is not None:
        # A saved tree includes workers reparented after the server died. Its
        # creation times also prevent signalling a reused parent PID.
        cleanup_server_identity(
            identity,
            timeout=timeout,
            _reap_server=(lambda: process.join(timeout=0)) if identity.get("pgid") is not None else None,
        )
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
