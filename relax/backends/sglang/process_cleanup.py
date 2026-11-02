# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Confirm exit of a server and its workers before returning GPU ownership."""

import multiprocessing
import os
import signal
import time
from copy import deepcopy
from typing import Any, Callable


SERVER_INCARNATION_ENV = "RELAX_SGLANG_SERVER_INCARNATION"


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
    # marker; preserve SGLang's pool mode. The isolated process group also
    # accounts for these workers after their server has exited.


class RouterCleanupError(RuntimeError):
    """Routing cleanup failed after the server tree confirmed exit."""

    released = True


class ServerCleanupError(RuntimeError):
    """Unconfirmed exit with generation-bound partial evidence for retry."""

    def __init__(self, message: str, identity: dict) -> None:
        super().__init__(message)
        self.identity = deepcopy(identity)

    def __reduce__(self) -> tuple:
        return ServerCleanupError, (str(self), self.identity)


def _process_is_live(process: Any) -> bool:
    import psutil

    try:
        # Linux can retain a zombie thread-group leader while its other
        # threads still run (or wait in D state). Only a lone zombie is done.
        return process.is_running() and (process.status() != psutil.STATUS_ZOMBIE or process.num_threads() > 1)
    except psutil.NoSuchProcess:
        return False


def capture_server_identity(process: Any, *, node_id: str, port: int, timeout: float = 5.0) -> dict:
    """Keep worker identities outside the actor, including reparented workers.

    ``timeout`` remains accepted for callers of the former observation API;
    capture no longer waits for process-group setup.
    """
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
        # The managed entry executes setpgid(0, 0) before anything else. If
        # entry never runs, this server cannot have created any descendants.
        # Retain the candidate even before entry or after the leader exits.
        identity["pgid"] = parent.pid
    return identity


def cleanup_server_identity(
    identity: dict, timeout: float = 10.0, *, _reap_server: Callable[[], None] | None = None
) -> None:
    """Node-local exit proof; never signal a PID belonging to a new process."""
    try:
        _cleanup_server_identity(identity, timeout, _reap_server=_reap_server)
    except Exception as error:
        # Ray task arguments are copies. Return generation-bound exit facts even
        # on failure so the owner can retry without losing partial evidence.
        raise ServerCleanupError(str(error), identity) from error


def _cleanup_server_identity(identity: dict, timeout: float, *, _reap_server: Callable[[], None] | None) -> None:
    import psutil

    records = identity.get("processes")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Missing server process-tree identity")
    token = identity.get("incarnation")
    pgid = identity.get("pgid")
    server_record = next((record for record in records if record["pid"] == identity.get("server_pid")), None)
    if pgid is not None and (
        type(pgid) is not int or pgid <= 1 or pgid != identity.get("server_pid") or pgid in (os.getpgrp(), os.getpid())
    ):
        raise RuntimeError(f"Unsafe server process group: {pgid!r}")
    if pgid is not None and server_record is None:
        raise RuntimeError("Missing server creation time for process group")
    deadline = time.monotonic() + timeout
    group_key = [pgid, server_record["create_time"] if server_record is not None else None, token]
    group_released = pgid is None or identity.get("released_process_group") == group_key

    def record_group_release() -> None:
        nonlocal group_released
        group_released = True
        identity["released_process_group"] = group_key

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
        return [process for process in processes if _process_is_live(process)]

    def group_retired() -> bool:
        if not group_released:
            try:
                # Linux cannot reuse this PID while the old group exists.
                # A different leader proves the old group was emptied. Latch
                # that proof even if this new leader subsequently disappears.
                leader = psutil.Process(pgid)
                if leader.create_time() != server_record["create_time"]:
                    record_group_release()
            except psutil.NoSuchProcess:
                # A dead leader does not disown its surviving fork workers,
                # which may have no exec-time token in /proc/environ.
                pass
        return group_released

    def live_group_after_kill() -> list:
        if group_retired():
            return []
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            record_group_release()
            return []
        # This fresh enumeration is AFTER a successful group SIGKILL. Linux
        # checks pending fatal signals before publishing a fork into its group,
        # so membership can only shrink now. Zombies hold no GPU resources;
        # init need not reap them for release. D/stopped/unreadable members
        # still prevent proof. Never substitute the earlier PID/token snapshot.
        remaining = []
        # process_iter caches Process identities by PID. A reused cached PID
        # could make is_running() hide a current, still-live group member.
        for pid in psutil.pids():
            try:
                process = psutil.Process(pid)
                if os.getpgid(process.pid) == pgid:
                    remaining.extend(live([process]))
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
        if not remaining:
            record_group_release()
        return remaining

    processes = members()
    while True:
        # Preserve visible reuse before cleaning an escapee can coincide with
        # the replacement leader's exit and hide that evidence.
        group_retired()
        # Stop the recorded server before treating an absent candidate group
        # as retired: it could otherwise setpgid/fork between ESRCH and PID
        # kill. The final group signal covers children born before PID kill.
        for process in reversed(processes):
            try:
                process.kill()  # psutil also guards against PID reuse after lookup.
            except psutil.NoSuchProcess:
                pass
        if not group_retired():
            try:
                # Unlike a PID snapshot, group SIGKILL also covers descendants
                # born while enumerating. SGLang does not leave its group.
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                record_group_release()
        if _reap_server is not None:
            _reap_server()
        remaining = max(0.0, deadline - time.monotonic())
        waiting = [
            p for p in live(processes) if (pgid is None and _reap_server is None) or p.pid != identity["server_pid"]
        ]
        # The actor's multiprocessing.Process must reap its own server. A
        # psutil waitpid here would steal its exit status and prevent join.
        _, alive = psutil.wait_procs(waiting, timeout=min(0.1, remaining) if token or pgid else remaining)
        if _reap_server is not None:
            _reap_server()
        # A worker may have spawned between enumeration and killing its parent.
        # Re-scan in the same budget after stopping the known members.
        # An unreadable later scan cannot revoke a known member's still-live
        # identity. Retain it until exit is confirmed, even after reparenting.
        processes = (
            list({p.pid: p for p in live([*alive, *members(), *live_group_after_kill()])}.values())
            if token or pgid
            else live(alive)
        )
        if not processes and group_released:
            return
        if not (token or pgid) or time.monotonic() >= deadline:
            raise RuntimeError(
                f"Server worker exit was not confirmed: {[p.pid for p in processes]}; process group={pgid} "
                "(requires exit of PID/token escapees and no live group members after SIGKILL)"
            )


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
            _reap_server=(lambda: process.join(timeout=0))
            if (identity.get("incarnation") or identity.get("pgid") is not None)
            and callable(getattr(process, "join", None))
            else None,
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
