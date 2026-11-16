# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Process exit checks shared by local shutdown and dead-actor recovery."""

import os
import time

import psutil


def process_is_live(process: psutil.Process) -> bool:
    """Zombies have exited and no longer own CUDA contexts or executable work."""
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def server_session_exited(session_id: int) -> bool:
    """Confirm exit on the server's node, independently of its Ray actor.

    Servers start a new session before launching backend children. Reparenting
    does not change session membership. A reused session id conservatively
    prevents recovery; this check never signals another process.
    """
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
