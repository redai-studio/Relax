# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import multiprocessing
import os
import subprocess
import sys
import time

import pytest

from relax.backends.sglang import process_lifecycle as lifecycle


@pytest.fixture
def backend_session():
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=lifecycle.run_in_backend_session, args=(time.sleep, (30,), send))
    process.start()
    send.close()
    try:
        assert receive.poll(10)
        yield process, receive.recv()
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        receive.close()


def test_backend_session_exit_requires_reaped_root(backend_session):
    process, identity = backend_session
    assert identity["session_id"] == process.pid
    assert identity["session_id"] != os.getsid(0)
    assert not lifecycle.backend_session_exited(identity)
    process.terminate()
    process.join(timeout=5)
    assert lifecycle.backend_session_exited(identity)


@pytest.mark.parametrize("field", ["boot_id", "pid_namespace", "pid", "session_id", "start_time"])
def test_backend_session_rejects_invalid_identity(backend_session, field):
    process, identity = backend_session
    process.terminate()
    process.join(timeout=5)
    assert not lifecycle.backend_session_exited({**identity, field: None})


def test_backend_session_rejects_pid_reuse_and_unreadable_proc(backend_session, monkeypatch):
    process, identity = backend_session
    process.terminate()
    process.join(timeout=5)
    monkeypatch.setattr(lifecycle, "_process_stat", lambda pid: (identity["session_id"], identity["start_time"] + 1))
    assert not lifecycle.backend_session_exited(identity)

    def denied(pid):
        raise PermissionError("cannot inspect process")

    monkeypatch.setattr(lifecycle, "_process_stat", denied)
    assert not lifecycle.backend_session_exited(identity)


def test_backend_session_detects_orphaned_worker_without_touching_other_processes():
    # A separate test supervisor reaps the orphan so this test neither relies
    # on container PID 1 nor changes pytest's process adoption policy.
    script = r"""
import ctypes
import json
import os
import signal
import subprocess
import sys
from relax.backends.sglang.process_lifecycle import _local_identity, _process_stat, backend_session_exited

assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
child_code = "import time; time.sleep(30)"
root_code = "import os,subprocess,sys,time; os.setsid(); p=subprocess.Popen([sys.executable,'-c',%r]); print(p.pid,flush=True); time.sleep(30)" % child_code
root = subprocess.Popen([sys.executable, "-c", root_code], stdout=subprocess.PIPE, text=True)
other = subprocess.Popen([sys.executable, "-c", child_code])
worker_pid = None
try:
    worker_pid = int(root.stdout.readline())
    session_id, start_time = _process_stat(root.pid)
    identity = {**_local_identity(), "pid": root.pid, "session_id": session_id, "start_time": start_time}
    root.kill()
    root.wait(timeout=5)
    assert not backend_session_exited(identity), "root death must not hide an orphaned scheduler"
    os.kill(worker_pid, signal.SIGTERM)
    os.waitpid(worker_pid, 0)
    worker_pid = None
    assert backend_session_exited(identity)
    assert other.poll() is None
finally:
    if root.poll() is None:
        root.kill()
    root.wait(timeout=5)
    if worker_pid is not None:
        os.kill(worker_pid, signal.SIGKILL)
        os.waitpid(worker_pid, 0)
    other.terminate()
    other.wait(timeout=5)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
