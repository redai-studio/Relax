# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import subprocess
import sys
import time

import psutil

from relax.utils.process_utils import process_is_live, server_session_exited


def test_server_session_blocks_release_while_a_real_process_is_live():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    try:
        assert process_is_live(psutil.Process(process.pid))
        assert not server_session_exited(process.pid)
        assert process.poll() is None
    finally:
        process.kill()
        process.wait(timeout=5)
    assert server_session_exited(process.pid)


def test_unreaped_zombie_has_released_its_server_session():
    process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    try:
        tracked = psutil.Process(process.pid)
        deadline = time.monotonic() + 5
        while tracked.status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert not process_is_live(tracked)
        assert server_session_exited(process.pid)
    finally:
        process.wait(timeout=5)


def test_orphan_server_child_blocks_release_until_exit(tmp_path):
    # Isolate subreaper ownership from pytest and reap the orphan ourselves.
    # The extra process is necessary: a dead leader must not hide a live child.
    script = tmp_path / "orphan_session.py"
    script.write_text(
        """
import ctypes
import os
import signal
import subprocess
import sys

from relax.utils.process_utils import server_session_exited

assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
parent = subprocess.Popen(
    [sys.executable, "-c", "import os,time; child=os.fork(); "
     "print(child,flush=True) if child else time.sleep(30)"],
    start_new_session=True, stdout=subprocess.PIPE, text=True,
)
child = int(parent.stdout.readline())
try:
    parent.wait(timeout=5)
    assert os.getsid(child) == parent.pid
    assert not server_session_exited(parent.pid)
    assert other.poll() is None
finally:
    os.kill(child, signal.SIGKILL)
    os.waitpid(child, 0)
    other.kill()
    other.wait(timeout=5)
assert server_session_exited(parent.pid)
"""
    )
    subprocess.run([sys.executable, str(script)], check=True, timeout=20)
