# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Native command launch through the real supervisor, without descendant
adoption."""

import json
import os
import select
import shlex
import subprocess
import sys
import time

import pytest


def supervise(tmp_path, command, *, terminate_when=None):
    request = {
        "command": command,
        "cwd": str(tmp_path),
        "env": {**os.environ, "NATIVE_ARG_TEST": "expanded-value"},
        "log_path": str(tmp_path / "agent.log"),
    }
    bootstrap = """import sys
from relax.agentic.runner import ipc
if sys.platform != 'linux':
    ipc._enable_child_subreaper = lambda: None
ipc._session_supervisor_main()
"""
    # waitpid(-1) must run in a private supervisor, never in the pytest process
    # where Ray or other tests may have unrelated children. On macOS these roots
    # reap their own helpers; detached adoption has separate Linux-only coverage.
    supervisor = subprocess.Popen(
        [sys.executable, "-c", bootstrap],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    frames = []
    try:
        supervisor.stdin.write((json.dumps(request) + "\n").encode())
        if terminate_when is not None:
            deadline = time.monotonic() + 5
            while not terminate_when.exists() and supervisor.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert terminate_when.exists()
            supervisor.stdin.write(b'{"op":"terminate"}\n')
        while len(frames) < 2:
            ready, _, _ = select.select([supervisor.stdout], [], [], 10)
            assert ready, "supervisor did not report process status"
            line = supervisor.stdout.readline()
            assert line, supervisor.stderr.read().decode()
            frames.append(json.loads(line))
            if "error" in frames[-1]:
                break
        return frames, supervisor.pid
    finally:
        supervisor.stdin.close()
        try:
            supervisor.wait(timeout=10)
        except subprocess.TimeoutExpired:
            supervisor.kill()
            supervisor.wait(timeout=3)
        supervisor.stdout.close()
        supervisor.stderr.close()


def test_native_launch_preserves_quoted_arguments_and_has_no_shell_parent(tmp_path):
    code = "import json,os,sys; print(json.dumps([sys.argv[1:],os.environ['NATIVE_ARG_TEST'],os.getppid()]))"
    command = f'{shlex.quote(sys.executable)} -c {shlex.quote(code)} "two words" "${{NATIVE_ARG_TEST}}"'
    frames, parent_pid = supervise(tmp_path, command)
    assert frames == [{"op": "started"}, {"op": "exit", "return_code": 0}]
    args, env, parent = json.loads((tmp_path / "agent.log").read_text())
    assert args == ["two words", "${NATIVE_ARG_TEST}"] and env == "expanded-value"
    assert parent == parent_pid


@pytest.mark.parametrize("command", ["", "'unterminated"])
def test_invalid_command_reports_launch_error(tmp_path, command):
    frames, _ = supervise(tmp_path, command)
    (frame,) = frames
    assert "error" in frame


def test_root_term_leaves_same_group_cleanup_helper_alive(tmp_path):
    ready, release, premature = (tmp_path / name for name in ("ready", "release", "premature"))
    helper = f"""import signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, lambda *_: Path({str(premature)!r}).touch())
Path({str(ready)!r}).touch()
deadline = time.monotonic() + 8
while not Path({str(release)!r}).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
"""
    root = f"""import signal, subprocess, sys, time
from pathlib import Path
stopping = False
def stop(*_):
    global stopping
    stopping = True
signal.signal(signal.SIGTERM, stop)
child = subprocess.Popen([sys.executable, '-c', {helper!r}])
try:
    deadline = time.monotonic() + 6
    while not stopping and time.monotonic() < deadline:
        time.sleep(0.01)
    assert stopping
    assert child.poll() is None
    Path({str(release)!r}).touch()
    assert child.wait(timeout=2) == 0
    assert not Path({str(premature)!r}).exists()
finally:
    if child.poll() is None:
        child.kill()
        child.wait()
"""
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(root)}"
    frames, _ = supervise(tmp_path, command, terminate_when=ready)
    assert frames[-1] == {"op": "exit", "return_code": 0}
    assert release.exists() and not premature.exists()
