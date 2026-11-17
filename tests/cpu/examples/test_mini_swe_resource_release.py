# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real Linux processes verify release receipts cannot outrun cleanup."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


EXAMPLE = Path(__file__).resolve().parents[3] / "examples/mini_swe_agent"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper and /proc contract")


def wait_for(predicate, timeout=8):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for owned process state")


@pytest.mark.parametrize("outcome", ["success", "nonzero", "truncated", "cancel", "cancel_pipe"])
def test_release_joins_detached_descendants_and_preserves_graceful_stop(tmp_path, outcome):
    cli = tmp_path / "apptainer"
    cli.write_text(
        f"#!{sys.executable}\nimport sys,time\nif 'stop' in sys.argv: time.sleep(0.4)\nelse: print('INSTANCE PID')\n"
    )
    cli.chmod(0o755)
    spec, report, child_pid = tmp_path / "spec.json", tmp_path / "report.json", tmp_path / "child.pid"
    spec.write_text(
        json.dumps(
            {
                "apptainer_args": [str(cli), []],
                "instance_name": "mswe-owned",
                "record": {"session_id": "client"},
                "release_nonce": "nonce",
            }
        )
    )
    child_code = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"
    worker_result = "{" if outcome == "truncated" else json.dumps({"info": {}, "error": ""})
    worker_code = (
        "import subprocess,sys,time;from pathlib import Path;"
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True);"
        f"Path({str(child_pid)!r}).write_text(str(p.pid));"
        f"Path({str(report.with_suffix('.worker.json'))!r}).write_text({worker_result!r});"
        + (
            "time.sleep(60)"
            if outcome in {"cancel", "cancel_pipe"}
            else f"time.sleep(0.05);sys.exit({7 if outcome == 'nonzero' else 0})"
        )
    )
    read_fd, write_fd = os.pipe()
    runner = (
        "import sys;from pathlib import Path;"
        f"sys.path.insert(0,{str(EXAMPLE)!r});from session_process import supervise;"
        f"supervise(Path({str(spec)!r}),Path({str(report)!r}),"
        f"worker_command=[sys.executable,'-c',{worker_code!r}],cancel_fd={read_fd})"
    )
    proc = subprocess.Popen([sys.executable, "-c", runner], pass_fds=(read_fd,))
    os.close(read_fd)
    try:
        wait_for(child_pid.exists)
        pid = int(child_pid.read_text())
        assert not report.exists()
        if outcome == "cancel":
            proc.send_signal(signal.SIGTERM)
        elif outcome == "cancel_pipe":
            descriptor, write_fd = write_fd, None
            os.close(descriptor)
        time.sleep(0.1)
        assert not report.exists()  # Even an empty instance list is not a child-release proof.
        assert proc.wait(timeout=8) == 0
        result = json.loads(report.read_text())
        assert result["resources_released"] is True
        assert result["children_reaped"] is True
        assert result["instance_absent"] is True
        assert result["client_session_id"] == "client"
        assert result["nonce"] == "nonce"
        assert not Path(f"/proc/{pid}").exists()  # Reaped, not just a zombie.
        if outcome == "nonzero":
            assert "code 7" in result["error"]  # The Popen worker owns its exit status.
        elif outcome == "truncated":
            assert "valid result" in result["error"]
        elif outcome in {"cancel", "cancel_pipe"}:
            assert result["error"]
        else:
            assert result["error"] == ""
    finally:
        if write_fd is not None:
            os.close(write_fd)
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)


@pytest.mark.parametrize("instance_state", ["absent", "present", "unknown"])
def test_cancel_before_supervisor_start_skips_worker_but_still_requires_release_proof(tmp_path, instance_state):
    cli = tmp_path / "apptainer"
    cli.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "print('INSTANCE PID')\n"
        + ("print('mswe-owned 123')\n" if instance_state == "present" else "")
        + ("sys.exit(1)\n" if instance_state == "unknown" else "")
    )
    cli.chmod(0o755)
    spec, report, launched = tmp_path / "spec.json", tmp_path / "report.json", tmp_path / "worker-launched"
    spec.write_text(
        json.dumps(
            {
                "apptainer_args": [str(cli), []],
                "instance_name": "mswe-owned",
                "record": {"session_id": "client"},
                "release_nonce": "nonce",
            }
        )
    )
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    worker = f"from pathlib import Path;Path({str(launched)!r}).write_text('started')"
    runner = (
        "import sys;from pathlib import Path;"
        f"sys.path.insert(0,{str(EXAMPLE)!r});from session_process import supervise;"
        f"supervise(Path({str(spec)!r}),Path({str(report)!r}),"
        f"worker_command=[sys.executable,'-c',{worker!r}],cancel_fd={read_fd});"
        f"\ntry:os.fstat({read_fd})\nexcept OSError:pass\nelse:raise AssertionError('cancel fd leaked')"
    )
    runner = "import os;" + runner
    try:
        subprocess.run([sys.executable, "-c", runner], pass_fds=(read_fd,), check=True, timeout=8)
    finally:
        os.close(read_fd)
    result = json.loads(report.read_text())
    assert not launched.exists()
    assert result["error"] == "Session cancelled before worker launch"
    assert result["client_session_id"] == "client" and result["nonce"] == "nonce"
    assert result["children_reaped"] is True
    assert result["instance_absent"] is (instance_state == "absent")
    assert result["resources_released"] is (instance_state == "absent")


def test_supervisor_closes_cancel_pipe_when_setup_fails(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text("{")
    read_fd, write_fd = os.pipe()
    runner = (
        "import os,sys;from pathlib import Path;"
        f"sys.path.insert(0,{str(EXAMPLE)!r});from session_process import supervise;"
        f"\ntry:supervise(Path({str(spec)!r}),Path({str(tmp_path / 'report.json')!r}),cancel_fd={read_fd})"
        "\nexcept ValueError:pass\nelse:raise AssertionError('invalid spec accepted')"
        f"\ntry:os.fstat({read_fd})\nexcept OSError:pass\nelse:raise AssertionError('cancel fd leaked')"
    )
    try:
        subprocess.run([sys.executable, "-c", runner], pass_fds=(read_fd,), check=True, timeout=8)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_cancel_between_initial_check_and_worker_spawn_still_joins_worker(tmp_path):
    cli = tmp_path / "apptainer"
    cli.write_text("#!/bin/sh\necho 'INSTANCE PID'\n")
    cli.chmod(0o755)
    spec, report = tmp_path / "spec.json", tmp_path / "report.json"
    ready, child_pid = tmp_path / "spawn-ready", tmp_path / "worker.pid"
    spec.write_text(
        json.dumps(
            {
                "apptainer_args": [str(cli), []],
                "instance_name": "mswe-owned",
                "record": {"session_id": "client"},
                "release_nonce": "nonce",
            }
        )
    )
    cancel_read, cancel_write = os.pipe()
    gate_read, gate_write = os.pipe()
    runner = f"""
import os,sys
from pathlib import Path
sys.path.insert(0,{str(EXAMPLE)!r})
import session_process
real_popen = session_process.subprocess.Popen
worker_command = [sys.executable, '-c', 'import time;time.sleep(60)']
def launch(command, *args, **kwargs):
    if command == worker_command:
        Path({str(ready)!r}).touch()
        os.read({gate_read}, 1)
    process = real_popen(command, *args, **kwargs)
    if command == worker_command:
        Path({str(child_pid)!r}).write_text(str(process.pid))
    return process
session_process.subprocess.Popen = launch
session_process.supervise(Path({str(spec)!r}), Path({str(report)!r}),
                          worker_command=worker_command, cancel_fd={cancel_read})
os.close({gate_read})
"""
    proc = subprocess.Popen([sys.executable, "-c", runner], pass_fds=(cancel_read, gate_read))
    os.close(cancel_read)
    os.close(gate_read)
    try:
        wait_for(ready.exists)
        descriptor, cancel_write = cancel_write, None
        os.close(descriptor)
        os.write(gate_write, b"1")
        assert proc.wait(timeout=8) == 0
        result = json.loads(report.read_text())
        assert result["resources_released"] is True
        assert result["children_reaped"] is True and result["instance_absent"] is True
        assert result["error"]
        assert not Path(f"/proc/{int(child_pid.read_text())}").exists()
    finally:
        if cancel_write is not None:
            os.close(cancel_write)
        os.close(gate_write)
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)


def test_repeated_cancel_pipe_eof_keeps_first_cleanup_deadline(tmp_path):
    spec, report = tmp_path / "spec.json", tmp_path / "report.json"
    spec.write_text(json.dumps({"record": {"session_id": "client"}, "release_nonce": "nonce"}))
    runner = f"""
import os,signal,sys
from pathlib import Path
sys.path.insert(0,{str(EXAMPLE)!r})
import session_process
read_fd, write_fd = os.pipe()
clock = -6.0
def monotonic():
    global clock
    clock += 6.0
    return clock
class Worker:
    pid = 999999999
    returncode = None
    polls = 0
    def poll(self):
        self.polls += 1
        if self.polls > 4 and self.returncode is None:
            self.returncode = 7
        return self.returncode
    def wait(self):
        return self.returncode
worker = Worker()
kills, deadlines = [], []
def killpg(_pid, signum):
    kills.append(signum)
    if signum == signal.SIGKILL:
        worker.returncode = -signum
def instance_absent(_spec, deadline, *, stop):
    deadlines.append(deadline)
    return True
# Keep the pipe live for the initial check; close its writer only at spawn.
session_process.time.monotonic = monotonic
session_process.os.killpg = killpg
def spawn(*args, **kwargs):
    os.close(write_fd)
    return worker
session_process.subprocess.Popen = spawn
session_process.instance_absent = instance_absent
session_process.supervise(Path({str(spec)!r}), Path({str(report)!r}),
                          worker_command=['unused'], cancel_fd=read_fd)
assert kills == [signal.SIGTERM, signal.SIGKILL], kills
assert deadlines == [35.0, 35.0], deadlines
"""
    subprocess.run([sys.executable, "-c", runner], check=True, timeout=8)


def test_client_fails_closed_on_unconfirmed_cleanup(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if "release_nonce" in payload:
                self.server.release_nonce = payload["release_nonce"]
            self.respond({"session_id": "server"})

        def do_GET(self):
            self.respond(
                {
                    "session_id": "server",
                    "client_session_id": "client",
                    "status": "failed",
                    "resource_release_failed": True,
                    "resource_release": {
                        "schema": 1,
                        "session_id": "client",
                        "nonce": self.server.release_nonce,
                        "status": "unknown",
                    },
                }
            )

        def respond(self, value):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def log_message(self, *_args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            [sys.executable, str(EXAMPLE / "agent_client.py")],
            timeout=5,
            env={
                **os.environ,
                "AGENT_SERVER_URL": f"http://127.0.0.1:{httpd.server_port}",
                "RELAX_SESSION_ID": "client",
                "RELAX_GROUP_ID": "g",
                "RELAX_ROLLOUT_MODE": "train",
                "RELAX_BASE_URL": "http://unused",
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_supervisor_reaps_daemon_exit_before_worker_cleanup_finishes(tmp_path):
    cli = tmp_path / "apptainer"
    cli.write_text("#!/bin/sh\necho 'INSTANCE PID'\n")
    cli.chmod(0o755)
    spec, report = tmp_path / "spec.json", tmp_path / "report.json"
    child_file = tmp_path / "child.pid"
    lived_file = tmp_path / "child-lived"
    spec.write_text(
        json.dumps(
            {
                "apptainer_args": [str(cli), []],
                "instance_name": "mswe-owned",
                "record": {"session_id": "client"},
                "release_nonce": "nonce",
            }
        )
    )
    bridge = (
        "import os,time;from pathlib import Path;pid=os.fork()\n"
        f"if pid: Path({str(child_file)!r}).write_text(str(pid))\n"
        f"else: time.sleep(0.5);Path({str(lived_file)!r}).write_text('alive')\n"
        "os._exit(0)"
    )
    worker = (
        "import subprocess,sys,time,json;from pathlib import Path;"
        f"subprocess.run([sys.executable,'-c',{bridge!r}],check=True);"
        f"pid=int(Path({str(child_file)!r}).read_text());deadline=time.monotonic()+2\n"
        "while Path(f'/proc/{pid}').exists() and time.monotonic()<deadline:time.sleep(0.01)\n"
        f"Path({str(report.with_suffix('.worker.json'))!r}).write_text(json.dumps({{'info':{{}},"
        "'error':'adopted zombie blocked worker cleanup' if Path(f'/proc/{pid}').exists() else ''}))"
    )
    runner = (
        "import sys;from pathlib import Path;"
        f"sys.path.insert(0,{str(EXAMPLE)!r});from session_process import supervise;"
        f"supervise(Path({str(spec)!r}),Path({str(report)!r}),worker_command=[sys.executable,'-c',{worker!r}])"
    )
    subprocess.run([sys.executable, "-c", runner], check=True, timeout=8)
    result = json.loads(report.read_text())
    assert result["error"] == ""
    assert result["resources_released"] is True
    assert lived_file.read_text() == "alive"  # Running adopted children must never be killed by the early reaper.
