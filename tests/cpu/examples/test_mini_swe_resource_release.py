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


@pytest.mark.parametrize("cancel", [False, True])
def test_release_joins_detached_descendants_and_preserves_graceful_stop(tmp_path, cancel):
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
    worker_code = (
        "import subprocess,sys,time;from pathlib import Path;"
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True);"
        f"Path({str(child_pid)!r}).write_text(str(p.pid));"
        + ("time.sleep(60)" if cancel else "time.sleep(0.05);sys.exit(7)")
    )
    runner = (
        "import sys;from pathlib import Path;"
        f"sys.path.insert(0,{str(EXAMPLE)!r});from session_process import supervise;"
        f"supervise(Path({str(spec)!r}),Path({str(report)!r}),"
        f"worker_command=[sys.executable,'-c',{worker_code!r}])"
    )
    proc = subprocess.Popen([sys.executable, "-c", runner])
    try:
        wait_for(child_pid.exists)
        pid = int(child_pid.read_text())
        assert not report.exists()
        if cancel:
            proc.send_signal(signal.SIGTERM)
        time.sleep(0.1)
        assert not report.exists()  # Even an empty instance list is not a child-release proof.
        assert proc.wait(timeout=8) == 0
        result = json.loads(report.read_text())
        assert result["resources_released"] is True
        assert result["children_reaped"] is True
        assert not Path(f"/proc/{pid}").exists()  # Reaped, not just a zombie.
        if not cancel:
            assert "code 7" in result["error"]  # The Popen worker owns its exit status.
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)


def test_client_fails_closed_on_unconfirmed_cleanup(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.respond({"session_id": "server"})

        def do_GET(self):
            self.respond(
                {
                    "session_id": "server",
                    "client_session_id": "client",
                    "status": "failed",
                    "resource_release_failed": True,
                    "resource_release": {"status": "unknown"},
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
    receipt = tmp_path / "release.json"
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
                "RELAX_REQUIRE_RESOURCE_RELEASE": "1",
                "RELAX_RESOURCE_RELEASE_NONCE": "nonce",
                "RELAX_RESOURCE_RELEASE_PATH": str(receipt),
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert json.loads(receipt.read_text()) == {
            "schema": 1,
            "session_id": "client",
            "nonce": "nonce",
            "status": "unknown",
        }
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
