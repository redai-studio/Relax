# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""The compatibility entrypoint must preserve external cleanup ownership."""

import json
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "mini_swe_agent"


@pytest.fixture
def client_server(tmp_path):
    state = {"status": "completed", "create": None}
    polled, cancelled, released = threading.Event(), threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path.endswith("/cancel"):
                state["status"] = "cancelled"
                cancelled.set()
            else:
                state["create"] = payload
            self.respond({"session_id": "server"})

        def do_GET(self):
            polled.set()
            self.respond(
                {
                    "session_id": "server",
                    "client_session_id": state["create"]["session_id"],
                    "status": state["status"],
                    "exit_code": {"completed": 0, "failed": 1, "cancelled": 143}[state["status"]],
                    "error": "reward test timed out" if state["status"] == "failed" else "",
                    "agent_exit_status": "Submitted",
                    "submission": "patch",
                    "n_calls": 3,
                    "reward": 0.5,
                    "resource_release_failed": False,
                    "resource_release": {
                        "schema": 1,
                        "session_id": state["create"]["session_id"],
                        "nonce": state["create"]["release_nonce"],
                        "status": "released" if released.is_set() else "unknown",
                    },
                }
            )

        def respond(self, payload):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    processes = []

    def start(*, receipt=False):
        env = {
            **os.environ,
            "AGENT_SERVER_URL": f"http://127.0.0.1:{server.server_port}",
            "RELAX_SESSION_ID": "client",
            "RELAX_GROUP_ID": "group",
            "RELAX_ROLLOUT_MODE": "train",
            "RELAX_BASE_URL": "http://unused",
            "RELAX_OUTPUT_JSON": str(tmp_path / "output.json"),
            "RELAX_REQUIRE_RESOURCE_RELEASE": "1" if receipt else "0",
            "RELAX_RESOURCE_RELEASE_NONCE": "nonce",
            "RELAX_RESOURCE_RELEASE_PATH": str(tmp_path / "release.json"),
            "POLL_INTERVAL_SECONDS": "0.01",
            "INITIAL_POLL_DELAY_SECONDS": "0",
            "AGENT_CLIENT_TRACE_DIR": str(tmp_path / "traces"),
        }
        process = subprocess.Popen(
            ["bash", str(EXAMPLE / "agent_client.sh")], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        processes.append(process)
        return process

    try:
        yield start, state, polled, cancelled, released
    finally:
        released.set()
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            process.communicate(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("receipt", [False, True])
def test_shell_entrypoint_waits_for_external_release_before_output(client_server, tmp_path, receipt):
    start, state, polled, _cancelled, released = client_server
    process = start(receipt=receipt)
    assert polled.wait(timeout=3)
    time.sleep(0.05)
    assert process.poll() is None
    assert not (tmp_path / "output.json").exists()
    released.set()
    _stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, stderr
    assert json.loads((tmp_path / "output.json").read_text()) == {
        "metadata": {"exit_status": "Submitted", "submission": "patch", "n_calls": 3},
        "reward": 0.5,
    }
    assert state["create"]["group_id"] == "group"
    if receipt:
        assert json.loads((tmp_path / "release.json").read_text()) == {
            "schema": 1,
            "session_id": "client",
            "nonce": "nonce",
            "status": "released",
        }


def test_client_cancellation_waits_for_release_and_does_not_write_failure_trace(client_server, tmp_path):
    start, _state, polled, cancelled, released = client_server
    process = start(receipt=True)
    assert polled.wait(timeout=3)
    process.send_signal(signal.SIGTERM)
    assert cancelled.wait(timeout=3)
    assert process.poll() is None
    assert json.loads((tmp_path / "release.json").read_text())["status"] == "unknown"
    released.set()
    process.communicate(timeout=5)
    assert process.returncode == 143
    assert not (tmp_path / "output.json").exists()
    assert not (tmp_path / "traces").exists()


def test_client_preserves_reward_failure_and_opt_in_trace(client_server, tmp_path):
    start, state, _polled, _cancelled, released = client_server
    state["status"] = "failed"
    released.set()
    process = start()
    _stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 1
    assert b"reward test timed out" in stderr
    assert not (tmp_path / "output.json").exists()
    trace = json.loads((tmp_path / "traces/client.failed.json").read_text())
    assert trace["status"] == "failed" and trace["error"] == "reward test timed out"
