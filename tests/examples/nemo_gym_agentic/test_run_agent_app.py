# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "nemo_gym_agentic" / "scripts" / "run_agent_app.sh"
PROTOCOL_VERSION = "relax-nemo-gym/v1"
REQUIRED_ENV = {
    "RELAX_INPUT_JSON": "/tmp/input.json",
    "RELAX_OUTPUT_JSON": "/tmp/output.json",
    "RELAX_SESSION_ID": "session-id",
    "RELAX_SESSION_IO_DIR": "/tmp/relax-agentic-command-test",
    "RELAX_BASE_URL": "http://relax.example/agentic_api/",
    "NEMO_GYM_URL": "http://gym.example",
    "NEMO_GYM_ENVIRONMENT": "multi_step",
}


@contextmanager
def fake_gateway(*, create_status: str) -> Iterator[tuple[str, dict[str, Any]]]:
    state: dict[str, Any] = {
        "request": None,
        "created": threading.Event(),
        "aborted": threading.Event(),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path == "/v1/trials":
                body = self.rfile.read(int(self.headers.get("content-length", "0")))
                state["request"] = json.loads(body)
                state["created"].set()
                self._json_response(create_status)
                return
            if self.path.endswith("/abort"):
                state["aborted"].set()
                self.send_response(204)
                self.end_headers()
                return
            if self.path.endswith("/renew"):
                self.send_response(204)
                self.end_headers()
                return
            self.send_error(404)

        def do_GET(self) -> None:
            self._json_response("running")

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _json_response(self, status: str) -> None:
            request = state["request"] or {}
            payload = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request.get("request_id"),
                "status": status,
                "reward": {"scalar": 1.0, "components": {"verifier": 1.0}} if status == "completed" else None,
                "metrics": {"model_calls": 2} if status == "completed" else {},
                "artifact_ref": None,
                "error": None,
            }
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_run_agent_app_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


@pytest.mark.parametrize("missing_name", sorted(set(REQUIRED_ENV) - {"NEMO_GYM_ENVIRONMENT"}))
def test_run_agent_app_fails_fast_when_required_env_is_missing(missing_name):
    env = {"PATH": os.environ["PATH"], **REQUIRED_ENV}
    env.pop(missing_name)

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert missing_name in result.stderr


def test_run_agent_app_completes_against_fake_gateway(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_payload = {
        "messages": [{"role": "user", "content": "solve"}],
        "metadata": {"task_id": "task-1"},
    }
    input_path.write_text(json.dumps(input_payload), encoding="utf-8")

    with fake_gateway(create_status="completed") as (gateway_url, state):
        env = {
            **os.environ,
            **REQUIRED_ENV,
            "RELAX_INPUT_JSON": str(input_path),
            "RELAX_OUTPUT_JSON": str(output_path),
            "NEMO_GYM_URL": gateway_url,
            "NEMO_GYM_MAX_RETRIES": "0",
        }
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )

    assert result.returncode == 0, result.stderr
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "reward": 1.0,
        "metadata": {
            "nemo_gym": {
                "request_id": state["request"]["request_id"],
                "status": "completed",
                "metrics": {"model_calls": 2},
                "reward_components": {"verifier": 1.0},
            }
        },
    }
    assert state["request"]["environment"]["task"] == input_payload
    assert state["request"]["model_endpoint"]["api_key"] == "session-id"
    assert "session-id" not in state["request"]["request_id"]


def test_process_group_sigterm_requests_abort_without_writing_success(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "solve"}]}),
        encoding="utf-8",
    )

    with fake_gateway(create_status="running") as (gateway_url, state):
        env = {
            **os.environ,
            **REQUIRED_ENV,
            "RELAX_INPUT_JSON": str(input_path),
            "RELAX_OUTPUT_JSON": str(output_path),
            "NEMO_GYM_URL": gateway_url,
            "NEMO_GYM_POLL_INTERVAL_S": "10",
            "NEMO_GYM_ABORT_TIMEOUT_S": "1",
            "NEMO_GYM_MAX_RETRIES": "0",
        }
        process = subprocess.Popen(
            ["bash", str(SCRIPT)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert state["created"].wait(timeout=2.0)
        os.killpg(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=3.0)

    assert process.returncode != 0, stdout
    assert "Managed session cancelled" in stderr
    assert state["aborted"].is_set()
    assert not output_path.exists()
