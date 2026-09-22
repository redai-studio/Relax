# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run one deterministic Workplace Assistant recipe trial through the full Gym
graph."""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx


TERMINAL_STATUSES = {"completed", "truncated", "aborted", "failed"}


class CallbackState:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.lock = threading.Lock()
        self.tool_result_counts: list[int] = []

    def next_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload.get("tools"), list) or not payload["tools"]:
            raise ValueError("Callback request did not preserve Workplace Assistant tools")
        input_items = payload.get("input")
        if not isinstance(input_items, list):
            raise ValueError("Callback request did not contain Responses input items")
        tool_results = sum(
            1 for item in input_items if isinstance(item, dict) and item.get("type") == "function_call_output"
        )
        with self.lock:
            self.tool_result_counts.append(tool_results)
            call_index = len(self.tool_result_counts) - 1

        if call_index == 0:
            return _response(
                tool_name="email_search_emails",
                arguments={"query": "Carlos Task Update"},
                call_id="call_search",
            )
        if call_index == 1:
            if tool_results < 1:
                raise ValueError("Second callback did not contain the first tool result")
            return _response(
                tool_name="email_reply_email",
                arguments={
                    "email_id": "00000057",
                    "body": "Thanks for the update - I will get back to you tomorrow.",
                },
                call_id="call_reply",
            )
        if call_index == 2:
            if tool_results < 2:
                raise ValueError("Third callback did not contain both tool results")
            return _response(content="Done.")
        raise ValueError(f"Unexpected callback number: {call_index + 1}")


class CallbackHandler(BaseHTTPRequestHandler):
    server: "CallbackServer"

    def do_POST(self) -> None:
        try:
            if self.path != "/v1/responses":
                self.send_error(404)
                return
            if self.headers.get("Authorization") != f"Bearer {self.server.state.api_key}":
                self.send_error(401)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            response = self.server.state.next_response(payload)
            body = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": {"message": str(exc)}}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class CallbackServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], state: CallbackState) -> None:
        super().__init__(address, CallbackHandler)
        self.state = state


def _response(
    *,
    content: str | None = None,
    tool_name: str | None = None,
    arguments: dict[str, Any] | None = None,
    call_id: str | None = None,
) -> dict[str, Any]:
    response_id = f"resp_{uuid.uuid4().hex}"
    output: list[dict[str, Any]] = []
    if tool_name is not None:
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": tool_name,
                "arguments": json.dumps(arguments, separators=(",", ":")),
            }
        )
    elif content is not None:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        )
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": "workplace-contract-model",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 2,
        },
    }


def _read_task(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        task = json.loads(handle.readline())
    if not isinstance(task, dict) or not isinstance(task.get("responses_create_params"), dict):
        raise ValueError("Expected a NeMo Gym Workplace Assistant JSONL row")
    return task


def _trial_payload(task: dict[str, Any], *, request_id: str, api_key: str, callback_url: str) -> dict[str, Any]:
    return {
        "protocol_version": "relax-nemo-gym/v1",
        "request_id": request_id,
        "session": {
            "session_id": api_key,
            "group_id": "workplace-contract",
            "rollout_mode": "eval",
            "attempt": 1,
        },
        "environment": {
            "name": "workplace_assistant",
            "config": "workplace-assistant-v1",
            "task": task,
        },
        "model_endpoint": {
            "base_url": callback_url,
            "api_key": api_key,
            "model": "workplace-contract-model",
        },
        "generation": {},
        "interrupt_policy": "protected",
        "deadline_s": 120,
        "lease_s": 30,
        "metadata": {"integration_test": True},
    }


def _wait_for_trial(client: httpx.Client, gateway_url: str, request_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 120
    next_renewal = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"{gateway_url}/v1/trials/{request_id}")
        response.raise_for_status()
        result = response.json()
        if result["status"] in TERMINAL_STATUSES:
            return result
        if time.monotonic() >= next_renewal:
            renewal = client.post(f"{gateway_url}/v1/trials/{request_id}/renew")
            renewal.raise_for_status()
            next_renewal = time.monotonic() + 10
        time.sleep(0.2)
    raise TimeoutError("Workplace Assistant trial did not finish within 120 seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:29000")
    parser.add_argument("--callback-host", default="127.0.0.1")
    parser.add_argument("--callback-port", type=int, default=28099)
    parser.add_argument(
        "--task-jsonl",
        type=Path,
        default=Path("/opt/nemo-gym/resources_servers/workplace_assistant/data/example.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request_id = f"workplace-contract-{uuid.uuid4().hex}"
    api_key = uuid.uuid4().hex
    state = CallbackState(api_key)
    server = CallbackServer((args.callback_host, args.callback_port), state)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    gateway_url = args.gateway_url.rstrip("/")
    callback_url = f"http://{args.callback_host}:{server.server_port}/v1"

    try:
        task = _read_task(args.task_jsonl)
        payload = _trial_payload(task, request_id=request_id, api_key=api_key, callback_url=callback_url)
        with httpx.Client(timeout=10, trust_env=False) as client:
            created = client.post(f"{gateway_url}/v1/trials", json=payload)
            created.raise_for_status()
            result = _wait_for_trial(client, gateway_url, request_id)
            health = client.get(f"{gateway_url}/readyz")
            health.raise_for_status()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    metrics = result.get("metrics", {})
    expected_metrics = {"tool_calls": 2, "tool_outputs": 2}
    if result["status"] != "completed" or float(result["reward"]) != 1.0:
        raise RuntimeError(f"Workplace trial failed: {json.dumps(result, sort_keys=True)}")
    if any(metrics.get(key) != value for key, value in expected_metrics.items()):
        raise RuntimeError(f"Workplace trial did not preserve tool interactions: {metrics}")
    if state.tool_result_counts != [0, 1, 2]:
        raise RuntimeError(f"Callback session history was not preserved: {state.tool_result_counts}")
    if health.json().get("active_trials") != 0:
        raise RuntimeError(f"Gateway retained an active trial: {health.json()}")

    print(
        "Workplace Assistant trial passed: "
        f"reward={result['reward']} tool_calls={metrics['tool_calls']} "
        f"tool_outputs={metrics['tool_outputs']} callbacks={len(state.tool_result_counts)}"
    )


if __name__ == "__main__":
    main()
