# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run a deterministic Claude Code trial through native Anthropic Messages."""

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


def _event(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _message_sse(*, model: str, blocks: list[dict[str, Any]], stop_reason: str) -> bytes:
    message_id = f"msg_{uuid.uuid4().hex}"
    frames = [
        _event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
        )
    ]
    for index, block in enumerate(blocks):
        if block["type"] == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"], separators=(",", ":"))}
        frames.extend(
            [
                _event(
                    "content_block_start",
                    {"type": "content_block_start", "index": index, "content_block": start},
                ),
                _event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": index, "delta": delta},
                ),
                _event("content_block_stop", {"type": "content_block_stop", "index": index}),
            ]
        )
    frames.extend(
        [
            _event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
            ),
            _event("message_stop", {"type": "message_stop"}),
        ]
    )
    return "".join(frames).encode()


class CallbackState:
    def __init__(self, api_key: str, answer: str) -> None:
        self.api_key = api_key
        self.answer = answer
        self.lock = threading.Lock()
        self.tool_result_counts: list[int] = []

    def next_response(self, payload: dict[str, Any]) -> bytes:
        tools = payload.get("tools")
        if not isinstance(tools, list) or not any(tool.get("name") == "Bash" for tool in tools):
            raise ValueError("Claude Code request did not expose the Bash tool")
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Claude Code request did not contain messages")
        tool_results = sum(
            block.get("type") == "tool_result"
            for message in messages
            if isinstance(message, dict) and isinstance(message.get("content"), list)
            for block in message["content"]
            if isinstance(block, dict)
        )
        with self.lock:
            self.tool_result_counts.append(tool_results)
            call_index = len(self.tool_result_counts) - 1
        model = str(payload.get("model") or "claude-sonnet-4-5")
        if call_index == 0:
            return _message_sse(
                model=model,
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "call_bash",
                        "name": "Bash",
                        "input": {"command": "python3 -c 'print(2 + 2)'"},
                    }
                ],
                stop_reason="tool_use",
            )
        if call_index == 1 and tool_results == 1:
            return _message_sse(
                model=model,
                blocks=[{"type": "text", "text": f"<answer>{self.answer}</answer>"}],
                stop_reason="end_turn",
            )
        raise ValueError(f"Unexpected callback state: call={call_index + 1} tool_results={tool_results}")


class CallbackHandler(BaseHTTPRequestHandler):
    server: "CallbackServer"

    def do_POST(self) -> None:
        try:
            if self.path.partition("?")[0] != "/v1/messages":
                self.send_error(404)
                return
            if self.headers.get("Authorization") != f"Bearer {self.server.state.api_key}":
                self.send_error(401)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if payload.get("stream") is not True:
                raise ValueError("Claude Code must request Messages streaming")
            body = self.server.state.next_response(payload)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"type": "error", "error": {"type": "api_error", "message": str(exc)}}).encode()
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


def _first_task(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = json.loads(line)
            if isinstance(task, dict) and isinstance(task.get("answer"), str):
                return task
    raise ValueError(f"{path} contained no row with a string answer")


def _trial_payload(task: dict[str, Any], *, request_id: str, api_key: str, callback_url: str) -> dict[str, Any]:
    return {
        "protocol_version": "relax-nemo-gym/v1",
        "request_id": request_id,
        "session": {
            "session_id": uuid.uuid4().hex,
            "group_id": "reasoning-gym-cc-contract",
            "rollout_mode": "eval",
            "attempt": 1,
        },
        "environment": {"name": "reasoning_gym", "config": "reasoning-gym-cc-v1", "task": task},
        "model_endpoint": {
            "base_url": callback_url,
            "api_key": api_key,
            "model": "reasoning-gym-cc-contract-model",
        },
        "generation": {},
        "interrupt_policy": "protected",
        "deadline_s": 300,
        "lease_s": 300,
        "metadata": {"integration_test": True},
    }


def _wait_for_trial(client: httpx.Client, gateway_url: str, request_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 330
    while time.monotonic() < deadline:
        result = client.get(f"{gateway_url}/v1/trials/{request_id}").raise_for_status().json()
        if result["status"] in TERMINAL_STATUSES:
            return result
        time.sleep(0.2)
    raise TimeoutError("reasoning-gym-cc trial did not finish within 330 seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:29200")
    parser.add_argument("--callback-host", default="127.0.0.1")
    parser.add_argument("--callback-port", type=int, default=0)
    parser.add_argument("--task-jsonl", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = _first_task(args.task_jsonl)
    request_id = f"reasoning-gym-cc-contract-{uuid.uuid4().hex}"
    api_key = uuid.uuid4().hex
    state = CallbackState(api_key, task["answer"])
    server = CallbackServer((args.callback_host, args.callback_port), state)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    gateway_url = args.gateway_url.rstrip("/")
    callback_url = f"http://{args.callback_host}:{server.server_port}/v1"
    try:
        payload = _trial_payload(task, request_id=request_id, api_key=api_key, callback_url=callback_url)
        with httpx.Client(timeout=30, trust_env=False) as client:
            client.post(f"{gateway_url}/v1/trials", json=payload).raise_for_status()
            result = _wait_for_trial(client, gateway_url, request_id)
            health = client.get(f"{gateway_url}/readyz").raise_for_status().json()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    metrics = result.get("metrics", {})
    if result["status"] != "completed" or float(result["reward"]) != 1.0:
        raise RuntimeError(f"reasoning-gym-cc trial failed: {json.dumps(result, sort_keys=True)}")
    if metrics.get("tool_calls") != 1 or metrics.get("tool_outputs") != 1:
        raise RuntimeError(f"Claude Code tool interaction was not preserved: {metrics}")
    if state.tool_result_counts != [0, 1]:
        raise RuntimeError(f"Claude Code full history was not preserved: {state.tool_result_counts}")
    if health.get("active_trials") != 0:
        raise RuntimeError(f"Gateway retained an active trial: {health}")
    print(f"reasoning-gym-cc trial passed: reward={result['reward']} callbacks=2")


if __name__ == "__main__":
    main()
