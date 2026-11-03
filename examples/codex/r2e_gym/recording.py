# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Transparent Responses recording for the Codex compaction recipe."""

from __future__ import annotations

import json
import re
import socket
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterator

import anyio
import httpx
import uvicorn
from fastapi import FastAPI, Request
from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse


class ContextRecorder:
    """Keep one endpoint per successive context, including summary
    generation."""

    def __init__(self) -> None:
        # ponytail: forward-only Codex exec; rollback/branching needs lineage tracking.
        self.contexts: list[dict[str, Any]] = []
        self.error: str | None = None

    def record(self, payload: dict[str, Any], response: dict[str, Any]) -> None:
        if response.get("error", {}).get("code") == "context_length_exceeded":
            return
        if response.get("status") not in {"completed", "incomplete"}:
            raise ValueError(f"Codex response did not complete: {response.get('status')}")
        messages = [{"role": "system", "content": payload["instructions"]}]
        assistant = None
        for item in [*payload["input"], *response["output"]]:
            kind = item["type"]
            if assistant and (kind == "function_call_output" or (kind == "message" and item["role"] != "assistant")):
                messages.append(assistant)
                assistant = None
            if kind == "message":
                content = "".join(part["text"] for part in item["content"])
                if item["role"] == "assistant":
                    assistant = assistant or {"role": "assistant", "content": ""}
                    assistant["content"] += content
                else:
                    messages.append({"role": item["role"], "content": content})
            elif kind == "reasoning":
                assistant = assistant or {"role": "assistant", "content": ""}
                assistant["reasoning_content"] = "\n\n".join(part["text"] for part in item["content"])
            elif kind == "function_call":
                assistant = assistant or {"role": "assistant", "content": ""}
                assistant.setdefault("tool_calls", []).append(
                    {
                        "id": item["call_id"],
                        "type": "function",
                        "function": {key: item[key] for key in ("name", "arguments")},
                    }
                )
            elif kind == "function_call_output":
                messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
        if assistant:
            messages.append(assistant)
        tools = [
            {"type": tool["type"], "function": {key: value for key, value in tool.items() if key != "type"}}
            for tool in payload["tools"]
        ]
        effort = payload.get("reasoning", {}).get("effort")
        kwargs = {} if effort is None else {"enable_thinking": effort != "none"}
        if self.contexts:
            current = self.contexts[-1]
            if (
                tools == current["tools"]
                and kwargs == current["chat_template_kwargs"]
                and messages[: len(current["messages"])] == current["messages"]
            ):
                self.contexts.pop()
        self.contexts.append({"messages": messages, "tools": tools, "chat_template_kwargs": kwargs})

    def exports(self, reward: float) -> list[dict[str, Any]]:
        if self.error is not None:
            raise RuntimeError(f"Incomplete Codex recording: {self.error}")
        if not self.contexts:
            raise RuntimeError("Codex produced no recorded response")
        records = [{"name": f"context_{index}", **context} for index, context in enumerate(self.contexts)]
        records[-1].update(reward=reward, metadata={"session_reward": reward})
        return records


async def _stream_response(
    response: httpx.Response, payload: bytes | None, recorder: ContextRecorder
) -> AsyncIterator[bytes]:
    """Record terminal content before forwarding it to Codex."""
    buffer = bytearray()
    complete = False
    try:
        async for chunk in response.aiter_bytes():
            if payload is not None and recorder.error is None:
                buffer.extend(chunk)
                try:
                    while match := re.search(rb"\r\n\r\n|\n\n|\r\r", buffer):
                        frame = bytes(buffer[: match.start()])
                        del buffer[: match.end()]
                        data = b"\n".join(
                            line[5:].lstrip(b" ") for line in frame.splitlines() if line.startswith(b"data:")
                        )
                        if not data or data == b"[DONE]":
                            continue
                        event = json.loads(data)
                        if event.get("type") in {"response.completed", "response.incomplete", "response.failed"}:
                            if complete:
                                raise ValueError("Multiple terminal responses on one request")
                            recorder.record(json.loads(payload), event["response"])
                            complete = True
                        elif event.get("type") == "error":
                            raise ValueError(f"Responses error: {event.get('error')}")
                except Exception as error:
                    recorder.error = str(error)
                    buffer.clear()
            yield chunk
    finally:
        if payload is not None and recorder.error is None:
            try:
                if not complete:
                    raise ValueError("Response ended before a complete terminal event")
            except Exception as error:
                recorder.error = str(error)
        await response.aclose()


def _headers(headers: Any, *, response: bool = False) -> dict[str, str]:
    excluded = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
    excluded.update(part.strip().lower() for part in headers.get("connection", "").split(","))
    if response:
        excluded.add("content-encoding")  # httpx yields decompressed entity bytes.
    return {key: value for key, value in headers.items() if key.lower() not in excluded}


@contextmanager
def recording_proxy(upstream: str) -> Iterator[tuple[str, ContextRecorder]]:
    """Run the recording proxy on a Session-local loopback port."""
    recorder = ContextRecorder()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=30, write=30, pool=30), trust_env=False
        ) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def forward(request: Request, path: str) -> StreamingResponse:
        body = await request.body()
        url = f"{upstream.rstrip('/')}/{path}"
        if request.url.query:
            url += f"?{request.url.query}"
        client = request.app.state.client
        outbound = client.build_request(request.method, url, content=body, headers=_headers(request.headers))

        async def disconnected() -> None:
            while (await request.receive())["type"] != "http.disconnect":
                pass
            raise ClientDisconnect

        upstream_response = None
        try:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(disconnected)
                upstream_response = await client.send(outbound, stream=True)
                tasks.cancel_scope.cancel()
        except BaseException:
            recorder.error = recorder.error or "Upstream request failed or was cancelled"
            if upstream_response is not None:
                await upstream_response.aclose()
            raise
        return StreamingResponse(
            _stream_response(upstream_response, body if path.rstrip("/").endswith("/responses") else None, recorder),
            status_code=upstream_response.status_code,
            headers=_headers(upstream_response.headers, response=True),
        )

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False, timeout_graceful_shutdown=1))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError("Codex recording proxy did not start")
            time.sleep(0.01)
        yield f"http://127.0.0.1:{listener.getsockname()[1]}/v1", recorder
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        if thread.is_alive():
            raise RuntimeError("Codex recording proxy did not stop")
