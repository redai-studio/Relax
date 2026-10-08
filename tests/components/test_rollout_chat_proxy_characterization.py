# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Characterization of the Rollout ``/v1/chat/completions`` proxy.

Pins what the endpoint forwards to the SGLang router and how it maps upstream
failures, so moving the proxy behind the shared gateway can be shown not to
change it. ``_make_rollout`` is the only place that knows how the deployment is
wired; the tests themselves must keep passing unmodified.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import HTTPException

from relax.components.rollout import Rollout as RolloutDeployment


Rollout = RolloutDeployment.func_or_class

ROUTER_URL = "http://10.0.0.1:3000"
CHAT_BODY = json.dumps({"model": "not-a-registered-model", "messages": [{"role": "user", "content": "hi"}]}).encode()
COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "served",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class _Awaitable:
    def __init__(self, value):
        self._value = value

    def __await__(self):
        yield
        return self._value


class _Request:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None):
        self._body = body
        self.headers = headers or {}

    async def body(self) -> bytes:
        return self._body


def _make_rollout(monkeypatch, handler, *, router=("10.0.0.1", 3000)):
    """A shell Rollout deployment whose proxy talks to ``handler``.

    Returns ``(rollout, client_kwargs)``; ``client_kwargs`` collects the
    keyword arguments every proxy ``httpx.AsyncClient`` was created with.
    """
    client_kwargs: list[dict] = []
    real_async_client = httpx.AsyncClient

    def fake_async_client(**kwargs):
        client_kwargs.append(kwargs)
        return real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)

    rollout = object.__new__(Rollout)
    rollout._logger_instance = None
    rollout._proxy_client = None
    rollout._sglang_base_url = None
    address = {"router_ip": router[0], "router_port": router[1]}
    rollout.rollout_manager = type("_Manager", (), {})()
    rollout.rollout_manager.get_router_address = type(
        "_Call", (), {"remote": staticmethod(lambda *a, **k: _Awaitable(address))}
    )()
    return rollout, client_kwargs


async def _chat(rollout, body: bytes = CHAT_BODY, headers: dict[str, str] | None = None):
    return await rollout.chat_completions(_Request(body, headers))


async def test_rollout_chat_proxy_forwards_body_and_filters_hop_by_hop_headers(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(url=str(request.url), body=request.content, headers=dict(request.headers))
        return httpx.Response(200, json=COMPLETION)

    rollout, _client_kwargs = _make_rollout(monkeypatch, handler)
    headers = {
        "host": "serve-host",
        "connection": "keep-alive",
        "keep-alive": "timeout=5",
        "transfer-encoding": "chunked",
        "upgrade": "h2c",
        "authorization": "Bearer token",
        "x-trace-id": "abc",
    }

    response = await _chat(rollout, headers=headers)

    assert seen["url"] == f"{ROUTER_URL}/v1/chat/completions"
    # The body goes through untouched: the model name is not validated or rewritten.
    assert seen["body"] == CHAT_BODY
    assert seen["headers"]["authorization"] == "Bearer token"
    assert seen["headers"]["x-trace-id"] == "abc"
    assert seen["headers"]["host"] != "serve-host"
    for dropped in ("keep-alive", "upgrade"):
        assert dropped not in seen["headers"]
    assert response.id == "chatcmpl-1"
    assert response.choices[0].message.content == "hello"


async def test_rollout_chat_proxy_returns_upstream_status_and_text(monkeypatch):
    rollout, _client_kwargs = _make_rollout(monkeypatch, lambda request: httpx.Response(429, text="slow down"))

    with pytest.raises(HTTPException) as excinfo:
        await _chat(rollout)

    assert excinfo.value.status_code == 429
    assert excinfo.value.detail == "slow down"


async def test_rollout_chat_proxy_maps_connection_failure_to_502(monkeypatch):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    rollout, _client_kwargs = _make_rollout(monkeypatch, refuse)

    with pytest.raises(HTTPException) as excinfo:
        await _chat(rollout)

    assert excinfo.value.status_code == 502


async def test_rollout_chat_proxy_rejects_invalid_body_with_400(monkeypatch):
    rollout, _client_kwargs = _make_rollout(monkeypatch, lambda request: httpx.Response(200, json=COMPLETION))

    with pytest.raises(HTTPException) as excinfo:
        await _chat(rollout, body=b"{not json")

    assert excinfo.value.status_code == 400


async def test_rollout_chat_proxy_reports_503_without_router(monkeypatch):
    rollout, _client_kwargs = _make_rollout(
        monkeypatch, lambda request: httpx.Response(200, json=COMPLETION), router=(None, None)
    )

    with pytest.raises(HTTPException) as excinfo:
        await _chat(rollout)

    assert excinfo.value.status_code == 503


def _stream_body() -> bytes:
    return json.dumps({"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode()


async def _collect(response) -> list[str]:
    return [chunk async for chunk in response.body_iterator]


async def test_rollout_chat_proxy_streams_upstream_lines(monkeypatch):
    upstream = b'data: {"id": "1"}\n\ndata: {"id": "2"}\n\ndata: [DONE]\n\n'
    rollout, _client_kwargs = _make_rollout(monkeypatch, lambda request: httpx.Response(200, content=upstream))

    response = await _chat(rollout, body=_stream_body())

    assert response.media_type == "text/event-stream"
    assert await _collect(response) == ['data: {"id": "1"}\n\n', 'data: {"id": "2"}\n\n', "data: [DONE]\n\n"]


# On current main the two streaming error paths below raise NameError instead of
# emitting an error chunk: ``_make_error_chunk`` is defined after the ``Rollout``
# class, and ``@serve.ingress`` rebuilds the class against a copy of the module
# globals taken before that definition exists. The tests state the intended
# behavior; strict xfail makes them fail loudly once the bug is gone, so the
# marker has to be removed by the change that fixes it.
_STREAM_ERROR_CHUNK_BUG = pytest.mark.xfail(
    strict=True,
    raises=NameError,
    reason="_make_error_chunk is not visible to the ingress-rewritten Rollout class",
)


@_STREAM_ERROR_CHUNK_BUG
async def test_rollout_chat_proxy_stream_turns_upstream_error_into_error_chunk(monkeypatch):
    rollout, _client_kwargs = _make_rollout(monkeypatch, lambda request: httpx.Response(500, text="boom"))

    chunks = await _collect(await _chat(rollout, body=_stream_body()))

    assert len(chunks) == 2 and chunks[1] == "data: [DONE]\n\n"
    error = json.loads(chunks[0].removeprefix("data: "))["error"]
    assert error == {"code": 500, "message": "boom"}


@_STREAM_ERROR_CHUNK_BUG
async def test_rollout_chat_proxy_stream_turns_connection_failure_into_502_chunk(monkeypatch):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    rollout, _client_kwargs = _make_rollout(monkeypatch, refuse)

    chunks = await _collect(await _chat(rollout, body=_stream_body()))

    assert chunks[1] == "data: [DONE]\n\n"
    assert json.loads(chunks[0].removeprefix("data: "))["error"]["code"] == 502


async def test_rollout_chat_proxy_client_has_no_timeout_and_wide_pool(monkeypatch):
    rollout, client_kwargs = _make_rollout(monkeypatch, lambda request: httpx.Response(200, json=COMPLETION))

    await _chat(rollout)
    await _chat(rollout)

    # One client, reused; no read timeout; pool wide enough for a full rollout batch.
    assert len(client_kwargs) == 1
    limits = client_kwargs[0]["limits"]
    assert (limits.max_connections, limits.max_keepalive_connections, limits.keepalive_expiry) == (4096, 4096, 600)
    timeout = client_kwargs[0]["timeout"]
    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (None, None, None, None)
