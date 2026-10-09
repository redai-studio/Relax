# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""``InferenceGateway``: discovery answers, snapshot-routed requests and what a
caller sees when a model cannot take requests."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import ClientDisconnect

from relax.engine.inference.discovery import EngineState, RoleSnapshot, build_model_snapshot
from relax.engine.inference.gateway import InferenceGateway, build_forward_headers


READY, SLEEPING = EngineState.READY, EngineState.SLEEPING


def _role(*models, revision=1, role="teacher", **kwargs):
    return RoleSnapshot(role=role, topology_revision=revision, models=models, **kwargs)


def _model(name, states=(READY, READY), host=None):
    host = host or name
    return build_model_snapshot(name, [(i, f"http://{host}-{i}:1", state) for i, state in enumerate(states)])


class _Manager:
    """Stands in for the role's manager: serves snapshots, and would record any
    attempt to change engine state."""

    def __init__(self, *snapshots):
        self.snapshots = list(snapshots)
        self.fetches = 0
        self.state_changes: list[str] = []

    def fetch(self):
        reply = self.snapshots[min(self.fetches, len(self.snapshots) - 1)]
        self.fetches += 1
        return reply

    def onload(self):
        self.state_changes.append("onload")


@pytest.fixture
def upstream(monkeypatch):
    """Route the gateway's proxy client into ``handler``; records requests."""
    seen: list[httpx.Request] = []
    state = {"handler": lambda request: httpx.Response(200, json={"text": "ok"})}
    real_async_client = httpx.AsyncClient

    def dispatch(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return state["handler"](request)

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: real_async_client(transport=httpx.MockTransport(dispatch))
    )

    def use(handler):
        state["handler"] = handler

    return type("_Upstream", (), {"seen": seen, "use": staticmethod(use)})


def _gateway(manager, **kwargs):
    return InferenceGateway("teacher", manager.fetch, **kwargs)


# ----------------------------------------------------------------------
# Discovery.
# ----------------------------------------------------------------------


async def test_gateway_engines_returns_the_role_snapshot():
    snapshot = _role(_model("math"), revision=4)

    assert await _gateway(_Manager(snapshot)).engines() == snapshot.to_dict()


async def test_gateway_health_answers_while_every_engine_sleeps():
    manager = _Manager(_role(_model("math", states=(SLEEPING, SLEEPING)), phase="generate"))

    health = await _gateway(manager).health()

    assert health == {
        "status": "ok",
        "role": "teacher",
        "phase": "generate",
        "topology_revision": 1,
        "models": {"math": "sleeping"},
    }
    assert manager.state_changes == []


async def test_gateway_models_lists_every_model():
    listing = await _gateway(_Manager(_role(_model("quality"), _model("safety")))).models()

    assert listing["object"] == "list"
    assert [item["id"] for item in listing["data"]] == ["quality", "safety"]


# ----------------------------------------------------------------------
# Snapshot-routed requests.
# ----------------------------------------------------------------------


async def test_gateway_generate_forwards_native_payload_to_a_replica(upstream):
    gateway = _gateway(_Manager(_role(_model("math"), _model("code"))))

    first = await gateway.generate({"model": "math", "input_ids": [1, 2], "sampling_params": {"temperature": 0}})
    await gateway.generate({"route_key": "math", "input_ids": [3]})

    assert first == {"text": "ok"}
    assert [str(request.url) for request in upstream.seen] == ["http://math-0:1/generate", "http://math-1:1/generate"]
    # Model selection fields are the gateway's, not the engine's.
    assert json.loads(upstream.seen[0].content) == {"input_ids": [1, 2], "sampling_params": {"temperature": 0}}
    assert json.loads(upstream.seen[1].content) == {"input_ids": [3]}


class _NativeStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False
        self.chunks = [b'event: token\ndata: {"text":', b' "hello"}\n\ndata: [DONE]\n\n']

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("role", ["rollout", "genrm", "teacher"])
async def test_gateway_native_stream_preserves_sse_and_closes_upstream(upstream, role):
    stream = _NativeStream()
    upstream.use(lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    gateway = InferenceGateway(role, _Manager(_role(_model("math"), role=role)).fetch)

    response = await gateway.generate({"model": "math", "input_ids": [1], "stream": True})

    assert not stream.closed
    assert b"".join([chunk async for chunk in response.body_iterator]) == b"".join(stream.chunks)
    assert response.headers["content-type"] == "text/event-stream"
    assert stream.closed
    assert str(upstream.seen[0].url) == "http://math-0:1/generate"
    assert json.loads(upstream.seen[0].content) == {"input_ids": [1], "stream": True}


async def test_gateway_native_stream_closes_when_client_disconnects_before_body(upstream):
    stream = _NativeStream()
    upstream.use(lambda request: httpx.Response(200, stream=stream))
    response = await _gateway(_Manager(_role(_model("math")))).generate({"input_ids": [1], "stream": True})

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        raise OSError("client disconnected")

    with pytest.raises(ClientDisconnect):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

    assert stream.closed


async def test_gateway_native_stream_returns_upstream_error_before_streaming(upstream):
    upstream.use(lambda request: httpx.Response(422, text="bad sampling params"))
    gateway = _gateway(_Manager(_role(_model("math"))))

    with pytest.raises(HTTPException) as excinfo:
        await gateway.generate({"input_ids": [1], "stream": True})

    assert (excinfo.value.status_code, excinfo.value.detail) == (422, "bad sampling params")


async def test_gateway_unknown_model_is_a_400_listing_models(upstream):
    gateway = _gateway(_Manager(_role(_model("math"), _model("code"))))

    with pytest.raises(HTTPException) as excinfo:
        await gateway.generate({"model": "physics", "input_ids": [1]})

    assert excinfo.value.status_code == 400
    assert "['math', 'code']" in excinfo.value.detail
    assert upstream.seen == []


async def test_gateway_sleeping_model_is_a_503_and_is_not_woken(upstream):
    manager = _Manager(_role(_model("math", states=(SLEEPING, SLEEPING))))
    gateway = _gateway(manager)

    with pytest.raises(HTTPException) as excinfo:
        await gateway.chat_completions(json.dumps({"model": "math", "messages": []}).encode(), {})

    assert excinfo.value.status_code == 503
    assert excinfo.value.headers == {"Retry-After": "5"}
    # Nothing was sent upstream and the manager was only ever asked for snapshots.
    assert upstream.seen == []
    assert manager.state_changes == []


async def test_gateway_chat_completions_routes_by_body_and_forwards_it_unmodified(upstream):
    upstream.use(lambda request: httpx.Response(200, json={"id": "chatcmpl-1"}))
    gateway = _gateway(_Manager(_role(_model("math"), _model("code"))))
    body = json.dumps({"model": "code", "messages": [{"role": "user", "content": "hi"}]}).encode()

    response = await gateway.chat_completions(body, {"authorization": "Bearer t", "connection": "keep-alive"})

    assert response == {"id": "chatcmpl-1"}
    request = upstream.seen[0]
    assert str(request.url) == "http://code-0:1/v1/chat/completions"
    assert request.content == body
    assert request.headers["authorization"] == "Bearer t"


async def test_gateway_chat_completions_streams(upstream):
    upstream.use(lambda request: httpx.Response(200, content=b'data: {"id": "1"}\n\ndata: [DONE]\n\n'))
    gateway = _gateway(_Manager(_role(_model("math"))))
    body = json.dumps({"stream": True, "messages": []}).encode()

    response = await gateway.chat_completions(body, {})

    assert response.media_type == "text/event-stream"
    assert [chunk async for chunk in response.body_iterator] == ['data: {"id": "1"}\n\n', "data: [DONE]\n\n"]


async def test_gateway_chat_completions_rejects_malformed_body():
    gateway = _gateway(_Manager(_role(_model("math"))))

    for body in (b"{not json", b"[1, 2]"):
        with pytest.raises(HTTPException) as excinfo:
            await gateway.chat_completions(body, {})
        assert excinfo.value.status_code == 400


async def test_gateway_passes_through_upstream_status(upstream):
    upstream.use(lambda request: httpx.Response(422, text="bad sampling params"))
    gateway = _gateway(_Manager(_role(_model("math"))))

    with pytest.raises(HTTPException) as excinfo:
        await gateway.generate({"input_ids": [1]})

    assert (excinfo.value.status_code, excinfo.value.detail) == (422, "bad sampling params")


async def test_gateway_refreshes_topology_after_an_unreachable_replica(upstream):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host.startswith("old"):
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={"text": "ok"})

    upstream.use(handler)
    manager = _Manager(
        _role(_model("math", states=(READY,), host="old"), revision=1),
        _role(_model("math", states=(READY,), host="new"), revision=2),
    )
    gateway = _gateway(manager, snapshot_max_age_s=3600)

    with pytest.raises(HTTPException) as excinfo:
        await gateway.generate({"input_ids": [1]})
    assert excinfo.value.status_code == 502

    assert await gateway.generate({"input_ids": [1]}) == {"text": "ok"}
    assert [request.url.host for request in upstream.seen] == ["old-0", "new-0"]


async def test_gateway_snapshot_is_reused_within_its_max_age():
    now = {"t": 100.0}
    manager = _Manager(_role(_model("math")))
    gateway = _gateway(manager, snapshot_max_age_s=2.0, clock=lambda: now["t"])

    await gateway.health()
    await gateway.health()
    assert manager.fetches == 1

    now["t"] += 2.0
    await gateway.health()
    assert manager.fetches == 2


def test_gateway_forward_headers_drop_hop_by_hop_only():
    headers = {"Host": "serve", "Connection": "close", "Keep-Alive": "1", "Upgrade": "h2c", "X-Trace": "1"}

    assert build_forward_headers(headers) == {"X-Trace": "1"}
