# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from relax.engine.inference import InferenceError, InferenceRouter, SnapshotVersion, actor_identity
from relax.engine.inference_discovery import (
    initialize_discovery,
    rollout_snapshot,
    static_snapshot,
    weight_update_notification,
)
from relax.engine.inference_http import GatewayRuntime, UpstreamStreamingResponse, legacy_http
from relax.utils.inference_client import InferenceClient


def gateway_app(runtime: GatewayRuntime) -> FastAPI:
    app = FastAPI()

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
    async def dispatch(request: Request, path: str) -> Response:
        return await runtime.handle(request, path)

    return app


def snapshot(state="ready"):
    return {
        "models": {
            "judge": {
                "state": state,
                "router_url": None,
                "backend_model": "backend",
                "engines": [
                    {
                        "engine_id": "judge/0",
                        "base_url": "http://engine",
                        "state": state,
                        "direct_eligible": state == "ready",
                    }
                ],
            }
        },
        "routing": {"default_model": "judge", "route_keys": {"math": "judge"}, "aliases": {"backend": "judge"}},
    }


class Remote:
    def __init__(self, function):
        self.function = function

    async def remote(self, *args):
        result = self.function(*args)
        return await result if asyncio.iscoroutine(result) else result


class Source:
    def __init__(self):
        self.value = snapshot()["models"]
        self.version = SnapshotVersion()
        self.get_inference_snapshot = Remote(lambda: self.version.publish(self.value))


@pytest.mark.parametrize("model,route", [(None, None), ("judge", None), ("backend", None), (None, "math")])
def test_inference_routing_resolves_model_and_alias(model, route):
    assert InferenceRouter().select(snapshot(), model, route).model == "judge"


def test_inference_routing_never_falls_back_from_invalid_or_sleeping_model():
    router = InferenceRouter()
    for model, route in [("wrong", "math"), (None, "wrong")]:
        with pytest.raises(InferenceError) as exc:
            router.select(snapshot(), model, route)
        assert exc.value.status_code == 400
    with pytest.raises(InferenceError) as exc:
        router.select(snapshot("sleeping"))
    assert exc.value.status_code == 503


def test_inference_pd_requires_router():
    data = snapshot()
    data["models"]["judge"].update(router_required=True)
    with pytest.raises(InferenceError):
        InferenceRouter().select(data)
    data["models"]["judge"]["router_url"] = "http://router"
    assert InferenceRouter().select(data).base_url == "http://router"


def test_inference_rollout_discovery_filters_followers_and_blocked_slots():
    group = SimpleNamespace(
        all_engines=[object(), object()],
        inference_urls={0: "http://engine"},
        inference_blocked=set(),
        lifecycle_status="ACTIVE",
        sglang_overrides={"model_path": "backend"},
        nodes_per_engine=2,
        rank_offset=0,
        worker_type="regular",
        num_gpus_per_engine=16,
        num_new_engines=0,
    )
    server = SimpleNamespace(engine_groups=[group], router_ip="router", router_port=80)
    manager = SimpleNamespace(servers={"judge": server}, args=SimpleNamespace())
    initialize_discovery(manager)
    manager._inference_state = "ready"
    initial = rollout_snapshot(manager)
    assert len(initial["models"]["judge"]["engines"]) == 1
    assert initial["models"]["judge"]["total_engines"] == 2
    group.inference_blocked.add(1)
    blocked = rollout_snapshot(manager)
    assert not blocked["models"]["judge"]["engines"][0]["direct_eligible"]
    group.inference_blocked.clear()
    group.all_engines[0] = object()
    recovered = rollout_snapshot(manager)
    assert recovered["revision"] > initial["revision"]
    manager._inference_state = "sleeping"
    manager._training_weight_updating = True
    assert rollout_snapshot(manager)["models"]["judge"]["state"] != "ready"
    manager._training_weight_updating = False
    assert rollout_snapshot(manager)["models"]["judge"]["state"] == "sleeping"


def test_inference_teacher_names_and_static_state():
    manager = SimpleNamespace(
        args=SimpleNamespace(_inference_model_name="math"),
        all_engines=[object()],
        nodes_per_engine=1,
        _engine_addr_and_ports={0: {"host": "::1", "port": 80}},
    )
    initialize_discovery(manager)
    manager._inference_state = "onloading"
    result = static_snapshot(manager)
    engine = result["models"]["math"]["engines"][0]
    assert engine["base_url"] == "http://[::1]:80"
    assert not engine["direct_eligible"]
    result["models"]["math"]["engines"].clear()
    assert static_snapshot(manager)["models"]["math"]["engines"]


@pytest.mark.asyncio
async def test_inference_gateway_proxy_and_direct_share_routing():
    source = Source()
    sent = []

    async def engine(request):
        import json

        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"text": "answer", "extension": [1, 2]})

    runtime = GatewayRuntime("teacher", client=httpx.AsyncClient(transport=httpx.MockTransport(engine)))
    runtime.rebind([{"manager": source}])
    app_transport = httpx.ASGITransport(app=gateway_app(runtime))

    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if request.url.host == "gateway":
                return await app_transport.handle_async_request(request)
            return await httpx.MockTransport(engine).handle_async_request(request)

    for mode in ("gateway", "direct"):
        async with InferenceClient("http://gateway", mode=mode, transport=Transport()) as client:
            assert await client.generate({"route_key": "judge", "input_ids": [1]}) == {
                "text": "answer",
                "extension": [1, 2],
            }
            await client.chat_completions({"model": "judge", "messages": []})
    assert sent[0] == sent[2] == {"input_ids": [1]}
    assert sent[1] == sent[3] == {"model": "backend", "messages": []}
    source.value["judge"]["state"] = "sleeping"
    async with httpx.AsyncClient(transport=app_transport, base_url="http://gateway") as client:
        response = await client.post("/generate", json={"input_ids": [1]})
        assert response.status_code == 503
        assert response.headers["retry-after"]
        health = await client.get("/health")
        assert health.status_code == 200 and health.json()["gateway_status"] == "healthy"
    await runtime.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("routing", [{"model": []}, {"route_key": {}}])
async def test_inference_gateway_and_direct_reject_invalid_routing_types(routing):
    def unexpected_engine_call(request):
        pytest.fail("Invalid routing must not reach an engine")

    runtime = GatewayRuntime(
        "rollout", client=httpx.AsyncClient(transport=httpx.MockTransport(unexpected_engine_call))
    )
    runtime.rebind([{"manager": Source()}])
    transport = httpx.ASGITransport(app=gateway_app(runtime))
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        response = await client.post("/generate", json={**routing, "input_ids": [1]})
        assert response.status_code == 400
    async with InferenceClient("http://gateway", mode="direct", transport=transport) as client:
        with pytest.raises(InferenceError) as exc:
            await client.generate({**routing, "input_ids": [1]})
        assert exc.value.status_code == 400
    await runtime.aclose()


@pytest.mark.asyncio
async def test_inference_rebind_rejects_late_snapshot():
    started, finish = asyncio.Event(), asyncio.Event()
    source = Source()

    async def delayed():
        started.set()
        await finish.wait()
        return source.version.publish(source.value)

    source.get_inference_snapshot = Remote(delayed)
    runtime = GatewayRuntime("rollout")
    runtime.rebind([{"manager": source}])
    pending = asyncio.create_task(runtime.snapshot())
    await started.wait()
    runtime.rebind([{"manager": Source()}])
    finish.set()
    with pytest.raises(InferenceError):
        await pending
    assert (await runtime.snapshot())["models"]["judge"]["state"] == "ready"
    await runtime.aclose()


@pytest.mark.asyncio
async def test_inference_legacy_preserves_fastapi_validation_and_errors():
    app = FastAPI()

    class Scale(BaseModel):
        num_replicas: int

    class Backend:
        async def scale(self, request: Scale):
            if request.num_replicas < 1:
                raise HTTPException(409, "conflict")
            return {"count": request.num_replicas}

    backend = Backend()
    app.post("/scale_out")(backend.scale)
    assert (await legacy_http(app, "POST", "/scale_out", "", b"{}"))["status"] == 422
    assert (await legacy_http(app, "POST", "/scale_out", "", b'{"num_replicas": 0}'))["status"] == 409
    result = await legacy_http(app, "POST", "/scale_out", "", b'{"num_replicas": 2}')
    assert result["status"] == 200 and result["body"] == b'{"count":2}'


@pytest.mark.asyncio
async def test_inference_upstream_error_is_not_replayed():
    calls = []

    async def unavailable(request):
        calls.append(request)
        return httpx.Response(429, json={"error": {"message": "limited"}}, headers={"retry-after": "5"})

    runtime = GatewayRuntime("rollout", client=httpx.AsyncClient(transport=httpx.MockTransport(unavailable)))
    runtime.rebind([{"manager": Source()}])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway_app(runtime)), base_url="http://gateway"
    ) as c:
        response = await c.post("/generate", json={"input_ids": [1], "stream": True})
        assert response.status_code == 429
        assert response.json()["error"]["message"] == "limited"
        assert response.headers["retry-after"] == "5"
    assert len(calls) == 1
    await runtime.aclose()


def test_inference_weight_completion_cannot_unlock_replacement_or_partial_onload():
    old, replacement = object(), object()
    group = SimpleNamespace(
        all_engines=[old, object()],
        inference_urls={0: "http://a", 1: "http://b"},
        nodes_per_engine=1,
        rank_offset=0,
        lifecycle_status="ACTIVE",
        sglang_overrides={},
        worker_type="regular",
        num_gpus_per_engine=1,
        num_new_engines=0,
    )
    manager = SimpleNamespace(
        servers={"actor": SimpleNamespace(engine_groups=[group], router_ip="router", router_port=80)},
        args=SimpleNamespace(),
        rollout_engines=group.all_engines,
    )
    initialize_discovery(manager)
    manager._inference_state = "ready"
    token = weight_update_notification(manager)
    assert rollout_snapshot(manager)["models"]["actor"]["state"] == "onloading"
    group.all_engines[0] = replacement
    manager._inference_pending_weights.add(actor_identity(replacement))
    weight_update_notification(manager, token)
    # Even though the second head is ready, the router could select the replacement.
    assert rollout_snapshot(manager)["models"]["actor"]["state"] == "onloading"
    token = weight_update_notification(manager)
    manager._inference_state = "onloading"
    weight_update_notification(manager, token)
    assert rollout_snapshot(manager)["models"]["actor"]["state"] == "onloading"
    manager._inference_state = "ready"
    assert rollout_snapshot(manager)["models"]["actor"]["state"] == "ready"
    token = weight_update_notification(manager)
    # DCS may prune an existing, formerly ready head before broadcasting.
    weight_update_notification(
        manager,
        {
            **token,
            "engines": token["engines"][:1],
            "unconfirmed_engines": token["engines"][1:],
        },
    )
    assert rollout_snapshot(manager)["models"]["actor"]["state"] == "onloading"
    weight_update_notification(manager)
    with pytest.raises(RuntimeError, match="Stale"):
        weight_update_notification(manager, token)
    assert manager._inference_weight_updating


def test_inference_pd_requires_both_live_pools_and_preserves_sleep_state():
    groups = [
        SimpleNamespace(
            all_engines=[object()],
            inference_urls={rank: f"http://engine{rank}"},
            nodes_per_engine=1,
            rank_offset=rank,
            lifecycle_status="ACTIVE",
            sglang_overrides={},
            worker_type=worker,
            num_gpus_per_engine=1,
            num_new_engines=0,
        )
        for rank, worker in enumerate(("prefill", "decode"))
    ]
    manager = SimpleNamespace(
        servers={"actor": SimpleNamespace(engine_groups=groups, router_ip="router", router_port=80)},
        args=SimpleNamespace(),
    )
    initialize_discovery(manager)
    manager._inference_state = "ready"
    assert rollout_snapshot(manager)["models"]["actor"]["state"] == "ready"
    groups[0].all_engines[0] = None
    assert rollout_snapshot(manager)["models"]["actor"]["state"] == "unavailable"
    manager._inference_state = "sleeping"
    assert rollout_snapshot(manager)["models"]["actor"]["state"] == "sleeping"


def test_inference_teacher_does_not_advertise_genrm_model_alias():
    manager = SimpleNamespace(
        args=SimpleNamespace(genrm_engine_config={"served_model_name": "judge"}),
        _overrides={"model_path": "teacher"},
        _teacher_args=SimpleNamespace(),
        all_engines=[object()],
        nodes_per_engine=1,
        _engine_addr_and_ports={0: {"host": "localhost", "port": 80}},
    )
    initialize_discovery(manager)
    manager._inference_state = "ready"
    assert static_snapshot(manager)["models"]["default"]["backend_model"] == "teacher"


@pytest.mark.asyncio
async def test_inference_multi_model_requires_explicit_unambiguous_route():
    runtime = GatewayRuntime("teacher")
    a, b = Source(), Source()
    runtime.rebind([{"manager": a, "model": "math"}, {"manager": b, "model": "code"}])
    data = await runtime.snapshot()
    assert data["routing"]["aliases"] == {}
    with pytest.raises(InferenceError):
        InferenceRouter().select(data)
    assert InferenceRouter().select(data, "code", "math").model == "code"
    old_revision = data["topology_revision"]

    def unreachable():
        raise RuntimeError("manager restarted")

    a.get_inference_snapshot = Remote(unreachable)
    data = await runtime.snapshot()
    assert data["models"]["math"]["state"] == "unavailable"
    assert data["models"]["code"]["state"] == "ready"
    assert data["topology_revision"] > old_revision
    await runtime.aclose()


class TrackedStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        yield b'data: {"text":"first"}\n\n'
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_on", ["http.response.start", "http.response.body"])
async def test_inference_sse_disconnect_closes_upstream(fail_on):
    stream = TrackedStream()
    upstream = httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    async def send(message):
        if message["type"] == fail_on:
            raise OSError("client disconnected")

    async def receive():
        return {"type": "http.disconnect"}

    response = UpstreamStreamingResponse(upstream)
    with pytest.raises(Exception):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert stream.closed and upstream.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["gateway", "direct"])
async def test_inference_client_early_stream_close_and_fresh_discovery(mode):
    streams, discovery_calls = [], []
    data = snapshot()

    async def handler(request):
        if request.url.path == "/engines":
            discovery_calls.append(request)
            return httpx.Response(200, json=data)
        stream = TrackedStream()
        streams.append(stream)
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    async with InferenceClient("http://gateway", mode=mode, transport=httpx.MockTransport(handler)) as client:
        for _ in range(2):
            stream = client.stream_generate({"input_ids": [1]})
            assert b"first" in await anext(stream)
            await stream.aclose()
            assert streams[-1].closed
        if mode == "direct":
            data["models"]["judge"]["state"] = "sleeping"
            with pytest.raises(InferenceError):
                await client.generate({"input_ids": [1]})
            assert len(discovery_calls) == 3
            assert len(streams) == 2


@pytest.mark.asyncio
async def test_inference_legacy_redirect_keeps_public_role_prefix():
    app = FastAPI()

    class Backend:
        def get_step(self):
            return {"step": 3}

    backend = Backend()
    app.get("/get_step")(backend.get_step)
    backend.inference_legacy_http = Remote(lambda *args: legacy_http(app, *args))
    runtime = GatewayRuntime("rollout")
    runtime.rebind([], backend)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway_app(runtime)), base_url="http://gateway"
    ) as c:
        response = await c.get("/get_step/?test=1")
        assert response.status_code == 307
        assert response.headers["location"] == "/rollout/get_step?test=1"
    await runtime.aclose()
