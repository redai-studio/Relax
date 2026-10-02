# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.requests import Request

from relax.components.inference_gateway import InferenceGateway
from relax.engine.inference.types import (
    LifecycleState,
    ModelSnapshot,
    ReplicaSnapshot,
    Role,
    RoleSnapshot,
    RoutingSpec,
)


def _snapshot(*, role: Role = Role.ROLLOUT, state: LifecycleState = LifecycleState.READY, admission: bool = True):
    return RoleSnapshot(
        role=role,
        manager_epoch="epoch-a",
        models=(
            ModelSnapshot(
                "model-a",
                (ReplicaSnapshot("model-a/replica-0", LifecycleState.STARTING, "http://engine/generate"),),
                router_url="http://router",
                state=state,
                admission=admission,
            ),
        ),
    )


class _Owner:
    """A task manager handle whose snapshot comes from ``provider``."""

    def __init__(self, provider) -> None:
        self.inflight: set[str] = set()

        async def snapshot(role):
            return provider()

        async def admit_request(role, model_id, request_id):
            self.inflight.add(request_id)
            return request_id

        async def complete_request(request_id):
            self.inflight.discard(request_id)

        self.snapshot = SimpleNamespace(remote=snapshot)
        self.admit_request = SimpleNamespace(remote=admit_request)
        self.complete_request = SimpleNamespace(remote=complete_request)


def _owner(provider):
    return _Owner(provider)


def test_gateway_requires_the_task_owner_handle() -> None:
    with pytest.raises(ValueError, match="task inference manager"):
        InferenceGateway(Role.ROLLOUT, manager_handle=None)


@pytest.mark.asyncio
async def test_one_gateway_class_serves_all_roles() -> None:
    gateways = [InferenceGateway(role, manager_handle=_owner(lambda role=role: _snapshot(role=role))) for role in Role]
    try:
        assert {gateway.role for gateway in gateways} == set(Role)
        model_lists = [await gateway.models() for gateway in gateways]
        assert all(models["data"][0]["id"] == "model-a" for models in model_lists)
    finally:
        for gateway in gateways:
            await gateway.close()


@pytest.mark.asyncio
async def test_gateway_reuses_discovery_routing_and_rejects_unavailable_models() -> None:
    ready = InferenceGateway(Role.ROLLOUT, manager_handle=_owner(lambda: _snapshot()))
    sleeping = InferenceGateway(
        Role.ROLLOUT,
        manager_handle=_owner(lambda: _snapshot(state=LifecycleState.SLEEPING, admission=False)),
    )
    try:
        assert await ready._target({"model": "model-a"}) == "http://router"
        with pytest.raises(Exception) as error:
            await sleeping._target({"model": "model-a"})
        assert error.value.status_code == 503
        assert "Retry-After" in (error.value.headers or {})
    finally:
        await ready.close()
        await sleeping.close()


@pytest.mark.asyncio
async def test_gateway_single_model_role_serves_any_openai_model_name() -> None:
    """A single-model role registers as "default"; OpenAI clients send the real
    name and must keep being served, while a multi-model role stays strict."""
    single = replace(_snapshot(), routing=RoutingSpec(default_model="model-a"))
    second = replace(single.models[0], model_id="model-b")
    multi = replace(single, models=(*single.models, second))
    single_gateway = InferenceGateway(Role.ROLLOUT, manager_handle=_owner(lambda: single))
    multi_gateway = InferenceGateway(Role.ROLLOUT, manager_handle=_owner(lambda: multi))
    try:
        assert await single_gateway._resolve_target({"model": "Qwen3-8B"}) == ("http://router", "model-a")
        with pytest.raises(Exception) as error:
            await multi_gateway._target({"model": "Qwen3-8B"})
        assert error.value.status_code == 400
        assert error.value.detail["code"] == "unknown_model"
    finally:
        await single_gateway.close()
        await multi_gateway.close()


@pytest.mark.asyncio
async def test_gateway_admission_race_returns_retryable_503_without_forwarding() -> None:
    owner = _owner(lambda: _snapshot(role=Role.TEACHER))

    async def reject(*_args):
        raise RuntimeError("model offloaded after snapshot")

    owner.admit_request.remote = reject
    gateway = InferenceGateway(Role.TEACHER, manager_handle=owner)

    def upstream(_request):
        pytest.fail("rejected requests must not reach the offloaded engine")

    await gateway._client.aclose()
    gateway._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        response = await gateway.proxy(_request({"model": "model-a", "input_ids": [1]}), "generate")
        assert response.status_code == 503
        assert response.headers["retry-after"] == "1"
        assert json.loads(response.body)["error"]["code"] == "unavailable"
        assert not owner.inflight
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_gateway_never_uses_replica_url_when_router_is_missing() -> None:
    snapshot = _snapshot()
    model = replace(
        snapshot.models[0],
        router_url=None,
        replicas=(replace(snapshot.models[0].replicas[0], state=LifecycleState.READY),),
    )
    gateway = InferenceGateway(Role.ROLLOUT, manager_handle=_owner(lambda: replace(snapshot, models=(model,))))
    try:
        with pytest.raises(Exception) as error:
            await gateway._target({"model": "model-a"})
        assert error.value.status_code == 503
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_gateway_rotates_across_direct_eligible_replicas() -> None:
    snapshot = _snapshot()
    model = replace(
        snapshot.models[0],
        router_url=None,
        replicas=tuple(
            ReplicaSnapshot(
                f"model-a/replica-{index}", LifecycleState.READY, f"http://engine-{index}", direct_eligible=True
            )
            for index in range(2)
        ),
    )
    gateway = InferenceGateway(Role.TEACHER, manager_handle=_owner(lambda: replace(snapshot, models=(model,))))
    try:
        targets = [await gateway._target({"model": "model-a"}) for _ in range(3)]
        assert targets == ["http://engine-0", "http://engine-1", "http://engine-0"]
    finally:
        await gateway.close()


def _request(payload: dict, headers: tuple[tuple[bytes, bytes], ...] = ()) -> Request:
    body = json.dumps(payload).encode()

    async def receive():
        return {"type": "http.request", "body": body}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/generate",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                *headers,
            ],
        },
        receive,
    )


@pytest.mark.asyncio
async def test_gateway_teacher_forwards_the_router_routing_key() -> None:
    gateway = InferenceGateway(Role.TEACHER, manager_handle=_owner(lambda: _snapshot(role=Role.TEACHER)))
    sent = []

    def upstream(request):
        sent.append(request.headers.get("x-smg-routing-key"))
        return httpx.Response(200, json={"meta_info": {}})

    await gateway._client.aclose()
    gateway._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        request = _request({"input_ids": [1], "model": "model-a"}, ((b"x-smg-routing-key", b"7"),))
        response = await gateway.proxy(request, "generate")
        assert response.status_code == 200
        # The teacher Router pins a GRPO group to one replica by this key.
        assert sent == ["7"]
    finally:
        await gateway.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_gateway_genrm_router_adapts_only_messages(native: bool) -> None:
    adapt = AsyncMock(return_value={"input_ids": [11, 12], "sampling_params": {"temperature": 0.3}})
    gateway = InferenceGateway(
        Role.GENRM,
        manager_handle=_owner(lambda: _snapshot(role=Role.GENRM)),
        genrm_backend_handle=SimpleNamespace(prepare_generate_payload=SimpleNamespace(remote=adapt)),
    )
    sent = []

    def upstream(request):
        body = json.loads(request.content)
        assert body.pop("rid")
        sent.append(body)
        assert request.url == "http://router/generate"
        assert int(request.headers["content-length"]) == len(request.content)
        return httpx.Response(200, json={"text": " score \n", "meta_info": {"id": "result"}})

    await gateway._client.aclose()
    gateway._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    content = {"text": "judge"} if native else {"messages": [{"role": "user", "content": "judge"}]}
    payload = {**content, "model": "model-a", "route_key": "model-a", "sampling_params": {"temperature": 0.3}}
    try:
        response = await gateway.proxy(_request(payload), "generate")
        assert response.status_code == 200
        assert int(response.headers["content-length"]) == len(response.body)
        assert "route_key" not in sent[0] and "model" not in sent[0]
        if native:
            adapt.assert_not_awaited()
            assert sent[0] == {**content, "sampling_params": {"temperature": 0.3}}
            assert json.loads(response.body) == {"text": " score \n", "meta_info": {"id": "result"}}
        else:
            adapt.assert_awaited_once_with("model-a", content["messages"], {"temperature": 0.3})
            assert sent[0] == {"input_ids": [11, 12], "sampling_params": {"temperature": 0.3}}
            assert json.loads(response.body) == {"response": "score"}
    finally:
        await gateway.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["input_ids", "text"])
async def test_gateway_genrm_rejects_mixed_protocols(field: str) -> None:
    gateway = InferenceGateway(Role.GENRM, manager_handle=_owner(lambda: _snapshot(role=Role.GENRM)))
    try:
        response = await gateway.proxy(_request({"messages": [], field: None}), "generate")
        assert response.status_code == 400
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_gateway_genrm_rejects_missing_router_without_backend_fallback() -> None:
    snapshot = _snapshot(role=Role.GENRM)
    snapshot = replace(snapshot, models=(replace(snapshot.models[0], router_url=None),))
    adapt = AsyncMock()
    gateway = InferenceGateway(
        Role.GENRM,
        manager_handle=_owner(lambda: snapshot),
        upstream_url="http://backend",
        genrm_backend_handle=SimpleNamespace(prepare_generate_payload=SimpleNamespace(remote=adapt)),
    )
    messages = [{"role": "user", "content": "judge"}]

    def upstream(request):
        raise AssertionError("Gateway must not bypass the Router")

    await gateway._client.aclose()
    gateway._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        response = await gateway.proxy(_request({"messages": messages, "model": "model-a"}), "generate")
        assert response.status_code == 503
        adapt.assert_not_awaited()
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_gateway_reads_one_task_manager_snapshot() -> None:
    snapshot = _snapshot(role=Role.TEACHER)
    manager = _owner(lambda: snapshot)
    gateway = InferenceGateway(Role.TEACHER, manager_handle=manager)
    try:
        first = await gateway._snapshot()
        assert first is snapshot
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_gateway_teacher_native_strips_routing_metadata() -> None:
    gateway = InferenceGateway(Role.TEACHER, manager_handle=_owner(lambda: _snapshot(role=Role.TEACHER)))

    def upstream(request):
        body = json.loads(request.content)
        assert body.pop("rid")
        assert body == {"input_ids": [1, 2], "return_logprob": True}
        return httpx.Response(200, json={"text": "", "meta_info": {"input_token_logprobs": []}})

    await gateway._client.aclose()
    gateway._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        response = await gateway.proxy(
            _request({"model": "model-a", "route_key": "model-a", "input_ids": [1, 2], "return_logprob": True}),
            "generate",
        )
        assert response.status_code == 200
        assert "meta_info" in json.loads(response.body)
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_gateway_engines_defaults_to_common_schema_and_keeps_legacy_opt_in() -> None:
    gateway = InferenceGateway(Role.GENRM, manager_handle=_owner(lambda: _snapshot(role=Role.GENRM)))
    try:
        default = await gateway.engines()
        assert default == await gateway.engines(schema_version=2)
        assert default["schema_version"] == 2 and default["role"] == "genrm"
        legacy = await gateway.engines(schema_version=1)
        assert legacy["models"]["model-a"]["total_engines"] == 1
        with pytest.raises(Exception) as error:
            await gateway.engines(schema_version=3)
        assert error.value.status_code == 400
        assert json.loads((await gateway.health()).body)["status"] == "healthy"
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_gateway_health_reports_manager_unavailable() -> None:
    gateway = InferenceGateway(
        Role.TEACHER, manager_handle=_owner(lambda: (_ for _ in ()).throw(RuntimeError("down")))
    )
    try:
        response = await gateway.health()
        assert response.status_code == 503
        assert json.loads(response.body)["status"] == "unavailable"
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_gateway_completes_admitted_requests_and_aborts_failed_ones() -> None:
    owner = _owner(lambda: _snapshot())
    gateway = InferenceGateway(Role.ROLLOUT, manager_handle=owner)
    aborted = []

    def upstream(request):
        if request.url.path == "/generate" and request.url.host == "router":
            raise httpx.ReadTimeout("lost", request=request)
        if request.url.path == "/workers":
            return httpx.Response(200, json={"workers": [{"url": "http://engine"}]})
        aborted.append(json.loads(request.content)["rid"])
        return httpx.Response(200, json={})

    await gateway._client.aclose()
    gateway._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        response = await gateway.proxy(_request({"model": "model-a", "input_ids": [1]}), "generate")
        assert response.status_code == 502
        assert len(aborted) == 1
        assert owner.inflight == set()
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_gateway_deployment_routes_reach_the_serving_replica(monkeypatch) -> None:
    from ray import serve

    from relax.components.inference_gateway import InferenceGatewayDeployment

    # Build the replica the way Serve does; its routes are served from the copy
    # of the app that ``serve.ingress`` made, not from the module-level one.
    cls = InferenceGatewayDeployment.func_or_class

    class TestReplica(cls):
        def __del__(self):
            # Python GC cannot await Serve's asynchronous destructor.
            pass

    replica = TestReplica.__new__(TestReplica)
    await replica.__init__(Role.ROLLOUT, manager_handle=_owner(_snapshot))
    monkeypatch.setattr(serve, "get_replica_context", lambda: SimpleNamespace(servable_object=replica))
    await replica._run_asgi_lifespan_startup()
    try:
        transport = httpx.ASGITransport(app=replica._asgi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/engines")).json()["models"]
            assert (await client.get("/v1/models")).json()["data"][0]["id"] == "model-a"
    finally:
        await replica.close()
        await cls.__del__(replica)


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect", ["before_first", "after_first", "send_failure"])
async def test_gateway_stream_disconnect_completes_admission(disconnect: str) -> None:
    owner = _owner(_snapshot)
    owner.inflight.add("disconnected")
    finished = asyncio.Event()
    sent = asyncio.Event()

    async def complete(request_id):
        owner.inflight.remove(request_id)
        finished.set()

    owner.complete_request.remote = complete
    gateway = InferenceGateway(Role.ROLLOUT, manager_handle=owner)
    gateway._abort_upstream = AsyncMock()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: first\n\n"
            await asyncio.Event().wait()

        async def aclose(self):
            await asyncio.sleep(0)

    upstream = httpx.Response(200, stream=Stream())
    response = gateway._stream_response(upstream, "engine-rid", "disconnected", "http://router")

    async def send(message):
        if disconnect == "send_failure":
            raise OSError("client disconnected")
        await asyncio.sleep(0)
        if message["type"] == "http.response.body":
            sent.set()

    async def receive():
        if disconnect == "after_first":
            await sent.wait()
        return {"type": "http.disconnect"}

    try:
        if disconnect == "send_failure":
            from starlette.requests import ClientDisconnect

            with pytest.raises(ClientDisconnect):
                await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        else:
            await response({"type": "http", "asgi": {"spec_version": "2.0"}}, receive, send)
        await asyncio.wait_for(finished.wait(), timeout=1)
        assert owner.inflight == set()
        gateway._abort_upstream.assert_awaited_once_with("http://router", "engine-rid")
        assert upstream.is_closed
    finally:
        await gateway.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("rid", [None, "caller-rid"])
async def test_gateway_chat_abort_uses_forwarded_request_id(rid: str | None) -> None:
    owner = _owner(_snapshot)
    gateway = InferenceGateway(Role.ROLLOUT, manager_handle=owner)
    gateway._abort_upstream = AsyncMock()
    forwarded = []

    def upstream(request):
        forwarded.append(json.loads(request.content)["rid"])
        raise httpx.ReadTimeout("lost", request=request)

    await gateway._client.aclose()
    gateway._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        response = await gateway.proxy(
            _request({"model": "model-a", "messages": [], "rid": rid}), "v1/chat/completions"
        )
        assert response.status_code == 502
        assert forwarded[0]
        if rid:
            assert forwarded == [rid]
        gateway._abort_upstream.assert_awaited_once_with("http://router", forwarded[0])
        assert not owner.inflight
    finally:
        await gateway.close()
