# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import gzip
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.requests import Request

from relax.components.inference_gateway import InferenceGateway
from relax.engine.inference.executor import AdmittedRequest
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
        self.traces: list[str] = []

        async def snapshot(role):
            return provider()

        async def admit_request(role, model_id, trace_id):
            permit = f"permit-{len(self.traces)}"
            self.traces.append(trace_id)
            self.inflight.add(permit)
            return permit

        async def complete_request(permit):
            self.inflight.discard(permit)

        self.snapshot = SimpleNamespace(remote=snapshot)
        self.admit_request = SimpleNamespace(remote=admit_request)
        self.complete_request = SimpleNamespace(remote=complete_request)


def _owner(provider):
    return _Owner(provider)


async def _use_upstream(gateway: InferenceGateway, upstream) -> None:
    await gateway._client.aclose()
    gateway._executor.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))


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

    await _use_upstream(gateway, upstream)
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

    await _use_upstream(gateway, upstream)
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

    await _use_upstream(gateway, upstream)
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

    await _use_upstream(gateway, upstream)
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

    await _use_upstream(gateway, upstream)
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

    await _use_upstream(gateway, upstream)
    try:
        response = await gateway.proxy(_request({"model": "model-a", "input_ids": [1]}), "generate")
        assert response.status_code == 502
        assert len(aborted) == 1
        assert owner.inflight == set()
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_gateway_http_routes_use_admission_and_preserve_payload() -> None:
    from fastapi import FastAPI, Request

    owner = _owner(_snapshot)
    gateway = InferenceGateway(Role.ROLLOUT, manager_handle=owner)
    forwarded = []

    def upstream(request):
        forwarded.append(json.loads(request.content))
        return httpx.Response(200, json={"text": "answer"})

    await _use_upstream(gateway, upstream)
    app = FastAPI()

    @app.post("/rollout/generate")
    async def generate(request: Request):
        return await gateway.proxy(request, "generate")

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
            response = await client.post(
                "/rollout/generate", json={"model": "model-a", "text": "question", "rid": "caller"}
            )
        assert response.json() == {"text": "answer"}
        assert forwarded[0]["text"] == "question"
        assert forwarded[0]["rid"].startswith("relax-inf-")
        assert not owner.inflight
    finally:
        await gateway.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", [False, True])
async def test_gateway_compressed_response_preserves_decoded_body_framing(backend: bool) -> None:
    from fastapi import FastAPI, Request

    owner = _owner(_snapshot)
    gateway = InferenceGateway(Role.ROLLOUT, manager_handle=owner, upstream_url="http://backend")
    content = b'{"text":"answer"}'
    compressed = gzip.compress(content)

    def upstream(request):
        return httpx.Response(
            200,
            content=compressed,
            headers={"content-encoding": "gzip", "content-length": str(len(compressed))},
        )

    await _use_upstream(gateway, upstream)
    app = FastAPI()

    @app.post("/generate")
    async def generate(request: Request):
        if backend:
            return await gateway.proxy_backend(request, "generate")
        return await gateway.proxy(request, "generate")

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
            response = await client.post("/generate", json={"model": "model-a", "text": "question"})
        assert response.content == content
        assert int(response.headers["content-length"]) == len(content)
        assert "content-encoding" not in response.headers
        assert not owner.inflight
    finally:
        await gateway.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [200, 503])
async def test_gateway_stream_eof_aborts_only_unknown_outcomes(status_code: int) -> None:
    from fastapi import FastAPI, Request

    owner = _owner(_snapshot)
    gateway = InferenceGateway(Role.ROLLOUT, manager_handle=owner)
    forwarded = []
    aborted = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: result\n\n"

    def upstream(request):
        if request.url.path == "/generate":
            forwarded.append(json.loads(request.content)["rid"])
            return httpx.Response(status_code, stream=Stream(), headers={"content-type": "text/event-stream"})
        if request.url.path == "/workers":
            return httpx.Response(200, json={"workers": [{"url": "http://engine"}]})
        aborted.append(json.loads(request.content)["rid"])
        return httpx.Response(200, json={})

    await _use_upstream(gateway, upstream)
    app = FastAPI()

    @app.post("/generate")
    async def generate(request: Request):
        return await gateway.proxy(request, "generate")

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
            response = await client.post("/generate", json={"model": "model-a", "text": "question"})
        await gateway.close()
        assert response.status_code == status_code
        assert response.content == b"data: result\n\n"
        assert aborted == (forwarded if status_code >= 500 else [])
        assert not owner.inflight
    finally:
        await gateway.close()


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
    gateway._executor.abort = AsyncMock()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: first\n\n"
            await asyncio.Event().wait()

        async def aclose(self):
            await asyncio.sleep(0)

    upstream = httpx.Response(200, stream=Stream())
    admitted = AdmittedRequest("model-a", "http://router", True, "disconnected", "engine-rid", dispatched=True)
    response = gateway._stream_response(upstream, admitted)

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
        gateway._executor.abort.assert_awaited_once_with(admitted)
        assert upstream.is_closed
    finally:
        await gateway.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("rid", [None, "caller-rid"])
async def test_gateway_chat_abort_uses_the_execution_rid(rid: str | None) -> None:
    owner = _owner(_snapshot)
    gateway = InferenceGateway(Role.ROLLOUT, manager_handle=owner)
    gateway._executor.abort = AsyncMock()
    forwarded = []

    def upstream(request):
        forwarded.append(json.loads(request.content)["rid"])
        raise httpx.ReadTimeout("lost", request=request)

    await _use_upstream(gateway, upstream)
    try:
        response = await gateway.proxy(
            _request({"model": "model-a", "messages": [], "rid": rid}), "v1/chat/completions"
        )
        assert response.status_code == 502
        # The caller's rid is only traced; the engine runs this execution's own.
        assert forwarded[0].startswith("relax-inf-") and forwarded[0] != rid
        ((aborted,), _) = gateway._executor.abort.await_args
        assert (aborted.target, aborted.rid) == ("http://router", forwarded[0])
        assert not owner.inflight
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_internal_generation_uses_executor_without_serve_callback():
    """Worker requests need only Manager RPC + upstream HTTP, even if Serve is
    busy."""
    from relax.engine.inference.executor import InferenceRequestExecutor

    owner = _owner(lambda: replace(_snapshot(), routing=RoutingSpec(default_model="model-a")))
    sent = []

    def upstream(request):
        assert owner.inflight
        assert str(request.url) == "http://router/generate"
        sent.append(json.loads(request.content))
        assert request.headers["x-test"] == "worker"
        return httpx.Response(200, json={"text": "ok"})

    payload = {"text": "hello", "rid": "trace"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        executor = InferenceRequestExecutor(Role.ROLLOUT, owner, client)
        assert await executor.generate_json(payload, {"x-test": "worker"}) == {"text": "ok"}
    assert payload["rid"] == "trace"
    assert sent[0]["rid"].startswith("relax-inf-")
    assert not owner.inflight


@pytest.mark.asyncio
async def test_internal_generation_cancel_aborts_and_returns_permit():
    from relax.engine.inference.executor import InferenceRequestExecutor

    owner = _owner(lambda: replace(_snapshot(), routing=RoutingSpec(default_model="model-a")))
    entered = asyncio.Event()

    async def upstream(request):
        entered.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        executor = InferenceRequestExecutor(Role.ROLLOUT, owner, client)
        executor.abort = AsyncMock()
        task = asyncio.create_task(executor.generate_json({"text": "hello"}))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        executor.abort.assert_awaited_once()
    assert not owner.inflight


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ReadError, httpx.ReadTimeout])
async def test_gateway_response_body_failure_returns_502(error):
    class BrokenBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise error("body disconnected")
            yield b""

    owner = _owner(_snapshot)
    gateway = InferenceGateway(Role.ROLLOUT, manager_handle=owner)

    def upstream(request):
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, stream=BrokenBody())
        if request.url.path == "/workers":
            return httpx.Response(200, json={"workers": []})
        return httpx.Response(200, json={})

    await _use_upstream(gateway, upstream)
    try:
        response = await gateway.proxy(_request({"model": "model-a", "messages": []}), "v1/chat/completions")
        assert response.status_code == 502
        assert owner.inflight == set()
    finally:
        await gateway.close()
