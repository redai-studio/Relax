# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import StreamingResponse

from relax.inference.gateway import InferenceGatewayHandler, forward_headers
from relax.inference.registry import InferenceRegistry
from relax.inference.routing import RouteResolver
from relax.inference.specs import ModelSnapshot, ReplicaSnapshot, RoutingSpec


def _snapshot(role: str = "teacher", state: str = "READY") -> dict:
    return InferenceRegistry(role).publish(
        [
            ModelSnapshot(
                "math",
                state,
                "DIRECT",
                engines=(
                    ReplicaSnapshot("math-0", "http://replica-0.example:15000", state, state == "READY"),
                    ReplicaSnapshot("math-1", "http://replica-1.example:15000", state, state == "READY"),
                ),
                served_model_name="organization/teacher-checkpoint",
            )
        ],
        RoutingSpec(default_model="math", route_key_map={"math-data": "math"}, aliases={"teacher": "math"}),
    )


@pytest.fixture
async def gateway_factory():
    handlers, clients = [], []

    def make(
        *,
        role="teacher",
        state="READY",
        response=None,
        snapshot=None,
        genrm_render=None,
        timeout=1.0,
    ):
        source = SimpleNamespace(snapshot=snapshot or _snapshot(role, state), reads=0, requests=[])

        async def provider():
            source.reads += 1
            return source.snapshot

        def dispatch(request):
            source.requests.append(request)
            return response(request) if callable(response) else httpx.Response(200, json=response or {"ok": True})

        http = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
        handler = InferenceGatewayHandler(role, provider, http, genrm_render, timeout=timeout)
        handlers.append(handler)
        clients.append(http)
        return handler, source

    yield make
    for handler in handlers:
        await handler.aclose()
    for client in clients:
        await client.aclose()


@pytest.mark.parametrize("role", ["rollout", "genrm", "teacher"])
async def test_gateway_and_direct_resolver_choose_same_affinity_target(gateway_factory, role):
    handler, source = gateway_factory(role=role)
    target = RouteResolver().resolve(source.snapshot, route_key="math-data", affinity_key="group-7")

    await handler.generate({"input_ids": [1, 2], "route_key": "math-data"}, {"x-smg-routing-key": "group-7"})

    assert str(source.requests[0].url) == f"{target.base_url}/generate"
    assert source.requests[0].headers["x-smg-routing-key"] == "group-7"
    assert json.loads(source.requests[0].content) == {"input_ids": [1, 2]}


async def test_gateway_teacher_preserves_raw_logprob_and_multimodal_fields(gateway_factory):
    expected_response = {"meta_info": {"input_token_logprobs": [[-0.5, 2, None]]}}
    handler, source = gateway_factory(response=expected_response)
    payload = {
        "input_ids": [1, 2, 3],
        "sampling_params": {"max_new_tokens": 0},
        "return_logprob": True,
        "logprob_start_len": 1,
        "token_ids_logprob": [4, 5],
        "image_data": [{"format": "opd_preexpanded_raw", "images_b64": ["image"], "image_grid_thw": [[1, 2, 3]]}],
        "rid": "teacher-request",
    }

    assert await handler.generate(payload) == expected_response
    assert json.loads(source.requests[0].content) == payload


async def test_gateway_genrm_renders_selected_model_and_preserves_legacy_text_shape(gateway_factory):
    rendered = []

    async def render(model_id, messages, sampling_params):
        rendered.append((model_id, messages, sampling_params))
        return {"input_ids": [9, 8], "sampling_params": {"temperature": 0.0, "max_new_tokens": 8}}

    handler, source = gateway_factory(
        role="genrm", genrm_render=render, response={"text": "  score: 1  ", "meta_info": {}}
    )
    messages = [{"role": "user", "content": "Judge answer."}]

    result = await handler.generate(
        {"messages": messages, "sampling_params": {"temperature": 0}, "route_key": "math-data"}
    )

    assert result == {"response": "score: 1"}
    assert rendered == [("math", messages, {"temperature": 0})]
    assert json.loads(source.requests[0].content) == {
        "input_ids": [9, 8],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 8},
    }


async def test_gateway_raw_genrm_request_bypasses_messages_adapter(gateway_factory):
    async def render(*args):
        raise AssertionError("Raw payload must not be retokenized")

    handler, source = gateway_factory(role="genrm", genrm_render=render, response={"text": "raw"})

    assert await handler.generate({"input_ids": [1]}) == {"text": "raw"}
    assert len(source.requests) == 1


async def test_gateway_genrm_template_rendering_obeys_request_deadline(gateway_factory):
    async def render(*args):
        await asyncio.Event().wait()

    handler, source = gateway_factory(role="genrm", genrm_render=render, timeout=0.02)

    with pytest.raises(HTTPException) as error:
        await handler.generate({"messages": []})

    assert error.value.status_code == 504
    assert source.requests == []


async def test_gateway_genrm_rechecks_state_after_template_rendering(gateway_factory):
    async def render(*args):
        source.snapshot = _snapshot("genrm", "SLEEPING")
        return {"input_ids": [1]}

    handler, source = gateway_factory(role="genrm", genrm_render=render)

    with pytest.raises(HTTPException) as error:
        await handler.generate({"messages": []})

    assert error.value.status_code == 503
    assert source.requests == []


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": [], "input_ids": [1]},
        {"messages": [], "text": "raw"},
        {"messages": "wrong"},
        {"messages": [], "sampling_params": []},
        {"messages": [], "stream": True},
        {"input_ids": [1], "stream": "true"},
        {},
        [],
    ],
)
async def test_gateway_invalid_generate_payload_refuses_before_backend(gateway_factory, payload):
    async def render(*args):
        raise AssertionError("Invalid input must not be rendered")

    handler, source = gateway_factory(role="genrm", genrm_render=render)

    with pytest.raises(HTTPException) as error:
        await handler.generate(payload)

    assert error.value.status_code == 400
    assert source.requests == []


@pytest.mark.parametrize("role", ["rollout", "genrm", "teacher"])
async def test_gateway_sleeping_models_have_discovery_and_liveness_without_gpu_requests(gateway_factory, role):
    handler, source = gateway_factory(role=role, state="SLEEPING")

    discovery = await handler.discovery()
    models = await handler.models()
    health = await handler.health()
    with pytest.raises(HTTPException) as error:
        await handler.generate({"input_ids": [1]})

    assert discovery["models"]["math"]["state"] == "SLEEPING"
    assert [model["id"] for model in models["data"]] == ["math"]
    assert health["gateway_alive"] is True
    assert health["registry_available"] is True
    assert health["ready"] is False
    assert error.value.status_code == 503
    assert error.value.headers == {"Retry-After": "1"}
    assert source.requests == []


async def test_gateway_lost_registry_reports_alive_but_not_ready():
    async def provider():
        raise ConnectionError("Manager unavailable")

    handler = InferenceGatewayHandler("teacher", provider)
    try:
        health = await handler.health()
        assert health["gateway_alive"] is True
        assert health["registry_available"] is False
        assert health["ready"] is False
        with pytest.raises(HTTPException) as error:
            await handler.discovery()
        assert error.value.status_code == 503
    finally:
        await handler.aclose()


async def test_gateway_wrong_role_snapshot_is_not_routable(gateway_factory):
    handler, source = gateway_factory(role="teacher", snapshot=_snapshot("rollout"))

    with pytest.raises(HTTPException) as error:
        await handler.generate({"input_ids": [1]})

    assert error.value.status_code == 503
    assert source.requests == []


async def test_gateway_chat_normalizes_backend_model_and_transport_headers(gateway_factory):
    handler, source = gateway_factory(response={"model": "organization/teacher-checkpoint", "choices": []})
    payload = {"model": "teacher", "messages": []}

    response = await handler.chat(
        payload,
        {
            "Host": "gateway.example",
            "Content-Length": "1",
            "Connection": "custom-hop",
            "custom-hop": "remove",
            "x-request-id": "keep",
        },
    )

    assert isinstance(response, httpx.Response)
    request = source.requests[0]
    assert json.loads(request.content)["model"] == "organization/teacher-checkpoint"
    assert payload["model"] == "teacher"
    assert request.headers["host"] == "replica-0.example:15000"
    assert request.headers["content-length"] == str(len(request.content))
    assert "custom-hop" not in request.headers
    assert request.headers["x-request-id"] == "keep"


class _Stream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed += 1


async def test_gateway_stream_preserves_sse_bytes_and_closes_upstream_once(gateway_factory):
    chunks = [b'event: token\r\ndata: {"text":"a"}\r\n\r\n', b"data: [DONE]\n\n"]
    stream = _Stream(chunks)
    handler, source = gateway_factory(
        response=lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )

    response = await handler.chat({"messages": [], "stream": True})

    assert isinstance(response, StreamingResponse)
    assert len(source.requests) == 1
    assert [chunk async for chunk in response.body_iterator] == chunks
    await response.aclose()
    assert stream.closed == 1


async def test_gateway_raw_stream_strips_routing_fields_without_retokenizing(gateway_factory):
    stream = _Stream([b"data: raw\n\n"])
    handler, source = gateway_factory(response=lambda request: httpx.Response(200, stream=stream))

    response = await handler.generate({"input_ids": [1], "model": "teacher", "route_key": "math-data", "stream": True})
    assert [chunk async for chunk in response.body_iterator] == [b"data: raw\n\n"]
    assert json.loads(source.requests[0].content) == {"input_ids": [1], "stream": True}
    assert stream.closed == 1


@pytest.mark.parametrize("status", [400, 503])
async def test_gateway_stream_upstream_error_is_http_error_before_response_headers(gateway_factory, status):
    stream = _Stream([b"upstream unavailable"])
    handler, source = gateway_factory(response=lambda request: httpx.Response(status, stream=stream))

    with pytest.raises(HTTPException) as error:
        await handler.chat({"messages": [], "stream": True})

    assert error.value.status_code == status
    assert len(source.requests) == 1
    assert stream.closed == 1


async def test_gateway_asgi_disconnect_before_first_chunk_closes_upstream(gateway_factory):
    stream = _Stream([b"data: token\n\n"])
    handler, _source = gateway_factory(response=lambda request: httpx.Response(200, stream=stream))
    response = await handler.chat({"messages": [], "stream": True})

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        raise OSError("client disconnected before headers")

    with pytest.raises(Exception):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

    assert stream.closed == 1


async def test_gateway_stream_failure_after_first_chunk_never_replays(gateway_factory):
    class FailingStream(_Stream):
        async def __aiter__(self):
            yield b"data: first\n\n"
            raise httpx.ReadError("connection lost")

    stream = FailingStream([])
    handler, source = gateway_factory(response=lambda request: httpx.Response(200, stream=stream))
    response = await handler.chat({"messages": [], "stream": True})
    chunks = []

    with pytest.raises(httpx.ReadError):
        async for chunk in response.body_iterator:
            chunks.append(chunk)

    assert chunks == [b"data: first\n\n"]
    assert len(source.requests) == 1
    assert stream.closed == 1


async def test_gateway_stream_deadline_closes_upstream(gateway_factory):
    class StalledStream(_Stream):
        async def __aiter__(self):
            await asyncio.Event().wait()
            yield b"unreachable"

    stream = StalledStream([])
    handler, source = gateway_factory(response=lambda request: httpx.Response(200, stream=stream), timeout=0.02)
    response = await handler.chat({"messages": [], "stream": True})

    with pytest.raises(TimeoutError):
        _ = [chunk async for chunk in response.body_iterator]

    assert stream.closed == 1
    assert len(source.requests) == 1


async def test_gateway_http_handlers_translate_bad_json_and_sleep_before_stream(gateway_factory):
    handler, source = gateway_factory(state="SLEEPING")
    app = FastAPI()

    @app.post("/generate")
    async def generate(request: Request):
        return await handler.handle_generate(request)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await handler.handle_chat(request)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://gateway.example") as client:
        invalid = await client.post("/generate", content=b"not-json")
        sleeping = await client.post("/v1/chat/completions", json={"messages": [], "stream": True})

    assert invalid.status_code == 400
    assert sleeping.status_code == 503
    assert sleeping.headers["content-type"] == "application/json"
    assert source.requests == []


def test_gateway_headers_remove_all_connection_nominated_fields():
    assert forward_headers(
        {"Connection": "x-hop, another-hop", "X-Hop": "a", "another-hop": "b", "x-request-id": "r"}
    ) == {"x-request-id": "r"}


def test_gateway_deployment_is_explicitly_cpu_only_without_gpu_placement():
    from relax.components.inference_gateway import InferenceGateway

    options = InferenceGateway.ray_actor_options

    assert options["num_gpus"] == 0
    assert options["num_cpus"] == 1
    assert "placement_group" not in options
    assert "scheduling_strategy" not in options
