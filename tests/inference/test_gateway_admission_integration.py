# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from relax.inference.admission import AdmissionGate, create_admission_app
from relax.inference.gateway import InferenceGatewayHandler
from relax.inference.registry import InferenceRegistry
from relax.inference.specs import ModelSnapshot, ReplicaSnapshot, RoutingSpec


class _BackendStream(httpx.AsyncByteStream):
    closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"text": "answer"}\n\n'
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        self.closed = True


@asynccontextmanager
async def _request_chain(role: str = "teacher") -> AsyncIterator[SimpleNamespace]:
    gate = AdmissionGate()
    gate.open()
    registry = InferenceRegistry(role)
    state = SimpleNamespace(
        gate=gate,
        registry_state="READY",
        backend=[],
        admission=[],
        rendered=[],
        hold=False,
        started=asyncio.Event(),
        stream=_BackendStream(),
    )

    async def backend(request: httpx.Request) -> httpx.Response:
        state.backend.append(request)
        state.started.set()
        if state.hold:
            await asyncio.Event().wait()
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=state.stream)
        return httpx.Response(200, json={"text": "  answer  ", "meta_info": {"id": "request-1"}})

    async def snapshot() -> dict[str, Any]:
        current = state.registry_state
        return registry.publish(
            [
                ModelSnapshot(
                    "model",
                    current,
                    "DIRECT",
                    engines=(ReplicaSnapshot("model/0", "http://admission.example", current, current == "READY"),),
                )
            ],
            RoutingSpec(default_model="model", route_key_map={"judge": "model"}),
        )

    async def render(model_id: str, messages: list[dict], sampling: dict | None) -> dict[str, Any]:
        state.rendered.append((model_id, messages, sampling))
        return {"input_ids": [7, 8], "sampling_params": sampling}

    async def record_admission(request: httpx.Request) -> None:
        state.admission.append(request)

    async with AsyncExitStack() as stack:
        upstream = await stack.enter_async_context(httpx.AsyncClient(transport=httpx.MockTransport(backend)))
        admission = create_admission_app("http://backend.example", gate, http_client=upstream)
        internal_http = await stack.enter_async_context(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=admission),
                event_hooks={"request": [record_admission]},
            )
        )
        handler = InferenceGatewayHandler(role, snapshot, http_client=internal_http, genrm_render=render, timeout=2)
        stack.push_async_callback(handler.aclose)
        gateway = FastAPI()
        gateway.add_api_route("/generate", handler.handle_generate, methods=["POST"])
        state.client = await stack.enter_async_context(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=gateway),
                base_url="http://gateway.example",
            )
        )
        yield state


@pytest.mark.parametrize("role", ["rollout", "genrm", "teacher"])
async def test_gateway_admission_raw_generation_preserves_payload_and_affinity(role: str) -> None:
    async with _request_chain(role) as chain:
        payload = {"input_ids": [1, 2], "image_data": ["image"], "return_logprob": True}
        response = await chain.client.post(
            "/generate", json={**payload, "model": "model"}, headers={"X-SMG-Routing-Key": "session-1"}
        )
        assert response.status_code == 200
        assert response.json() == {"text": "  answer  ", "meta_info": {"id": "request-1"}}
        assert len(chain.admission) == len(chain.backend) == 1
        assert json.loads(chain.backend[0].content) == payload
        assert chain.backend[0].headers["x-smg-routing-key"] == "session-1"
        assert chain.rendered == []
        assert chain.gate.status() == {"ready": True, "inflight": 0, "backend_drained": False}


async def test_gateway_admission_genrm_messages_render_before_backend() -> None:
    async with _request_chain("genrm") as chain:
        messages, sampling = [{"role": "user", "content": "Evaluate answer."}], {"temperature": 0}
        response = await chain.client.post(
            "/generate", json={"messages": messages, "sampling_params": sampling, "route_key": "judge"}
        )
        assert response.status_code == 200
        assert response.json() == {"response": "answer"}
        assert chain.rendered == [("model", messages, sampling)]
        assert len(chain.admission) == len(chain.backend) == 1
        assert json.loads(chain.backend[0].content) == {"input_ids": [7, 8], "sampling_params": sampling}


@pytest.mark.parametrize(
    "closed_at",
    [
        "registry",
        pytest.param(
            "admission",
            marks=pytest.mark.xfail(
                strict=True, raises=KeyError, reason="Gateway drops admission 503 Retry-After header"
            ),
        ),
    ],
)
async def test_gateway_admission_sleeping_refuses_without_backend_post(closed_at: str) -> None:
    async with _request_chain() as chain:
        if closed_at == "registry":
            chain.registry_state = "SLEEPING"
        else:
            chain.gate.close()
        response = await chain.client.post("/generate", json={"input_ids": [1]})
        assert response.status_code == 503
        assert len(chain.admission) == (closed_at == "admission")
        assert chain.backend == []
        assert response.headers["retry-after"] == "1"


async def test_gateway_admission_close_cancels_without_replaying_post() -> None:
    async with _request_chain() as chain:
        chain.hold = True
        pending = asyncio.create_task(chain.client.post("/generate", json={"input_ids": [1]}))
        try:
            await asyncio.wait_for(chain.started.wait(), timeout=1)
            chain.gate.close()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert len(chain.admission) == len(chain.backend) == 1
            response = await chain.client.post("/generate", json={"input_ids": [2]})
            assert response.status_code == 503
            assert len(chain.admission) == 2
            assert len(chain.backend) == 1
            assert chain.gate.status() == {"ready": False, "inflight": 0, "backend_drained": False}
            chain.gate.confirm_backend_drained()
            chain.gate.wait_drained(0.01)
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)


async def test_gateway_admission_stream_completion_closes_backend_without_claiming_drain() -> None:
    async with _request_chain() as chain:
        response = await chain.client.post("/generate", json={"input_ids": [1], "stream": True})
        assert response.status_code == 200
        assert response.content == b'data: {"text": "answer"}\n\ndata: [DONE]\n\n'
        assert chain.stream.closed
        assert len(chain.admission) == len(chain.backend) == 1
        assert chain.gate.status() == {"ready": True, "inflight": 0, "backend_drained": False}
