# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Control failures preserve completed results and the public ASGI contract."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request

from relax.engine.inference.client import InferenceRequestClient
from relax.engine.inference.gateway import InferenceGateway
from relax.engine.inference.routing import ModelUnavailableError
from tests.cpu.engine.inference.helpers import AdmissionManager
from tests.cpu.engine.inference.test_request_dispatch import snapshot


class Stream(httpx.AsyncByteStream):
    def __init__(self, body_error):
        self.closed = False
        self.body_error = body_error

    async def __aiter__(self):
        yield b'data: {"choices":[]}\n\n'
        if self.body_error:
            raise ModelUnavailableError("control unavailable after headers")
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


@asynccontextmanager
async def asgi_case(entry, *, upstream_status=200, confirm_error=False, body_error=False):
    from relax.components.rollout import Rollout as RolloutDeployment

    manager = AdmissionManager(snapshot())
    release_receipt = asyncio.Event()
    state = {"stall_release": False}
    errors, messages, streams = [], [], []

    async def acquire(request_id, **selection):
        return manager.acquire_inference(request_id, **selection)

    async def release(permit):
        if state["stall_release"]:
            await release_receipt.wait()
        manager.release_inference(permit)

    async def confirm(permit):
        if confirm_error:
            raise ConnectionError("dispatch response lost")
        manager.confirm_inference_dispatch(permit)

    remote = SimpleNamespace(
        acquire_inference=SimpleNamespace(remote=acquire),
        release_inference=SimpleNamespace(remote=release),
        confirm_inference_dispatch=SimpleNamespace(remote=confirm),
    )

    async def resources():
        return {"snapshot": manager.fetch().to_dict(), "managers": {"default": remote}}

    client = InferenceRequestClient("rollout", resources)

    def upstream(request):
        if upstream_status != 200:
            return httpx.Response(upstream_status, text="slow down", headers={"Retry-After": "3"})
        if entry.endswith("stream"):
            stream = Stream(body_error)
            streams.append(stream)
            return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})
        body = (
            {"text": "success"}
            if entry.startswith("native")
            else {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "default",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as proxy:
        gateway = InferenceGateway("rollout", manager.fetch, request_client=client, proxy_client=proxy)
        rollout = SimpleNamespace(_gateway=gateway)
        handler = (
            RolloutDeployment.func_or_class.generate
            if entry.startswith("native")
            else RolloutDeployment.func_or_class.chat_completions
        )
        app = FastAPI()

        @app.post("/probe")
        async def endpoint(request: Request):
            return await handler(rollout, request)

        async def audited_app(scope, receive, send):
            async def observed(message):
                messages.append(message)
                await send(message)

            try:
                await app(scope, receive, observed)
            except BaseException as error:
                errors.append(error)
                raise

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=audited_app, raise_app_exceptions=False), base_url="http://test"
        ) as downstream:
            payload = {"stream": entry.endswith("stream")}
            payload.update(
                {"input_ids": [1]} if entry.startswith("native") else {"messages": [{"role": "user", "content": "hi"}]}
            )
            try:
                yield SimpleNamespace(
                    request=lambda: downstream.post("/probe", json=payload),
                    state=state,
                    manager=manager,
                    client=client,
                    errors=errors,
                    messages=messages,
                    streams=streams,
                )
            finally:
                release_receipt.set()
                await asyncio.gather(*client._cleanup_tasks, return_exceptions=True)
                await asyncio.sleep(0)
                assert not manager.gate._permits and not client._cleanup_tasks
                assert manager.released == manager.acquired


@pytest.mark.parametrize(
    "entry,status",
    [(entry, status) for entry in ("native", "chat") for status in (200, 429)]
    + [("native_stream", 200), ("chat_stream", 200)],
)
async def test_release_timeout_preserves_upstream_result_and_late_return(monkeypatch, entry, status):
    monkeypatch.setenv("RELAX_INFERENCE_CONTROL_TIMEOUT_S", "0.02")
    async with asgi_case(entry, upstream_status=status) as case:
        expected = await case.request()
        assert expected.status_code == status
        case.state["stall_release"] = True
        actual = await asyncio.wait_for(case.request(), 0.5)
        assert actual.status_code == expected.status_code
        assert actual.content == expected.content
        assert actual.headers.get("retry-after") == expected.headers.get("retry-after")
        assert not case.errors
        assert len(case.manager.gate._permits) == 1
        assert case.client._cleanup_tasks
        assert all(stream.closed for stream in case.streams)


@pytest.mark.parametrize("entry", ["native_stream", "chat_stream"])
async def test_dispatch_rpc_error_uses_http_or_sse_unavailable_contract(monkeypatch, entry):
    monkeypatch.setenv("RELAX_INFERENCE_CONTROL_TIMEOUT_S", "0.02")
    async with asgi_case(entry, confirm_error=True) as case:
        response = await case.request()
        if entry == "native_stream":
            assert response.status_code == 503
            assert response.headers["retry-after"] == "5"
            assert "dispatch" in response.json()["detail"].lower()
        else:
            assert response.status_code == 200
            frames = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
            assert len(frames) == 2 and frames[-1] == "[DONE]"
            assert json.loads(frames[0])["error"]["code"] == 503
        assert not case.errors
        assert case.messages[-1]["type"] == "http.response.body" and not case.messages[-1].get("more_body", False)
        assert case.streams and all(stream.closed for stream in case.streams)


@pytest.mark.parametrize("entry", ["native_stream", "chat_stream"])
async def test_control_error_after_stream_start_finishes_with_error_and_done(entry):
    async with asgi_case(entry, body_error=True) as case:
        response = await case.request()
        assert response.status_code == 200
        frames = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        assert len(frames) == 3 and frames[-1] == "[DONE]"
        assert json.loads(frames[1])["error"]["code"] == 503
        assert not case.errors
        assert all(stream.closed for stream in case.streams)
