# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Public ASGI cancellation hands off ordered HTTP close and permit return."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request

from relax.engine.inference.admission import AdmissionDrainError
from relax.engine.inference.client import InferenceRequestClient
from relax.engine.inference.gateway import InferenceGateway
from relax.engine.inference.holder import current_holder
from tests.cpu.engine.inference.helpers import AdmissionManager
from tests.cpu.engine.inference.test_request_dispatch import snapshot


@pytest.fixture(autouse=True)
def control_budget(monkeypatch):
    current_holder()  # Warm process/Ray identity lookup before the short outer budget.
    monkeypatch.setenv("RELAX_INFERENCE_CONTROL_TIMEOUT_S", "1")


async def settle(client):
    while client._cleanup_tasks:
        await asyncio.wait_for(asyncio.gather(*client._cleanup_tasks), 2)
        await asyncio.sleep(0)  # Deliver completion callbacks, not a timing barrier.


class ProbeStream(httpx.AsyncByteStream):
    def __init__(self, case):
        self.case = case
        self.closed = False
        self.close_calls = 0

    async def __aiter__(self):
        if self.case.status != 200:
            yield b"slow down"
            return
        yield b'data: {"choices":[]}\n\n'
        self.case.body_entered.set()
        if self.case.phase == "body":
            await self.case.body_finish.wait()
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.close_calls += 1
        self.case.close_entered.set()
        await self.case.close_finish.wait()
        self.closed = True
        self.case.events.append("upstream_closed")


@asynccontextmanager
async def asgi_case(entry, *, status=200, phase="transport", block_close=False):
    from relax.components.rollout import Rollout as RolloutDeployment

    case = SimpleNamespace(
        entry=entry,
        status=status,
        phase=phase,
        events=[],
        messages=[],
        streams=[],
        returns=[],
        transport_entered=asyncio.Event(),
        transport_finish=asyncio.Event(),
        body_entered=asyncio.Event(),
        body_finish=asyncio.Event(),
        close_entered=asyncio.Event(),
        close_finish=asyncio.Event(),
        release_entered=asyncio.Event(),
        release_receipt=asyncio.Event(),
    )
    if not block_close:
        case.close_finish.set()
    case.manager = AdmissionManager(snapshot())

    async def acquire(request_id, **selection):
        return case.manager.acquire_inference(request_id, **selection)

    async def release(permit):
        case.events.append("release_entered")
        case.returns.append(permit)
        case.release_entered.set()
        await case.release_receipt.wait()
        case.manager.release_inference(permit)
        case.events.append("release_receipt")

    async def confirm(permit):
        case.manager.confirm_inference_dispatch(permit)

    remote = SimpleNamespace(
        acquire_inference=SimpleNamespace(remote=acquire),
        release_inference=SimpleNamespace(remote=release),
        confirm_inference_dispatch=SimpleNamespace(remote=confirm),
    )

    async def resources():
        return {"snapshot": case.manager.fetch().to_dict(), "managers": {"default": remote}}

    case.client = InferenceRequestClient("rollout", resources)

    async def upstream(request):
        case.transport_entered.set()
        if phase == "transport":
            try:
                await case.transport_finish.wait()
            except asyncio.CancelledError:
                # Model a transport unwinding its socket before cancellation
                # reaches client.send/post, possibly blocked in close itself.
                case.close_entered.set()
                await case.close_finish.wait()
                case.events.append("upstream_closed")
                raise
        if entry.endswith("stream") or status != 200:
            stream = ProbeStream(case)
            case.streams.append(stream)
            return httpx.Response(status, stream=stream, headers={"Retry-After": "3"})
        body = (
            {"text": "ok"}
            if entry == "native"
            else {
                "id": "probe",
                "object": "chat.completion",
                "created": 1,
                "model": "default",
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as proxy:
        gateway = InferenceGateway("rollout", case.manager.fetch, request_client=case.client, proxy_client=proxy)
        case.gateway = gateway
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
                case.messages.append(message)
                await send(message)

            await app(scope, receive, observed)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=audited_app, raise_app_exceptions=False), base_url="http://test"
        ) as downstream:
            payload = {"stream": entry.endswith("stream")}
            payload.update(
                {"input_ids": [1]} if entry.startswith("native") else {"messages": [{"role": "user", "content": "hi"}]}
            )
            case.request = lambda: downstream.post("/probe", json=payload)
            try:
                yield case
            finally:
                case.transport_finish.set()
                case.body_finish.set()
                case.close_finish.set()
                case.release_receipt.set()
                await settle(case.client)
                assert not case.manager.gate._permits


def assert_unknown_permit(case):
    assert len(case.manager.gate._permits) == 1 and case.client._cleanup_tasks
    assert not case.manager.released
    case.manager.gate.close()
    with pytest.raises(AdmissionDrainError, match="old request permits"):
        case.manager.gate.open()


async def finish_return(case):
    await asyncio.wait_for(case.release_entered.wait(), 1)
    assert case.events.index("upstream_closed") < case.events.index("release_entered")
    assert_unknown_permit(case)
    case.release_receipt.set()
    await settle(case.client)
    assert case.returns == case.manager.acquired == case.manager.released
    assert len(case.returns) == 1
    assert not case.manager.gate._permits and not case.client._cleanup_tasks


@pytest.mark.parametrize("cause", ["deadline", "cancel"])
@pytest.mark.parametrize(
    "entry,phase",
    [
        ("native", "transport"),
        ("chat", "transport"),
        ("native_stream", "transport"),
        ("chat_stream", "transport"),
        ("native_stream", "body"),
        ("chat_stream", "body"),
    ],
)
async def test_gateway_cancellation_does_not_restart_release_budget(entry, phase, cause):
    async with asgi_case(entry, phase=phase) as case:
        call = case.request()
        task = asyncio.create_task(asyncio.wait_for(call, 0.03) if cause == "deadline" else call)
        try:
            entered = case.transport_entered if phase == "transport" else case.body_entered
            await asyncio.wait_for(entered.wait(), 1)
            if cause == "cancel":
                task.cancel()
            done, _ = await asyncio.wait({task}, timeout=0.15)
            assert done, "Gateway cancellation must not start another one-second release wait"
            with pytest.raises(asyncio.TimeoutError if cause == "deadline" else asyncio.CancelledError):
                await task
            if phase == "body":
                assert any(m["type"] == "http.response.body" and m.get("body") for m in case.messages)
            await finish_return(case)
        finally:
            case.close_finish.set()
            case.release_receipt.set()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("entry", ["native", "chat", "native_stream", "chat_stream"])
async def test_gateway_429_repeated_cancellation_preserves_one_late_return(entry):
    async with asgi_case(entry, status=429, phase=None) as case:
        task = asyncio.create_task(case.request())
        try:
            await asyncio.wait_for(case.release_entered.wait(), 1)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
            done, _ = await asyncio.wait({task}, timeout=0.15)
            assert done, "repeated cancellation must not settle the release RPC in front"
            with pytest.raises(asyncio.CancelledError):
                await task
            if entry == "chat_stream":
                errors = [m["body"] for m in case.messages if m["type"] == "http.response.body" and m.get("body")]
                assert any(
                    json.loads(frame.decode().split("data: ", 1)[1])["error"]["code"] == 429
                    for frame in errors
                    if b'"error"' in frame
                )
            await finish_return(case)
        finally:
            case.release_receipt.set()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize(
    "entry,phase",
    [
        ("native", "transport"),
        ("chat", "transport"),
        ("native_stream", "transport"),
        ("chat_stream", "transport"),
        ("native_stream", "body"),
        ("chat_stream", "body"),
    ],
)
async def test_gateway_cancellation_owns_blocked_upstream_close_before_permit_return(entry, phase):
    async with asgi_case(entry, phase=phase, block_close=True) as case:
        task = asyncio.create_task(case.request())
        try:
            entered = case.body_entered if phase == "body" else case.transport_entered
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            await asyncio.wait_for(case.close_entered.wait(), 1)
            done, _ = await asyncio.wait({task}, timeout=0.15)
            assert done, "blocked upstream close must be owned off the cancelled foreground"
            with pytest.raises(asyncio.CancelledError):
                await task
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
            assert_unknown_permit(case)
            assert not case.release_entered.is_set()
            case.close_finish.set()
            await finish_return(case)
            assert all(stream.closed for stream in case.streams)
        finally:
            case.close_finish.set()
            case.release_receipt.set()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("entry", ["native", "chat", "native_stream", "chat_stream"])
async def test_gateway_cancel_during_automatic_close_waits_for_original_close_receipt(entry):
    async with asgi_case(entry, status=429, phase=None, block_close=True) as case:
        task = asyncio.create_task(case.request())
        try:
            # HTTPX has set Response.is_closed, but the actual stream close is
            # still pending. A second Response.aclose() is not completion proof.
            await asyncio.wait_for(case.close_entered.wait(), 1)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
            done, _ = await asyncio.wait({task}, timeout=0.15)
            assert done
            with pytest.raises(asyncio.CancelledError):
                await task
            assert_unknown_permit(case)
            assert not case.release_entered.is_set()
            case.close_finish.set()
            await finish_return(case)
            assert all(stream.closed and stream.close_calls == 1 for stream in case.streams)
        finally:
            case.close_finish.set()
            case.release_receipt.set()
            await asyncio.gather(task, return_exceptions=True)


async def test_gateway_cancel_at_stream_response_handoff_closes_unreceived_response(monkeypatch):
    async with asgi_case("native_stream", phase=None) as case:
        original = case.gateway._stream_generate

        async def stream_ready(*args, **kwargs):
            response = await original(*args, **kwargs)
            # Queue cancellation before the completed dispatch task can wake
            # the frontend that would take ownership of this response.
            asyncio.get_running_loop().call_soon(task.cancel)
            return response

        monkeypatch.setattr(case.gateway, "_stream_generate", stream_ready)
        task = asyncio.create_task(case.request())
        try:
            done, _ = await asyncio.wait({task}, timeout=0.15)
            assert done
            with pytest.raises(asyncio.CancelledError):
                await task
            await finish_return(case)
            assert len(case.streams) == 1 and case.streams[0].closed
            assert not any(m["type"] == "http.response.start" for m in case.messages)
        finally:
            case.close_finish.set()
            case.release_receipt.set()
            await asyncio.gather(task, return_exceptions=True)
