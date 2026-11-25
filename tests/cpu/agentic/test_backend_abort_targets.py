# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from relax.agentic.pipeline import runtime


def adapter(monkeypatch, client: httpx.AsyncClient, *, lifecycle: bool = False):
    backend = object.__new__(runtime.SGLangBackendAdapter)
    backend._args = SimpleNamespace(
        sglang_router_ip="router",
        sglang_router_port=80,
        use_rollout_routing_replay=False,
        sglang_router_policy="cache_aware",
        slime_router_sticky=False,
    )
    attempt = runtime._BackendAttempt("relax-agentic-r3-mm-exact-wire-id", set())
    attempt.send_done.set()  # The generate response was lost before cleanup began.
    backend._requests = {"request": attempt}
    backend._session_endpoints = {}
    backend._session_lifecycle = lifecycle

    backend._managed_http = client

    async def get(url, **_kwargs):
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

    async def post(url, payload, **_kwargs):
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    monkeypatch.setattr(runtime, "get", get)
    monkeypatch.setattr(runtime, "post", post)
    return backend


@pytest.mark.asyncio
async def test_abort_includes_unhealthy_worker_and_preserves_failed_target_for_retry(monkeypatch):
    calls = []
    discovered = 0
    failed = False

    def handle(request):
        nonlocal discovered, failed
        if request.url.path == "/workers":
            discovered += 1
            # A failing worker disappears from discovery before the retry.
            workers = [{"url": "http://healthy", "is_healthy": True}]
            if discovered == 1:
                workers.append({"url": "http://unhealthy", "is_healthy": False})
            return httpx.Response(200, json={"workers": workers})
        calls.append((request.url.host, request.content))
        if request.url.host == "unhealthy" and not failed:
            failed = True
            return httpx.Response(503, json={"error": "temporarily unreachable"})
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        backend = adapter(monkeypatch, client)
        with pytest.raises(httpx.HTTPStatusError):
            await backend.abort_request("request")
        assert backend._requests["request"].wire_id == "relax-agentic-r3-mm-exact-wire-id"
        await backend.abort_request("request")
        assert not backend._requests
    assert [host for host, _ in calls] == ["healthy", "unhealthy", "unhealthy"]
    assert all(json.loads(body) == {"rid": "relax-agentic-r3-mm-exact-wire-id"} for _, body in calls)


@pytest.mark.asyncio
async def test_abort_empty_worker_discovery_keeps_request_owner(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"workers": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        backend = adapter(monkeypatch, client)
        with pytest.raises(RuntimeError, match="without any worker targets"):
            await backend.abort_request("request")
        assert backend._requests["request"].wire_id == "relax-agentic-r3-mm-exact-wire-id"
    assert calls == ["/workers"]


@pytest.mark.asyncio
@pytest.mark.parametrize("lifecycle", [False, True])
async def test_session_close_retains_managed_endpoint_until_abort_ack(monkeypatch, lifecycle):
    calls = []

    def handle(request):
        calls.append((request.url.host, request.url.path))
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        backend = adapter(monkeypatch, client, lifecycle=lifecycle)
        backend._requests["request"].targets.add("http://original-engine")
        backend._session_endpoints["session"] = {"http://original-engine"}
        assert await backend.close_session("session", timeout_s=1)
        assert backend._requests["request"].targets == {"http://original-engine"}
        await backend.abort_request("request")
        assert not backend._requests and not backend._session_endpoints
    assert ("original-engine", "/abort_request") in calls
    assert all(host == "original-engine" for host, _ in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("context_error", [False, True])
async def test_rejected_generate_400_releases_wire_owner(monkeypatch, context_error):
    def handle(request):
        detail = "maximum context length" if context_error else "invalid sampling parameter"
        return httpx.Response(400, json={"error": detail})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        backend = adapter(monkeypatch, client)
        error_type = runtime.BackendContextLengthExceededError if context_error else httpx.HTTPStatusError
        with pytest.raises(error_type):
            await backend.generate(
                input_ids=[1], sampling_params={"max_new_tokens": 1}, session_id="session", request_id="request"
            )
        assert not backend._requests


@pytest.mark.asyncio
async def test_terminal_abort_reaches_worker_while_generate_holds_only_connection(monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from relax.utils import http_utils

    started, aborted = asyncio.Event(), asyncio.Event()
    wire_ids = []

    async def generate(request):
        wire_ids.append((await request.json())["rid"])
        started.set()
        await aborted.wait()
        return web.json_response({"output_ids": [], "meta_info": {"finish_reason": {"type": "abort"}}})

    async def workers(request):
        return web.json_response({"workers": [{"url": f"http://{request.host}"}]})

    async def abort(request):
        wire_ids.append((await request.json())["rid"])
        aborted.set()
        return web.json_response({})

    app = web.Application()
    app.add_routes([web.post("/generate", generate), web.get("/workers", workers), web.post("/abort_request", abort)])
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setattr(http_utils, "_http_client", None)
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", False)
    monkeypatch.setattr(http_utils, "_client_concurrency", 0)
    monkeypatch.setattr(runtime, "load_agentic_compiler_resources", lambda _: Mock(processor=None))
    async with TestServer(app) as server:
        args = SimpleNamespace(
            sglang_router_ip=server.host,
            sglang_router_port=server.port,
            sglang_server_concurrency=1,
            rollout_num_gpus=1,
            rollout_num_gpus_per_engine=1,
            use_distributed_post=False,
            agentic_session_lifecycle=False,
            apply_chat_template_kwargs={},
            use_audio_in_video=False,
        )
        backend = runtime.SGLangBackendAdapter(args)
        task = asyncio.create_task(
            backend.generate(input_ids=[1], sampling_params={}, session_id=None, request_id="r")
        )
        try:
            await asyncio.wait_for(started.wait(), 3)
            await asyncio.wait_for(backend.abort_request("r"), 2)
            assert (await asyncio.wait_for(task, 2)).finish_type == "abort"
            assert len(wire_ids) >= 2 and set(wire_ids) == {"r"}
        finally:
            aborted.set()
            await asyncio.gather(task, return_exceptions=True)
            await backend.shutdown()
            await http_utils._http_client.aclose()
