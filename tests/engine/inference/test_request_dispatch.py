# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Delayed HTTP dispatch cannot cross an untracked inference generation."""

import asyncio
import json

import httpx
import pytest

from relax.engine.inference.admission import AdmissionDrainError
from relax.engine.inference.discovery import EngineState, RoleSnapshot, build_model_snapshot
from relax.engine.inference.gateway import InferenceGateway
from relax.utils import http_utils
from tests.engine.inference.helpers import AdmissionManager


def snapshot(url="http://engine:1"):
    return RoleSnapshot("rollout", 1, models=(build_model_snapshot("default", [(0, url, EngineState.READY)]),))


@pytest.mark.parametrize("resume", [False, True])
async def test_unconsumed_chat_stream_admits_only_when_dispatched(resume):
    manager = AdmissionManager(snapshot())
    sent = []

    async def dispatch(request):
        permits = list(manager.gate._permits.values())
        assert len(permits) == 1
        assert permits[0].generation == manager.gate._generation
        sent.append(request)
        return httpx.Response(200, text='data: {"choices":[]}\n\ndata: [DONE]\n\n')

    async with httpx.AsyncClient(transport=httpx.MockTransport(dispatch)) as http:
        gateway = InferenceGateway("rollout", manager.fetch, request_client=manager.client(), proxy_client=http)
        response = await gateway.chat_completions(json.dumps({"messages": [], "stream": True}).encode(), {})
        assert manager.acquired == [], "an unconsumed stream must not hold an unsent permit"
        old_generation = manager.gate._generation
        manager.gate.drain(lambda: None, timeout=0)
        manager.gate.confirm_drained()  # Stand-in for the backend idle/release receipt.
        if resume:
            manager.gate.open()
            assert manager.gate._generation != old_generation
        chunks = [chunk async for chunk in response.body_iterator]
        assert chunks
        assert len(sent) == int(resume)
        if not resume:
            assert '"code": 503' in "".join(chunks)
        assert manager.acquired == manager.released


@pytest.mark.parametrize("entry", ["proxy", "lookaside"])
async def test_connection_pool_queue_prevents_handoff_until_send_is_cancelled(monkeypatch, entry):
    occupied, finish = asyncio.Event(), asyncio.Event()
    received = []

    async def server(reader, writer):
        try:
            received.append(await reader.readuntil(b"\r\n\r\n"))
            occupied.set()
            await finish.wait()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    listener = await asyncio.start_server(server, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
    manager = AdmissionManager(snapshot(url))
    client = manager.client()
    monkeypatch.setattr(http_utils, "get_request_client", lambda role: client)
    async with (
        listener,
        httpx.AsyncClient(limits=httpx.Limits(max_connections=1), timeout=None, trust_env=False) as http,
    ):
        first = asyncio.create_task(http.get(url))
        pending = None
        try:
            await asyncio.wait_for(occupied.wait(), 2)
            gateway = InferenceGateway("rollout", manager.fetch, request_client=client, proxy_client=http)
            pending = asyncio.create_task(
                gateway.generate({"input_ids": [1]})
                if entry == "proxy"
                else http_utils._post(http, f"{url}/generate", {}, inference={"role": "rollout"})
            )

            async def queued():
                # Observe the actual HTTPX pool queue, not a timing assumption.
                while len(http._transport._pool._requests) != 2:
                    await asyncio.sleep(0)

            await asyncio.wait_for(queued(), 2)
            assert len(manager.gate._permits) == 1 and len(received) == 1
            with pytest.raises(AdmissionDrainError):
                await asyncio.to_thread(manager.gate.drain, lambda: None, timeout=0)
            with pytest.raises(AdmissionDrainError):
                manager.gate.confirm_drained()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            manager.gate.drain(lambda: None, timeout=0)
            manager.gate.confirm_drained()
            manager.gate.open()
            finish.set()
            await first
            assert len(received) == 1
            assert manager.released == manager.acquired
        finally:
            finish.set()
            if pending is not None:
                pending.cancel()
            await asyncio.gather(first, *([pending] if pending else []), return_exceptions=True)


@pytest.mark.parametrize("runtime_kind", ["rollout", "static"])
def test_shutdown_keeps_queued_senders_fenced(monkeypatch, runtime_kind):
    from types import SimpleNamespace

    from relax.distributed.ray.multi_engine_manager import MultiEngineManager
    from relax.distributed.ray.rollout import RolloutEngineRuntime
    from relax.engine.inference.routing import ModelUnavailableError

    monkeypatch.setenv("RELAX_INFERENCE_ADMISSION_GRACE_S", "0")
    manager = AdmissionManager(snapshot())
    # Exercise the production entrypoint's default: no client-provided tracking flag.
    cls = RolloutEngineRuntime if runtime_kind == "rollout" else MultiEngineManager
    runtime = object.__new__(cls)
    runtime._admission = manager.gate
    runtime._inference_snapshot = manager.fetch
    events = []
    if runtime_kind == "rollout":
        runtime._stop_eviction_monitor = lambda **kwargs: None
        runtime._health_monitors = []
        runtime.servers = {}
        runtime._shutdown_all_engines = lambda: events.append("released")
        monkeypatch.setattr("relax.distributed.ray.rollout.stop_launched_routers", lambda: None)
        shutdown = runtime.dispose
    else:
        runtime.all_engines, runtime.nodes_per_engine = [], 1
        runtime._pool = SimpleNamespace(shutdown=lambda: events.append("released"))
        shutdown = runtime.shutdown
    permit = runtime.acquire_inference("queued")
    generation = permit.generation
    shutdown()
    assert events == ["released"]
    runtime.release_inference(permit)
    with pytest.raises(ModelUnavailableError):
        runtime.acquire_inference("queued-after-shutdown")
    with pytest.raises(AdmissionDrainError):
        manager.gate.open()
    assert manager.gate._generation == generation
    assert not manager.gate._permits


def test_dead_sender_is_reclaimed_but_stale_dispatch_receipt_cannot_authorize_a_send(monkeypatch):
    from relax.engine.inference.holder import HolderIdentity
    from relax.engine.inference.routing import ModelUnavailableError

    manager = AdmissionManager(snapshot())
    old = manager.gate.acquire(
        manager.fetch, "dead-sender", holder=HolderIdentity(actor_id="dead"), dispatch_tracked=True
    )
    monkeypatch.setattr("relax.engine.inference.admission.holder_alive", lambda holder: False)
    manager.gate.drain(lambda: None, timeout=0)
    manager.gate.confirm_drained()
    manager.gate.open()
    with pytest.raises(ModelUnavailableError, match="revoked"):
        manager.gate.confirm_dispatch(old)
    assert not manager.gate._permits
