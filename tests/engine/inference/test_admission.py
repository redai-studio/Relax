# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from relax.engine.inference.admission import AdmissionDrainError, AdmissionGate
from relax.engine.inference.client import InferenceRequestClient
from relax.engine.inference.discovery import EngineState, RoleSnapshot, build_model_snapshot
from relax.engine.inference.gateway import InferenceGateway
from relax.engine.inference.routing import ModelUnavailableError
from relax.utils import http_utils
from tests.engine.inference.helpers import AdmissionManager


def snapshot(url="http://engine:1", **kwargs):
    return RoleSnapshot(
        "rollout", 1, models=(build_model_snapshot("default", [(0, url, EngineState.READY)], **kwargs),)
    )


def test_duplicate_request_ids_remain_independent_during_drain():
    gate = AdmissionGate()
    current = snapshot()
    first = gate.acquire(lambda: current, "same-client-id")
    second = gate.acquire(lambda: current, "same-client-id")
    assert first != second
    gate.release(first)
    draining, completed = threading.Event(), threading.Event()

    def drain():
        gate.close()
        draining.set()
        gate.drain(lambda: None, timeout=5)
        completed.set()

    thread = threading.Thread(target=drain)
    thread.start()
    try:
        assert draining.wait(1)
        assert not completed.is_set()
        gate.release(second)
        assert completed.wait(1)
    finally:
        gate.release(second)
        thread.join(timeout=6)


def test_dead_permit_holder_does_not_replace_the_engine_idle_receipt(monkeypatch):
    from relax.engine.inference import admission
    from relax.engine.inference.holder import HolderIdentity

    gate = AdmissionGate()
    current = snapshot()
    old = gate.acquire(lambda: current, "request", holder=HolderIdentity(actor_id="dead"))
    monkeypatch.setattr(admission, "holder_alive", lambda holder: False, raising=False)
    idle, release = threading.Event(), threading.Event()
    events = []

    def offload():
        gate.drain(lambda: events.append("abort"), timeout=0)
        idle.set()
        release.wait()
        events.append("engine idle")
        gate.confirm_drained()
        events.append("release")

    thread = threading.Thread(target=offload)
    thread.start()
    try:
        assert idle.wait(1)
        assert "release" not in events
        with pytest.raises(ModelUnavailableError):
            gate.acquire(lambda: current, "late")
        release.set()
        thread.join(timeout=1)
        assert events[-2:] == ["engine idle", "release"]
        gate.open()
        new = gate.acquire(lambda: current, "request")
        gate.release(old)
        assert gate._permits
        gate.release(new)
    finally:
        release.set()
        thread.join(timeout=1)


async def test_proxy_and_lookaside_share_drain_even_with_cached_ready_snapshot(monkeypatch):
    manager = AdmissionManager(snapshot())
    client = manager.client()
    monkeypatch.setattr(http_utils, "get_request_client", lambda role: client)
    gateway = InferenceGateway("rollout", manager.fetch, request_client=client)
    await gateway.snapshot()
    entered = asyncio.Queue()
    finish = asyncio.Event()

    async def dispatch(request):
        entered.put_nowait(request)
        await finish.wait()
        return httpx.Response(200, json={"text": "done"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(dispatch)) as http:
        gateway._proxy_client = http
        proxy = asyncio.create_task(gateway.generate({"input_ids": [1]}))
        direct = asyncio.create_task(
            http_utils._post(http, "http://startup:9/generate", {}, inference={"role": "rollout"})
        )
        assert str((await entered.get()).url) == "http://engine:1/generate"
        assert str((await entered.get()).url) == "http://engine:1/generate"
        aborted = []
        manager.gate.close()
        drain = asyncio.create_task(asyncio.to_thread(manager.gate.drain, lambda: aborted.append(True), timeout=1))
        with pytest.raises(HTTPException) as error:
            await gateway.generate({"input_ids": [2]})
        assert error.value.status_code == 503
        assert error.value.headers == {"Retry-After": "5"}
        with pytest.raises(httpx.HTTPStatusError) as error:
            await http_utils._post(http, "http://startup:9/generate", {}, max_retries=1, inference={"role": "rollout"})
        assert error.value.response.status_code == 503
        assert not drain.done()
        assert entered.empty()
        finish.set()
        await asyncio.gather(proxy, direct, drain)
        assert aborted == []
        assert manager.released == manager.acquired


def test_unreturned_permit_is_soft_but_reopen_requires_engine_receipt():
    gate = AdmissionGate()
    current = snapshot()
    permit = gate.acquire(lambda: current, "request")
    aborted = []
    gate.drain(lambda: aborted.append(True), timeout=0)
    assert aborted == [True]
    assert gate.view(current).models[0].state == EngineState.DRAINING
    with pytest.raises(ModelUnavailableError):
        gate.acquire(lambda: current, "next")
    with pytest.raises(AdmissionDrainError):
        gate.open()
    gate.confirm_drained()
    gate.open()
    new = gate.acquire(lambda: current, "request")
    assert new.generation != permit.generation
    gate.release(permit)  # A late duplicate return cannot discharge the new generation.
    gate.drain(lambda: None, timeout=0)
    assert len(gate._permits) == 1
    gate.release(new)
    gate.drain(lambda: pytest.fail("already drained"), timeout=0)


@pytest.mark.parametrize("reject", [False, True])
async def test_cancelled_acquire_returns_a_late_permit_even_after_repeated_cancel(reject):
    manager = AdmissionManager(snapshot())
    queued, complete = asyncio.Event(), asyncio.Event()

    async def acquire(request_id, **selection):
        queued.set()
        await complete.wait()
        if reject:
            raise ModelUnavailableError("draining")
        return manager.acquire_inference(request_id, **selection)

    async def release(permit):
        manager.release_inference(permit)

    remote = SimpleNamespace(
        acquire_inference=SimpleNamespace(remote=acquire), release_inference=SimpleNamespace(remote=release)
    )

    async def resources():
        return {"snapshot": manager.fetch().to_dict(), "managers": {"default": remote}}

    permit = InferenceRequestClient("rollout", resources).request()
    task = asyncio.create_task(permit.__aenter__())
    await queued.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    complete.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(manager.acquired) == (0 if reject else 1)
    assert manager.released == manager.acquired
    manager.gate.drain(lambda: pytest.fail("permit leaked"), timeout=0)


async def test_retry_releases_then_acquires_the_current_replica(monkeypatch):
    manager = AdmissionManager(snapshot())
    monkeypatch.setattr(http_utils, "get_request_client", lambda role: manager.client())
    seen = []

    async def dispatch(request):
        seen.append(str(request.url))
        if len(seen) == 1:
            manager.snapshot = snapshot("http://replacement:2")
            return httpx.Response(503, text="restarting")
        assert len(manager.released) == 1
        return httpx.Response(200, json={"text": "done"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(dispatch)) as http:
        await http_utils._post(http, "http://startup:9/generate", {}, inference={"role": "rollout"})
    assert seen == ["http://engine:1/generate", "http://replacement:2/generate"]
    assert manager.acquired == manager.released
    assert manager.acquired[0].request_id != manager.acquired[1].request_id


@pytest.mark.parametrize("stream", [False, True])
async def test_external_model_validation_and_served_alias_match_models_listing(stream):
    manager = AdmissionManager(snapshot(served_model_name="/models/policy"))
    client = manager.client()
    gateway = InferenceGateway("rollout", manager.fetch, request_client=client)
    with pytest.raises(HTTPException) as error:
        await gateway.generate({"model": "typo", "stream": stream})
    assert error.value.status_code == 400
    assert manager.acquired == []
    assert [row["id"] for row in (await gateway.models())["data"]] == ["default", "/models/policy"]
    async with client.request(model="/models/policy") as permit:
        assert permit.target.model == "default"


def test_predict_direct_selects_ready_heads_but_pd_stays_behind_router():
    gate = AdmissionGate()
    regular = snapshot(router_url="http://router:3")
    direct = gate.acquire(lambda: regular, "direct", direct=True)
    assert direct.target.base_url == "http://engine:1"
    gate.release(direct)
    pd = snapshot(
        router_url="http://router:3",
        diagnostic_workers=[
            ("prefill-0", "http://prefill:1", EngineState.READY),
            ("decode-0", "http://decode:1", EngineState.READY),
        ],
    )
    routed = gate.acquire(lambda: pd, "pd", direct=True)
    assert routed.target.base_url == "http://router:3"
    assert routed.target.via_router


async def test_distributed_sender_owns_permit_until_its_transport_finishes(monkeypatch):
    manager = AdmissionManager(snapshot())
    monkeypatch.setattr(http_utils, "get_request_client", lambda role: manager.client())
    entered, finish = asyncio.Event(), asyncio.Event()

    async def dispatch(request):
        entered.set()
        await finish.wait()
        return httpx.Response(200, json={"text": "done"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(dispatch)) as http:

        def remote(*args, **kwargs):
            return asyncio.create_task(http_utils._post_with_error_envelope(http, *args, **kwargs))

        actor = SimpleNamespace(do_post=SimpleNamespace(remote=remote))
        monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
        monkeypatch.setattr(http_utils, "_post_actors", [actor])
        caller = asyncio.create_task(http_utils.post("http://startup:9/generate", {}, inference={"role": "rollout"}))
        await entered.wait()
        with pytest.raises(AdmissionDrainError):
            manager.gate.drain(lambda: None, timeout=0)
        assert manager.gate._permits
        finish.set()
        assert await caller == {"text": "done"}
        manager.gate.drain(lambda: pytest.fail("sender leaked its permit"), timeout=0)


async def test_distributed_managed_request_never_resends_after_unknown_actor_failure(monkeypatch):
    async def fail(*args, **kwargs):
        raise RuntimeError("lost actor reply")

    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    monkeypatch.setattr(http_utils, "_post_actors", [SimpleNamespace(do_post=SimpleNamespace(remote=fail))])
    monkeypatch.setattr(http_utils, "_post", lambda *a, **kw: pytest.fail("duplicate local send"))
    with pytest.raises(RuntimeError, match="lost actor reply"):
        await http_utils.post("http://startup:9/generate", {}, inference={"role": "rollout"})


@pytest.mark.parametrize("iterate", [False, True])
async def test_native_stream_holds_permit_until_upstream_closes(iterate):
    manager = AdmissionManager(snapshot())
    gateway = InferenceGateway("rollout", manager.fetch, request_client=manager.client())

    class Stream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b"data: token\n\n"

        async def aclose(self):
            assert manager.released == []
            self.closed = True

    stream = Stream()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))
    ) as http:
        gateway._proxy_client = http
        response = await gateway.generate({"stream": True})
        if iterate:
            assert await anext(response.body_iterator) == b"data: token\n\n"
        manager.gate.drain(lambda: None, timeout=0)
        assert manager.gate._permits
        await response.aclose()
        assert stream.closed
        assert manager.acquired == manager.released
        manager.gate.drain(lambda: pytest.fail("stream leaked permit"), timeout=0)


async def test_bootstrap_is_shared_and_one_cancelled_waiter_does_not_cancel_others():
    owner = AdmissionManager(snapshot())
    entered, finish = asyncio.Event(), asyncio.Event()
    fetches = []

    async def resources():
        fetches.append(True)
        entered.set()
        await finish.wait()
        return {"snapshot": owner.fetch().to_dict(), "managers": {"default": owner}}

    client = InferenceRequestClient("rollout", resources)
    first = asyncio.create_task(client.resources())
    second = asyncio.create_task(client.resources())
    await entered.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    finish.set()
    assert (await second)["managers"]["default"] is owner
    assert fetches == [True]


async def test_managed_connection_failure_reports_without_waiting_for_recovery():
    manager = AdmissionManager(snapshot())
    reports = []
    # A remote submission returns an unresolved reference. Request cleanup
    # must still return its permit without waiting for the health operation.
    manager.report_failure = SimpleNamespace(remote=lambda url: reports.append(url) or object())
    with pytest.raises(ConnectionError):
        async with manager.client().request():
            raise ConnectionError("teacher connection lost")
    assert reports == ["http://engine:1"]
    assert manager.released == manager.acquired
