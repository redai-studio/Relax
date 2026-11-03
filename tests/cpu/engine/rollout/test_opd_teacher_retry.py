# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import aiohttp
import pytest

from relax.engine.inference.client import InferenceRequestClient
from relax.engine.inference.discovery import EngineState
from relax.engine.inference.routing import ModelUnavailableError, UnknownModelError
from relax.engine.rollout import on_policy_distillation as opd
from relax.utils.types import Sample
from tests.cpu.engine.inference.helpers import AdmissionManager
from tests.cpu.engine.rollout.test_opd_teacher_discovery import Session, snapshot


@pytest.fixture
def retry_clock(monkeypatch):
    clock = SimpleNamespace(now=0.0, delays=[], after_wait=lambda: None)

    async def sleep(delay):
        clock.delays.append(delay)
        clock.now += delay
        clock.after_wait()

    monkeypatch.setattr(opd, "monotonic", lambda: clock.now, raising=False)
    monkeypatch.setattr(opd, "sleep", sleep, raising=False)
    return clock


def _manager(timeout=2):
    manager = object.__new__(opd.OpdManager)
    manager.args = SimpleNamespace(opd_teacher_timeout_s=timeout)
    return manager


async def _send(manager, session, **selection):
    return await manager._post_logprob(
        session,
        "http://startup:9/generate",
        {},
        Sample(index=0),
        "teacher prefill",
        inference={"role": "teacher", **selection},
    )


@pytest.mark.parametrize("state", [EngineState.SLEEPING, EngineState.DRAINING, EngineState.DEAD])
async def test_managed_teacher_retries_until_replacement_is_ready(monkeypatch, retry_clock, state):
    owner = AdmissionManager(snapshot("__default__", state=state))
    monkeypatch.setattr(opd, "get_request_client", lambda role: owner.client(role))
    session = Session()

    def recover():
        assert session.urls == []
        owner.snapshot = snapshot("__default__", host="replacement")

    retry_clock.after_wait = recover
    assert await _send(_manager(), session) is not None
    assert session.urls == ["http://replacement:1/generate"]
    assert owner.acquired == owner.released


async def test_teacher_discovery_failure_retries_without_startup_fallback(monkeypatch, retry_clock):
    owner = AdmissionManager(snapshot("__default__", host="new-port"))
    attempts = 0

    async def resources():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("discovery unreachable")
        return await owner.client("teacher").resources()

    client = InferenceRequestClient("teacher", resources)
    monkeypatch.setattr(opd, "get_request_client", lambda role: client)
    session = Session()
    await _send(_manager(), session)
    assert attempts == 3
    assert retry_clock.delays == [0.25, 0.5]
    assert session.urls == ["http://new-port:1/generate"]


async def test_teacher_retry_budget_exhaustion_is_explicit(monkeypatch, retry_clock):
    owner = AdmissionManager(snapshot("__default__", state=EngineState.DEAD))
    monkeypatch.setattr(opd, "get_request_client", lambda role: owner.client(role))
    session = Session()
    with pytest.raises(ModelUnavailableError, match="retry budget.*exhausted"):
        await _send(_manager(timeout=1), session)
    assert retry_clock.now == 1
    assert session.urls == []


async def test_unknown_teacher_fails_without_retry(monkeypatch, retry_clock):
    owner = AdmissionManager(snapshot("math", "code"))
    monkeypatch.setattr(opd, "get_request_client", lambda role: owner.client(role))
    with pytest.raises(UnknownModelError):
        await _send(_manager(), Session(), model="missing")
    assert retry_clock.delays == []


async def test_mopd_retry_keeps_teacher_selection(monkeypatch, retry_clock):
    owner = AdmissionManager(snapshot("math", "code", state=EngineState.DRAINING))
    monkeypatch.setattr(opd, "get_request_client", lambda role: owner.client(role))
    retry_clock.after_wait = lambda: setattr(owner, "snapshot", snapshot("math", "code", host="rebuilt"))
    await _send(_manager(), Session(), route_key="code", affinity_key=3)
    assert owner.acquired[0].target.model == "code"


async def test_teacher_http_retry_after_is_respected_and_permit_released(monkeypatch, retry_clock):
    owner = AdmissionManager(snapshot("__default__"))
    monkeypatch.setattr(opd, "get_request_client", lambda role: owner.client(role))

    class Busy(Session):
        @asynccontextmanager
        async def post(self, url, **kwargs):
            self.urls.append(url)
            if len(self.urls) == 1:
                raise aiohttp.ClientResponseError(
                    SimpleNamespace(real_url=url), (), status=503, headers={"Retry-After": "1"}
                )
            yield self

    session = Busy()
    retry_clock.after_wait = lambda: pytest.fail("retry kept a permit") if owner.acquired != owner.released else None
    await _send(_manager(), session)
    assert retry_clock.delays == [1.0]
    assert len(session.urls) == 2 and owner.acquired == owner.released


async def test_external_teacher_does_not_gain_managed_retries(monkeypatch, retry_clock):
    class Failed(Session):
        @asynccontextmanager
        async def post(self, url, **kwargs):
            self.urls.append(url)
            raise aiohttp.ClientConnectionError("external unavailable")
            yield  # pragma: no cover

    session = Failed()
    assert await _manager()._post_logprob(session, "external", {}, Sample(index=0), "teacher") is None
    assert retry_clock.delays == [] and session.urls == ["external"]


@pytest.mark.parametrize("recovers", [True, False])
async def test_teacher_connection_retries_hint_each_failure_without_holding_permits(
    monkeypatch, retry_clock, recovers
):
    owner = AdmissionManager(snapshot("__default__"))
    hints = []
    # Model an owner whose recovery RPC has not replied. Neither permit
    # cleanup nor a later HTTP attempt may wait for that reply.
    pending_hint = asyncio.get_running_loop().create_future()
    owner.report_failure = SimpleNamespace(remote=lambda url: hints.append(url) or pending_hint)
    monkeypatch.setattr(opd, "get_request_client", lambda role: owner.client(role))

    class Disconnected(Session):
        @asynccontextmanager
        async def post(self, url, **kwargs):
            self.urls.append(url)
            if len(self.urls) <= 2 or not recovers:
                raise aiohttp.ClientConnectionError("teacher vanished")
            yield self

    session = Disconnected()

    def after_wait():
        assert len(hints) == len(session.urls)
        assert owner.acquired == owner.released
        assert not pending_hint.done()
        if len(hints) == 2 and recovers:
            owner.snapshot = snapshot("__default__", host="replacement")

    retry_clock.after_wait = after_wait
    manager = _manager(timeout=2 if recovers else 0.75)
    if recovers:
        assert await asyncio.wait_for(_send(manager, session), 5) is not None
        assert session.urls == ["http://engine:1/generate"] * 2 + ["http://replacement:1/generate"]
    else:
        with pytest.raises(ModelUnavailableError, match="retry budget.*exhausted"):
            await asyncio.wait_for(_send(manager, session), 5)
        assert session.urls == ["http://engine:1/generate"] * 2
    assert hints == ["http://engine:1"] * 2
    assert retry_clock.delays == [0.25, 0.5]
    assert owner.acquired == owner.released
    assert not pending_hint.done()


async def test_hanging_teacher_discovery_obeys_total_budget(monkeypatch):
    started = asyncio.Event()

    async def resources():
        started.set()
        await asyncio.Event().wait()

    client = InferenceRequestClient("teacher", resources)
    monkeypatch.setattr(opd, "get_request_client", lambda role: client)
    task = asyncio.create_task(_send(_manager(timeout=0.05), Session()))
    try:
        await started.wait()
        with pytest.raises(ModelUnavailableError, match="retry budget.*exhausted"):
            await asyncio.wait_for(task, 1)
    finally:
        for pending in list(client._bootstraps.values()):
            pending.cancel()
        await asyncio.gather(*client._bootstraps.values(), return_exceptions=True)
