# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Both GenRM entry points run one request protocol.

The Gateway ``/generate`` and the GenRM backend ``/generate`` share
``InferenceRequestExecutor``: the same admission, target resolution, retry,
abort and permit release. Every case drives both entries against one fake
Manager and one mock engine transport.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import Request


try:
    from fastapi import HTTPException

    import relax.components.genrm as genrm_module
    from relax.components.inference_gateway import InferenceGateway
    from relax.engine.inference import executor as executor_module
    from relax.engine.inference.executor import InferenceRequestExecutor
    from relax.engine.inference.types import (
        LifecycleState,
        ModelSnapshot,
        ReplicaSnapshot,
        Role,
        RoleSnapshot,
        RoutingSpec,
    )

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="requires ray[serve] + relax deps")

MODEL = "judge"
OLD = "http://192.0.2.1:16001"
NEW = "http://192.0.2.2:16002"
MESSAGES = [{"role": "user", "content": "q"}]


class _Manager:
    """A Manager handle serving one direct-routed GenRM model."""

    def __init__(self) -> None:
        self.urls = [OLD]
        self.admission = True
        self.admitted: list[str] = []
        self.completed: list[str] = []

        async def snapshot(role):
            state = LifecycleState.READY if self.admission else LifecycleState.DRAINING
            replicas = tuple(
                ReplicaSnapshot(f"{MODEL}/replica-{index}", state, url, direct_eligible=self.admission)
                for index, url in enumerate(self.urls)
            )
            return RoleSnapshot(
                role=Role.GENRM,
                manager_epoch="epoch",
                models=(ModelSnapshot(MODEL, replicas, state=state, admission=self.admission),),
                routing=RoutingSpec(default_model=MODEL, route_key_to_model=((MODEL, MODEL),)),
            )

        async def admit_request(role, model_id, trace_id):
            assert (role, model_id) == (Role.GENRM, MODEL)
            if not self.admission:
                raise RuntimeError("not ready for inference")
            permit = f"permit-{len(self.admitted)}"
            self.admitted.append(permit)
            return permit

        async def complete_request(permit):
            self.completed.append(permit)

        self.snapshot = SimpleNamespace(remote=snapshot)
        self.admit_request = SimpleNamespace(remote=admit_request)
        self.complete_request = SimpleNamespace(remote=complete_request)

    def assert_each_permit_returned_once(self) -> None:
        assert self.completed == self.admitted


@dataclass
class _Engine:
    """Mock engines: ``reply`` answers each ``/generate`` the way a test
    needs."""

    reply: object = None
    generates: list[tuple[str, dict]] = field(default_factory=list)
    aborts: list[tuple[str, str]] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        base = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        body = json.loads(request.content)
        if request.url.path == "/abort_request":
            self.aborts.append((base, body["rid"]))
            return httpx.Response(200, json={})
        assert request.url.path == "/generate"
        self.generates.append((base, body))
        self.started.set()
        if self.reply is not None:
            return await self.reply(request, base)
        return httpx.Response(200, json={"text": " 1 "})


async def _adapt(key, messages, sampling_params=None):
    assert key == MODEL
    return {"input_ids": [1], "sampling_params": sampling_params or {}}


def _gateway_entry(manager: _Manager, client: httpx.AsyncClient, adapt):
    gateway = InferenceGateway(
        Role.GENRM,
        manager_handle=manager,
        genrm_backend_handle=SimpleNamespace(prepare_generate_payload=SimpleNamespace(remote=adapt)),
    )
    gateway._executor.client = client
    body = json.dumps({"messages": MESSAGES, "model": MODEL}).encode()

    async def call() -> tuple[int, str | None]:
        async def receive():
            return {"type": "http.request", "body": body}

        request = Request(
            {"type": "http", "method": "POST", "path": "/generate", "query_string": b"", "headers": []}, receive
        )
        response = await gateway.proxy(request, "generate")
        data = json.loads(response.body)
        return response.status_code, data.get("response")

    return call


def _backend_entry(manager: _Manager, client: httpx.AsyncClient, adapt):
    replica = object.__new__(genrm_module.GenRM.func_or_class.__bases__[0])
    replica._logger_instance = None
    replica.genrm_managers = {MODEL: object()}
    replica._executor = InferenceRequestExecutor(Role.GENRM, manager, client)
    replica.prepare_generate_payload = adapt

    async def call() -> tuple[int, str | None]:
        try:
            response = await replica.generate(genrm_module.GenerateRequest(messages=MESSAGES, route_key=MODEL))
        except HTTPException as exc:
            return exc.status_code, None
        return 200, response.response

    return call


@pytest.fixture(params=["gateway", "backend"])
def entry(request, monkeypatch):
    monkeypatch.setattr(executor_module, "_RETRY_BACKOFF_S", 0.0)
    build = _gateway_entry if request.param == "gateway" else _backend_entry

    def make(manager: _Manager, engine: _Engine, adapt=_adapt):
        return build(manager, httpx.AsyncClient(transport=httpx.MockTransport(engine)), adapt)

    return make


@pytest.mark.asyncio
async def test_executor_refuses_requests_after_drain_starts(entry) -> None:
    manager, engine = _Manager(), _Engine()
    manager.admission = False

    status, _ = await entry(manager, engine)()

    assert status == 503
    assert engine.generates == [] and manager.admitted == manager.completed == []


@pytest.mark.asyncio
async def test_executor_finishes_requests_admitted_before_drain(entry) -> None:
    manager = _Manager()

    async def drain_then_reply(request, base):
        manager.admission = False
        return httpx.Response(200, json={"text": " 1 "})

    engine = _Engine(drain_then_reply)

    assert await entry(manager, engine)() == (200, "1")
    assert engine.aborts == []
    manager.assert_each_permit_returned_once()


@pytest.mark.asyncio
async def test_executor_retries_an_unsent_request_on_the_recovered_endpoint(entry) -> None:
    manager = _Manager()

    async def old_endpoint_gone(request, base):
        if base == OLD:
            manager.urls = [NEW]
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={"text": " 1 "})

    engine = _Engine(old_endpoint_gone)

    assert await entry(manager, engine)() == (200, "1")
    (old, first), (new, second) = engine.generates
    assert (old, new) == (OLD, NEW) and first["rid"] == second["rid"]
    assert engine.aborts == []
    manager.assert_each_permit_returned_once()


@pytest.mark.asyncio
async def test_executor_does_not_retry_an_unsent_request_into_a_drain(entry) -> None:
    manager = _Manager()

    async def refuse_during_drain(request, base):
        manager.admission = False
        raise httpx.ConnectError("connection refused", request=request)

    engine = _Engine(refuse_during_drain)

    status, _ = await entry(manager, engine)()

    assert status == 503
    assert len(engine.generates) == 1 and engine.aborts == []
    manager.assert_each_permit_returned_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["read_timeout", "server_error"])
async def test_executor_aborts_and_never_replays_a_sent_request(entry, failure: str) -> None:
    manager = _Manager()

    async def outcome_unknown(request, base):
        if failure == "read_timeout":
            raise httpx.ReadTimeout("reply lost", request=request)
        return httpx.Response(500, json={"error": "engine failed"})

    engine = _Engine(outcome_unknown)

    status, _ = await entry(manager, engine)()

    assert status == (502 if failure == "read_timeout" else 500)
    ((target, sent),) = engine.generates
    assert engine.aborts == [(target, sent["rid"])]
    manager.assert_each_permit_returned_once()


@pytest.mark.asyncio
async def test_executor_aborts_a_cancelled_request(entry) -> None:
    manager = _Manager()

    async def hang(request, base):
        await asyncio.Event().wait()

    engine = _Engine(hang)
    task = asyncio.create_task(entry(manager, engine)())
    await engine.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    ((target, sent),) = engine.generates
    assert engine.aborts == [(target, sent["rid"])]
    manager.assert_each_permit_returned_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancels", [1, 2])
async def test_executor_returns_a_permit_whose_reply_was_cancelled(entry, cancels: int) -> None:
    """Cancelled after the Manager recorded the permit, before its reply."""
    manager = _Manager()
    admit, recorded, reply = manager.admit_request.remote, asyncio.Event(), asyncio.Event()

    async def record_then_hold_reply(*args):
        permit = await admit(*args)
        recorded.set()
        await reply.wait()
        return permit

    manager.admit_request = SimpleNamespace(remote=record_then_hold_reply)
    engine = _Engine()
    task = asyncio.create_task(entry(manager, engine)())
    await recorded.wait()
    for _ in range(cancels):
        task.cancel()
        await asyncio.sleep(0)
    assert not task.done()  # Still waiting to claim the permit.
    reply.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert engine.generates == []
    assert manager.admitted == ["permit-0"]
    manager.assert_each_permit_returned_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "batch"),
    [
        ({"input_ids": [1], "sampling_params": {}}, None),
        ({"input_ids": [[1], [2]], "sampling_params": {}}, 2),
        # A one-item batch is still a batch: its logical ID is sent explicitly.
        ({"input_ids": [[1]], "sampling_params": {}}, 1),
        # n > 1: aborting the logical ID aborts its parallel-sampling children.
        ({"input_ids": [1], "sampling_params": {"n": 2}}, None),
        ({"input_ids": [[1], [2]], "sampling_params": {"n": 2}}, 2),
    ],
)
async def test_gateway_aborts_the_logical_rids_the_engine_runs(monkeypatch, payload: dict, batch: int | None) -> None:
    monkeypatch.setattr(executor_module, "_RETRY_BACKOFF_S", 0.0)
    manager = _Manager()

    async def lost(request, base):
        raise httpx.ReadTimeout("reply lost", request=request)

    async def adapt(key, messages, sampling_params=None):
        return dict(payload)

    engine = _Engine(lost)
    call = _gateway_entry(manager, httpx.AsyncClient(transport=httpx.MockTransport(engine)), adapt)

    status, _ = await call()

    assert status == 502
    ((target, sent),) = engine.generates
    if batch is None:
        assert isinstance(sent["rid"], str)
        logical = [sent["rid"]]
    else:
        assert isinstance(sent["rid"], list) and len(sent["rid"]) == batch
        logical = sent["rid"]
    assert engine.aborts == [(target, rid) for rid in logical]
    manager.assert_each_permit_returned_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_rids", [("job", "job_extra"), ("job", "job")])
async def test_executions_own_backend_rids_no_other_request_can_match(
    entry, monkeypatch, caller_rids: tuple[str, str]
) -> None:
    """SGLang aborts every rid starting with the aborted one."""
    sent = []

    async def adapt(key, messages, sampling_params=None):
        return {"input_ids": [1], "sampling_params": {}, "rid": caller_rids[len(sent)]}

    async def record(request, base):
        sent.append(json.loads(request.content)["rid"])
        return httpx.Response(200, json={"text": " 1 "})

    manager, engine = _Manager(), _Engine(record)
    call = entry(manager, engine, adapt)
    await call()
    await call()

    first, second = sent
    assert first not in caller_rids and second not in caller_rids
    assert not first.startswith(second) and not second.startswith(first)
    assert all(rid.startswith(executor_module._BACKEND_RID_PREFIX) for rid in sent)
    manager.assert_each_permit_returned_once()


@pytest.mark.asyncio
async def test_executor_returns_the_permit_when_tokenization_fails(entry) -> None:
    manager, engine = _Manager(), _Engine()

    async def broken_template(key, messages, sampling_params=None):
        raise ValueError("chat template failed")

    with pytest.raises(ValueError, match="chat template"):
        await entry(manager, engine, broken_template)()

    assert engine.generates == [] and engine.aborts == []
    assert manager.admitted and manager.completed == manager.admitted
