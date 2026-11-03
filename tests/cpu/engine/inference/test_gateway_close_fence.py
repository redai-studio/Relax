# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Response headers let engine evidence retire a failed HTTP close's permit."""

import asyncio
import inspect
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
import ray
from fastapi import HTTPException
from ray.exceptions import ActorDiedError, ActorUnavailableError

from relax.distributed.ray.engine_pool import EnginePool, EnginePoolSpec
from relax.distributed.ray.inference_manager import InferenceManager
from relax.distributed.ray.multi_engine_manager import MultiEngineManager
from relax.distributed.ray.placement_planner import plan_placement
from relax.engine.inference.admission import AdmissionGate
from relax.engine.inference.client import InferenceRequestClient
from relax.engine.inference.discovery import RoleSnapshot, TopologyRevision
from relax.engine.inference.gateway import InferenceGateway
from relax.engine.inference.holder import current_holder
from relax.engine.inference.ownership import OwnershipLedger


ENTRIES = ("native", "chat", "native_stream", "chat_stream")
MODEL = "__default__"


class Handle:
    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        async def remote(*args, **kwargs):
            result = getattr(self.target, name)(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result

        return SimpleNamespace(remote=remote)


class Engine:
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return SimpleNamespace(remote=lambda **kwargs: (name, kwargs))


def make_runtime():
    runtime = object.__new__(MultiEngineManager)
    runtime._admission = AdmissionGate()
    runtime._unhealthy = set()
    runtime.nodes_per_engine = 1
    runtime.all_engines = [Engine()]
    runtime._topology_revision = TopologyRevision()
    runtime._pool = EnginePool(
        EnginePoolSpec(
            actor_class=lambda: None,
            resolve_placement=lambda slot: None,
            actor_options=lambda *args: {},
            ctor=lambda *args: ((), {}),
            init_kwargs=lambda *args: {},
        ),
        runtime.all_engines,
    )
    runtime._engine_addr_and_ports = runtime._pool.addresses
    runtime._pool.addresses[0] = {"host": "engine", "port": 9000}
    return runtime


class Upstream(httpx.AsyncByteStream):
    def __init__(self, case):
        self.case = case

    async def __aiter__(self):
        if self.case.entry.endswith("stream"):
            yield b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
        else:
            yield b'{"text":"ok","choices":[]}'

    async def aclose(self):
        self.case.close_calls += 1
        self.case.closing.set()
        await self.case.allow_close.wait()
        if self.case.close_fails:
            raise self.case.close_error


async def settle(client):
    async def finish():
        while client._cleanup_tasks:
            await asyncio.gather(*client._cleanup_tasks, return_exceptions=True)
            await asyncio.sleep(0)  # Deliver tracking callbacks, not a timing barrier.

    await asyncio.wait_for(finish(), 2)


class CloseFenceCase:
    def __init__(self, entry):
        self.entry = entry
        self.events = []
        args = SimpleNamespace(
            resource={"genrm": [1, 1]},
            colocate=False,
            _genrm_instances_resolved={MODEL: {"num_gpus": 1, "num_gpus_per_engine": 1}},
        )
        self.plan = plan_placement(args)
        self.ledger = OwnershipLedger(self.plan)
        args._inference_authority = Handle(self.ledger)
        self.owner = InferenceManager.__ray_metadata__.modified_class(args, "genrm")
        self.runtime = make_runtime()
        self.owner._runtimes[MODEL] = self.runtime
        self.owner._claims[MODEL] = (self.plan.claim("genrm"),)
        self.owner._locks[MODEL] = asyncio.Lock()
        self.ledger.register_model("genrm", MODEL, Handle(self.owner), self.plan.claim("genrm"))
        self.acquired, self.confirmed, self.released = [], [], []
        self.confirm_failures = []
        self.terminal_error = None
        self.confirming, self.allow_confirm = asyncio.Event(), asyncio.Event()
        self.closing, self.allow_close = asyncio.Event(), asyncio.Event()
        self.allow_confirm.set()
        self.allow_close.set()
        self.close_fails, self.close_calls, self.status = True, 0, 200
        self.close_error = OSError("transport close did not confirm exit")
        self.client = InferenceRequestClient("genrm", self.resources)

    async def acquire_inference(self, request_id, **selection):
        permit = await self.owner.acquire_inference(MODEL, request_id, **selection)
        self.acquired.append(permit)
        return permit

    async def confirm_inference_dispatch(self, permit):
        self.confirmed.append(permit)
        if self.confirm_failures:
            raise self.confirm_failures.pop(0)
        self.confirming.set()
        await self.allow_confirm.wait()
        if self.terminal_error is not None:
            raise self.terminal_error
        await self.owner.confirm_inference_dispatch(MODEL, permit)

    async def release_inference(self, permit):
        self.released.append(permit)
        await self.owner.release_inference(MODEL, permit)

    async def resources(self):
        return {"snapshot": self.owner.get_inference_snapshot(), "managers": {MODEL: Handle(self)}}

    def ray_get(self, ref, timeout=None):
        if isinstance(ref, list):
            return [self.ray_get(item, timeout) for item in ref]
        self.events.append(ref[0])
        return True

    async def invoke(self):
        stream = self.entry.endswith("stream")
        if self.entry.startswith("native"):
            response = await self.gateway.generate({"input_ids": [1], "stream": stream})
        else:
            response = await self.gateway.chat_completions(json.dumps({"messages": [], "stream": stream}).encode(), {})
        if stream:
            return [chunk async for chunk in response.body_iterator]
        return response

    async def failed_close(self):
        with pytest.raises(RuntimeError, match="Upstream close did not confirm completion"):
            await self.invoke()
        assert self.close_calls == 1
        assert not self.released
        assert len(self.runtime._admission._permits) == 1
        assert self.ledger.snapshot()["resident"]

    async def offload(self):
        await self.owner.apply(MODEL, "offload", self.lease, "offload")
        assert not self.runtime._admission._permits
        assert not self.runtime._admission._dispatched
        assert not self.runtime.is_onloaded()
        assert not self.ledger.snapshot()["resident"]
        assert self.events.index("abort_requests") < self.events.index("release_memory_occupation")
        assert not self.released  # Only engine evidence retired the permit.


@asynccontextmanager
async def close_case(monkeypatch, entry):
    monkeypatch.setenv("RELAX_INFERENCE_ADMISSION_GRACE_S", "0")
    monkeypatch.setenv("RELAX_INFERENCE_HOLDER_CHECK_S", "0.1")
    monkeypatch.setenv("RELAX_INFERENCE_HOLDER_POLL_S", "0.01")
    monkeypatch.setenv("RELAX_INFERENCE_CONTROL_TIMEOUT_S", "1")
    current_holder()  # Initialize holder evidence before testing control deadlines.
    case = CloseFenceCase(entry)
    monkeypatch.setattr(ray, "get", case.ray_get)
    monkeypatch.setattr(ray, "kill", lambda *args, **kwargs: None)
    case.lease = await case.ledger.acquire("case", "case", case.plan.claims, role="genrm")
    case.ledger.reserve_activation(case.lease, "genrm", MODEL)

    async def upstream(request):
        return httpx.Response(case.status, stream=Upstream(case))

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http:
        case.gateway = InferenceGateway(
            "genrm",
            lambda: RoleSnapshot.from_dict(case.owner.get_inference_snapshot()),
            request_client=case.client,
            proxy_client=http,
        )
        try:
            yield case
        finally:
            # Release event barriers even on a failed assertion, then revoke
            # through real shutdown so a broken retry cannot leak into tests.
            case.allow_close.set()
            await case.owner.apply(MODEL, "shutdown", case.lease, "cleanup")
            case.allow_confirm.set()
            await settle(case.client)
            await case.ledger.release(case.lease)


@pytest.mark.parametrize("entry", ENTRIES)
async def test_failed_close_dispatch_evidence_allows_real_owner_offload(monkeypatch, entry):
    async with close_case(monkeypatch, entry) as case:
        if entry.endswith("stream"):
            case.confirm_failures = [ConnectionError("first dispatch confirmation lost")]
        await case.failed_close()
        await settle(case.client)
        await case.offload()
        assert case.confirmed == case.acquired * (2 if entry.endswith("stream") else 1)


@pytest.mark.parametrize("entry", ["native", "chat"])
@pytest.mark.parametrize("error", [ConnectionError("lost reply"), ActorUnavailableError("partition", None)])
async def test_failed_close_retries_transient_confirmation(monkeypatch, entry, error):
    async with close_case(monkeypatch, entry) as case:
        case.confirm_failures = [error]
        await case.failed_close()
        await settle(case.client)
        await case.offload()
        assert case.confirmed == case.acquired * 2


@pytest.mark.parametrize("entry", ["native_stream", "chat_stream"])
async def test_failed_close_reuses_late_confirmation_ack(monkeypatch, entry):
    async with close_case(monkeypatch, entry) as case:
        monkeypatch.setenv("RELAX_INFERENCE_CONTROL_TIMEOUT_S", "0.03")
        case.allow_confirm.clear()
        await case.failed_close()  # The foreground deadline ends before the ACK.
        assert case.confirming.is_set()
        assert case.confirmed == case.acquired
        assert case.client._cleanup_tasks
        assert not case.runtime._admission._dispatched
        case.allow_confirm.set()
        await settle(case.client)
        await case.offload()
        assert case.confirmed == case.acquired  # No duplicate RPC while waiting.


@pytest.mark.parametrize("entry", ["native", "chat"])
@pytest.mark.parametrize("terminal", ["owner_dead", "revoked", "replaced"])
async def test_failed_close_confirmation_ends_on_owner_death_or_revocation(monkeypatch, entry, terminal):
    async with close_case(monkeypatch, entry) as case:
        case.confirm_failures = [ConnectionError("first attempt interrupted")]
        case.allow_confirm.clear()
        await case.failed_close()
        await asyncio.wait_for(case.confirming.wait(), 1)
        assert case.confirmed == case.acquired * 2
        replacement = None
        if terminal == "owner_dead":
            case.terminal_error = ActorDiedError()
        else:
            await case.owner.apply(MODEL, "shutdown", case.lease, "retire-old-generation")
            if terminal == "replaced":
                case.runtime = make_runtime()
                case.owner._runtimes[MODEL] = case.runtime
                await case.owner.apply(MODEL, "onload", case.lease, "replace-generation")
                replacement = await case.owner.acquire_inference(MODEL, "replacement", holder=current_holder())
                assert replacement.generation != case.acquired[0].generation
        case.allow_confirm.set()
        await settle(case.client)
        assert not case.client._cleanup_tasks
        assert case.confirmed == case.acquired * 2
        assert not case.released
        assert not case.runtime._admission._dispatched
        if replacement is not None:
            assert list(case.runtime._admission._permits.values()) == [replacement]


@pytest.mark.parametrize("entry", ["native_stream", "chat_stream"])
async def test_confirmed_stream_close_failure_still_offloads_without_extra_rpc(monkeypatch, entry):
    async with close_case(monkeypatch, entry) as case:
        await case.failed_close()
        await settle(case.client)
        await case.offload()
        assert case.confirmed == case.acquired


@pytest.mark.parametrize("entry", ["native", "chat"])
@pytest.mark.parametrize("status", [200, 429])
async def test_successful_close_nonstream_keeps_two_control_rpcs(monkeypatch, entry, status):
    async with close_case(monkeypatch, entry) as case:
        case.close_fails, case.status = False, status
        if status == 200:
            assert await case.invoke() == {"text": "ok", "choices": []}
        else:
            with pytest.raises(HTTPException) as error:
                await case.invoke()
            assert error.value.status_code == status
        await settle(case.client)
        assert case.close_calls == 1
        assert not case.confirmed
        assert len(case.acquired) == 1
        assert case.released == case.acquired
        assert not case.runtime._admission._permits


@pytest.mark.parametrize("entry", ["native", "chat"])
async def test_cancelled_frontend_still_confirms_late_close_failure(monkeypatch, entry):
    async with close_case(monkeypatch, entry) as case:
        case.allow_close.clear()
        frontend = asyncio.create_task(case.invoke())
        await asyncio.wait_for(case.closing.wait(), 1)
        frontend.cancel()
        with pytest.raises(asyncio.CancelledError):
            await frontend
        assert not case.released
        assert not case.confirmed
        assert len(case.runtime._admission._permits) == 1
        case.allow_close.set()
        await settle(case.client)
        await case.offload()
        assert case.confirmed == case.acquired


@pytest.mark.parametrize("entry", ENTRIES)
async def test_cancelled_close_task_keeps_permit_until_engine_retirement(monkeypatch, entry):
    async with close_case(monkeypatch, entry) as case:
        # This is the close operation's own terminal failure, not cancellation
        # of the frontend waiting on its shielded receipt.
        case.close_error = asyncio.CancelledError("transport close cancelled itself")
        if entry.endswith("stream"):
            case.confirm_failures = [ConnectionError("first dispatch confirmation lost")]
        await case.failed_close()
        await settle(case.client)
        await case.offload()
        assert case.confirmed == case.acquired * (2 if entry.endswith("stream") else 1)
