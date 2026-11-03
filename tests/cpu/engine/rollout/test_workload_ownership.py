# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from relax.distributed.ray.inference_manager import InferenceManager, ModelManager
from relax.distributed.ray.placement_planner import Phase, PlacementClaim, PlacementPlan
from relax.distributed.ray.rollout import RolloutEngineRuntime
from relax.engine.inference.admission import AdmissionGate
from relax.engine.inference.ownership import OwnershipLedger
from relax.engine.inference.phase_client import current_phase
from relax.engine.rollout import workload as workload_module
from tests.cpu.engine.inference.test_coresidency import Handle


@pytest.fixture
def workload(monkeypatch):
    claim = PlacementClaim("rollout", "__default__", "actor", 0, 1, frozenset({Phase.GENERATE}))
    ledger = OwnershipLedger(PlacementPlan((), (claim,)))
    events = []

    async def apply(model, action, lease, command, **kwargs):
        ledger.check(lease, [claim])
        events.append(action)

    ledger.register_model("rollout", "__default__", SimpleNamespace(apply=SimpleNamespace(remote=apply)), claim)

    async def claims(role):
        return ledger.claims_for(role)

    authority = SimpleNamespace(
        claims_for=SimpleNamespace(remote=claims),
        acquire=SimpleNamespace(remote=ledger.acquire),
        command=SimpleNamespace(remote=ledger.command),
        block=SimpleNamespace(remote=ledger.block),
        release=SimpleNamespace(remote=ledger.release),
    )
    cls = workload_module.RolloutWorkload.__ray_metadata__.modified_class
    instance = cls.__new__(cls)
    instance.args = SimpleNamespace(_inference_authority=authority, offload_rollout=True)
    instance.rollout_id = 0
    instance._pending_operations = set()
    instance._closing = False
    instance.data_source = None
    instance.data_system_client = object()
    instance.inference_manager = SimpleNamespace(handle=SimpleNamespace(shutdown=SimpleNamespace(remote=AsyncMock())))
    monkeypatch.setattr(workload_module, "detach_tq_client", lambda: events.append("detach"))
    return instance, ledger, events


@pytest.mark.parametrize("failure", [False, True])
async def test_cancelled_rollout_settles_callback_before_handoff_and_disposal(workload, failure):
    instance, ledger, events = workload
    entered, finish = threading.Event(), threading.Event()

    def generate(*args, **kwargs):
        assert current_phase().lease is not None  # context follows the worker thread
        entered.set()
        assert finish.wait(5)
        events.append("published")
        if failure:
            raise ValueError("callback failed after cancellation")
        return []

    task = asyncio.create_task(instance._run_phase(generate, 0))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    disposal = asyncio.create_task(instance.dispose())
    try:
        await asyncio.sleep(0)
        assert not task.done() and not disposal.done()
        assert ledger.snapshot()["owners"]
        assert events == ["onload"]
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await disposal
    assert events == ["onload", "published", "offload", "detach"]
    assert ledger.snapshot()["owners"] == {}
    assert ledger.snapshot()["blocked"] == {}
    assert instance.data_system_client is None


async def test_predict_disposal_waits_for_cleanup_and_rejects_new_work(workload, monkeypatch):
    from relax.engine.sft.predict import loop

    instance, ledger, events = workload
    entered, cleaning, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def predict(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await finish.wait()
            events.append("predict closed")

    monkeypatch.setattr(loop, "run_predict_loop", predict)
    task = asyncio.create_task(instance.run_predict(1))
    await entered.wait()
    task.cancel()
    await cleaning.wait()
    disposal = asyncio.create_task(instance.dispose())
    await asyncio.sleep(0)
    try:
        assert not disposal.done()
        assert ledger.snapshot()["owners"]
        with pytest.raises(RuntimeError, match="shutting down"):
            await instance.run_predict(2)
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await disposal
    assert events == ["onload", "predict closed", "detach"]
    assert not ledger.snapshot()["owners"]


@pytest.mark.parametrize("release_inference", [False, True])
@pytest.mark.parametrize("failure", [False, True])
async def test_public_disposal_delegates_cleanup_and_detaches_on_owner_failure(workload, release_inference, failure):
    instance, _, events = workload
    shutdown = instance.inference_manager.handle.shutdown.remote
    if failure:
        shutdown.side_effect = RuntimeError("engine exit unconfirmed")
    if release_inference and failure:
        with pytest.raises(RuntimeError, match="engine exit unconfirmed"):
            await instance.dispose(release_inference=release_inference)
    else:
        await instance.dispose(release_inference=release_inference)
    if release_inference:
        shutdown.assert_awaited_once_with()
    else:
        shutdown.assert_not_called()
    assert instance._closing
    assert events == ["detach"]
    assert instance.data_system_client is None


@pytest.mark.parametrize("failure", [False, True])
async def test_public_disposal_preserves_owner_receipts_across_retries_and_epochs(workload, monkeypatch, failure):
    instance, initial_ledger, events = workload
    ledger = OwnershipLedger(initial_ledger.plan)
    instance.args._inference_authority = Handle(ledger)
    owner = InferenceManager.__ray_metadata__.modified_class(instance.args, "rollout")
    runtime = object.__new__(RolloutEngineRuntime)
    runtime._admission = AdmissionGate()
    runtime._stop_eviction_monitor = Mock()
    runtime._health_monitors = []
    runtime._shutdown_all_engines = Mock(side_effect=RuntimeError("engine exit unconfirmed") if failure else None)
    stop_routers = Mock()
    monkeypatch.setattr("relax.distributed.ray.rollout.stop_launched_routers", stop_routers)
    owner._runtimes["__default__"] = runtime
    owner._claims["__default__"] = ledger.claims_for("rollout")
    owner._locks["__default__"] = asyncio.Lock()
    handle = Handle(owner)
    ledger.register_manager("rollout", handle)
    ledger.register_model("rollout", "__default__", handle, owner._claims["__default__"][0])
    instance.inference_manager = ModelManager(handle, "__default__")
    # Global recovery first proves training exit, then retries blocked owners.
    await ledger.stop_training()

    if failure:
        with pytest.raises(RuntimeError, match="engine exit unconfirmed"):
            await instance.dispose()
        assert ledger.snapshot()["blocked"]
        stop_routers.assert_called_once_with()
        runtime._shutdown_all_engines.side_effect = None
    await instance.dispose()
    assert runtime._admission._closed
    assert not ledger.snapshot()["owners"]
    assert not owner._blocked
    assert bool(ledger.snapshot()["blocked"]) is failure
    calls = runtime._shutdown_all_engines.call_count
    assert calls == (2 if failure else 1)
    assert stop_routers.call_count == calls

    await ledger.new_epoch()
    epoch = ledger.epoch
    await instance.dispose()
    assert ledger.epoch == epoch
    assert runtime._shutdown_all_engines.call_count == calls
    assert stop_routers.call_count == calls
    assert not ledger.snapshot()["owners"]
    assert not ledger.snapshot()["blocked"]
    assert events == ["detach"] * (3 if failure else 2)


async def test_concurrent_owner_shutdown_survives_caller_cancellation(workload, monkeypatch):
    instance, _, _ = workload
    owner = InferenceManager.__ray_metadata__.modified_class(instance.args, "rollout")
    entered, finish = asyncio.Event(), asyncio.Event()

    async def shutdown():
        entered.set()
        await finish.wait()

    cleanup = AsyncMock(side_effect=shutdown)
    monkeypatch.setattr(owner, "_shutdown", cleanup)
    first = asyncio.create_task(owner.shutdown())
    await entered.wait()
    second = asyncio.create_task(owner.shutdown())
    first.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not second.done()
        cleanup.assert_awaited_once_with()
    finally:
        finish.set()
    await second
    await owner.shutdown()
    cleanup.assert_awaited_once_with()


@pytest.mark.parametrize("operation", ["generate", "eval", "predict"])
async def test_only_generate_releases_policy_weights(workload, monkeypatch, operation):
    from relax.engine.sft.predict import loop

    instance, ledger, events = workload

    def callback(*args, **kwargs):
        events.append(operation)
        return SimpleNamespace(samples=[])

    async def predict(*args):
        callback()

    monkeypatch.setattr(loop, "run_predict_loop", predict)
    instance.args.partial_rollout = False
    instance.generate_rollout = callback
    if operation == "generate":
        await instance.generate(1)
    elif operation == "eval":
        await instance._run_phase(callback, 1, evaluation=True)
    else:
        await instance.run_predict(1)

    # Weight sync precedes eval/predict. Releasing here would discard the
    # current policy before the next generation, which has no intervening sync.
    assert events == ["onload", operation] + (["offload"] if operation == "generate" else [])
    assert not ledger.snapshot()["owners"]
