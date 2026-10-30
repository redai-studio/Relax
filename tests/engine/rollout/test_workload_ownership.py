# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
from types import SimpleNamespace

import pytest

from relax.distributed.ray.placement_planner import Phase, PlacementClaim, PlacementPlan
from relax.engine.inference.ownership import OwnershipLedger
from relax.engine.inference.phase_client import current_phase
from relax.engine.rollout import workload as workload_module


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
    assert events == ["onload", "predict closed", "offload", "detach"]
    assert not ledger.snapshot()["owners"]
