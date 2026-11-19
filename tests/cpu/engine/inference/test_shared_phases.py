# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio

import pytest

from relax.engine.inference.lifecycle import AdmissionLedger
from relax.engine.inference.phases import PhaseCoordinator


@pytest.mark.asyncio
async def test_shared_phase_waits_for_real_permit_completion_before_loading_scorer():
    ledger = AdmissionLedger()
    permit = ledger.admit("student")
    draining = asyncio.Event()
    loaded = asyncio.Event()

    async def call(owner, method):
        if method == "offload":
            draining.set()
            await ledger.drain("phase", timeout=1)
        else:
            assert not ledger.permits
            loaded.set()

    coordinator = PhaseCoordinator({"generate": ("student",), "score": ("teacher",)}, "generate", call)
    task = asyncio.create_task(coordinator.switch("score"))
    await draining.wait()
    assert not loaded.is_set()
    with pytest.raises(RuntimeError, match="admission is closed"):
        ledger.admit("student")
    await ledger.complete(permit)
    await task
    assert loaded.is_set()


@pytest.mark.asyncio
async def test_shared_phase_unconfirmed_release_blocks_all_next_handoffs():
    loaded = []

    async def call(owner, method):
        if method == "offload":
            raise TimeoutError("release unconfirmed")
        loaded.append(owner)

    coordinator = PhaseCoordinator({"generate": ("student",), "score": ("teacher",)}, "generate", call)
    with pytest.raises(TimeoutError, match="release unconfirmed"):
        await coordinator.switch("score")
    with pytest.raises(RuntimeError, match="requires recovery"):
        await coordinator.switch("generate")
    assert loaded == []


@pytest.mark.asyncio
async def test_shared_phase_cancellation_waits_until_handoff_finishes():
    started = asyncio.Event()
    release = asyncio.Event()
    loaded = []

    async def call(owner, method):
        if method == "offload":
            started.set()
            await release.wait()
        else:
            loaded.append(owner)

    coordinator = PhaseCoordinator({"generate": ("student",), "score": ("teacher",)}, "generate", call)
    task = asyncio.create_task(coordinator.switch("score"))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert loaded == ["teacher"]


@pytest.mark.asyncio
async def test_shared_rollout_allows_training_only_after_round_and_all_releases(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from relax.distributed.ray import rollout_worker
    from relax.distributed.ray.inference_handle import ManagedRolloutHandle

    scoring = asyncio.Event()
    finish_scoring = asyncio.Event()
    releasing = asyncio.Event()
    finish_release = asyncio.Event()
    released = []
    dynamic_size = None

    async def call(owner, method):
        if method == "offload":
            if owner == "judge":
                releasing.set()
                await finish_release.wait()
            released.append(owner)

    coordinator = PhaseCoordinator(
        {"inference": ("student", "judge"), "teacher": ("teacher",), "startup": ()}, "inference", call
    )
    coordinator_rpc = SimpleNamespace(switch=SimpleNamespace(remote=coordinator.switch))

    async def generate(rollout_id):
        nonlocal dynamic_size
        await coordinator.switch("teacher")
        scoring.set()
        await finish_scoring.wait()
        await coordinator.switch("inference")
        dynamic_size = 4

    monkeypatch.setattr(rollout_worker, "init_tracking", Mock())
    monkeypatch.setattr(rollout_worker, "init_http_client", Mock())
    monkeypatch.setattr(rollout_worker, "attach_tq_client", Mock())
    monkeypatch.setattr(
        rollout_worker,
        "RolloutWorkload",
        Mock(return_value=SimpleNamespace(generate=generate, get_dynamic_global_batch_size=lambda: dynamic_size)),
    )
    args = SimpleNamespace(_inference_phase_coordinator=coordinator_rpc, tq_config=None)
    manager = SimpleNamespace(
        begin_rollout=SimpleNamespace(remote=AsyncMock()), wait_ready=SimpleNamespace(remote=AsyncMock())
    )
    worker = rollout_worker.RolloutWorker.__ray_metadata__.modified_class(args, None, manager)
    worker_rpc = SimpleNamespace(
        offload=SimpleNamespace(remote=lambda: worker.offload()),
        get_status=SimpleNamespace(remote=lambda: worker.get_status()),
    )
    owner = SimpleNamespace(
        pool_call=SimpleNamespace(remote=lambda method: "offload" if "student" in released else "onload")
    )
    handle = ManagedRolloutHandle(owner, worker_rpc, coordinator_rpc)

    assert handle.get_status.remote() == "onload"
    round_task = asyncio.create_task(worker.generate(0))
    # The first handoff already released the student. Treat a first batch as
    # visible while the workload is still computing its final batch size.
    await asyncio.wait_for(releasing.wait(), 2)
    finish_release.set()
    await asyncio.wait_for(scoring.wait(), 2)
    assert "student" in released
    assert handle.get_status.remote() == "onload"
    assert worker.get_dynamic_global_batch_size() is None
    with pytest.raises(RuntimeError, match="before generation and scoring complete"):
        await handle.offload.remote()

    finish_scoring.set()
    await asyncio.wait_for(round_task, 2)
    assert worker.get_dynamic_global_batch_size() == 4
    assert handle.get_status.remote() == "onload"

    releasing.clear()
    finish_release.clear()
    handoff = asyncio.create_task(handle.offload.remote())
    await asyncio.wait_for(releasing.wait(), 2)
    assert handle.get_status.remote() == "onload"
    finish_release.set()
    await asyncio.wait_for(handoff, 2)
    assert released == ["student", "judge", "teacher", "student", "judge"]
    assert handle.get_status.remote() == "offload"


@pytest.mark.asyncio
async def test_failed_handoff_allows_teardown_without_reopening_generation():
    calls = []
    failed = True

    async def call(owner, method):
        calls.append((owner, method))
        if owner == "teacher" and method == "onload" and failed:
            raise TimeoutError("partial onload")

    coordinator = PhaseCoordinator(
        {"inference": ("student",), "teacher": ("teacher",), "startup": ()}, "inference", call
    )
    with pytest.raises(TimeoutError):
        await coordinator.switch("teacher")
    failed = False
    await coordinator.switch("startup")
    assert calls[-2:] == [("student", "offload"), ("teacher", "offload")]
    with pytest.raises(RuntimeError, match="requires recovery"):
        await coordinator.switch("inference")
