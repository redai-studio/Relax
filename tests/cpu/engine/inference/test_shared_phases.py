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
async def test_managed_rollout_offload_releases_all_generation_phase_owners():
    from types import SimpleNamespace

    from relax.distributed.ray.inference_handle import ManagedRolloutHandle

    released = []

    async def call(owner, method):
        released.append((owner, method))

    coordinator = PhaseCoordinator({"inference": ("student", "teacher", "judge"), "startup": ()}, "inference", call)
    handle = ManagedRolloutHandle(
        object(), object(), SimpleNamespace(switch=SimpleNamespace(remote=coordinator.switch))
    )
    await handle.offload.remote()
    assert released == [("student", "offload"), ("teacher", "offload"), ("judge", "offload")]


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
