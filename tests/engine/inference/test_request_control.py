# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Control timeouts leave late admission/release work owned and observable."""

import asyncio
from types import SimpleNamespace

import pytest

from relax.engine.inference.admission import AdmissionDrainError
from relax.engine.inference.client import InferenceRequestClient
from relax.engine.inference.routing import ModelUnavailableError
from tests.engine.inference.helpers import AdmissionManager
from tests.engine.inference.test_request_dispatch import snapshot


@pytest.mark.parametrize("stall", ["acquire", "grant", "release"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_request_control_deadline_keeps_unknown_permits_until_late_receipt(monkeypatch, stall, cancel):
    monkeypatch.setenv("RELAX_INFERENCE_CONTROL_TIMEOUT_S", "0.04")
    manager = AdmissionManager(snapshot())
    entered, finish, returned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    sent = []

    async def acquire(request_id, **selection):
        if stall == "acquire":
            entered.set()
            await finish.wait()
        permit = manager.acquire_inference(request_id, **selection)
        if stall == "grant":
            entered.set()
            await finish.wait()
        return permit

    async def release(permit):
        if stall == "release":
            entered.set()
            await finish.wait()
        manager.release_inference(permit)
        returned.set()

    remote = SimpleNamespace(
        acquire_inference=SimpleNamespace(remote=acquire), release_inference=SimpleNamespace(remote=release)
    )

    async def resources():
        return {"snapshot": manager.fetch().to_dict(), "managers": {"default": remote}}

    client = InferenceRequestClient("rollout", resources)

    async def work():
        async with client.request():
            sent.append("HTTP completed")

    task = asyncio.create_task(work())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if cancel:
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
        done, _ = await asyncio.wait({task}, timeout=0.3)
        assert done, "control RPC exceeded its budget, even after repeated cancellation"
        if stall == "release" and not cancel:
            await task
        else:
            with pytest.raises(asyncio.CancelledError if cancel else ModelUnavailableError):
                await task
        assert sent == (["HTTP completed"] if stall == "release" else [])
        assert client._cleanup_tasks, "late RPC cleanup must remain owned after the caller exits"
        if stall != "acquire":
            assert len(manager.gate._permits) == 1
            with pytest.raises(AdmissionDrainError):
                manager.gate.drain(lambda: None, timeout=0)
            with pytest.raises(AdmissionDrainError):
                manager.gate.confirm_drained()
        finish.set()
        await asyncio.wait_for(returned.wait(), 1)
        await asyncio.gather(*client._cleanup_tasks)
        await asyncio.sleep(0)  # Run completion callbacks even if gather saw an already-done task.
        assert manager.released == manager.acquired
        assert len(manager.released) == 1
        assert not client._cleanup_tasks
        manager.gate.drain(lambda: pytest.fail("late permit leaked"), timeout=0)
        manager.gate.confirm_drained()
        manager.gate.open()
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_bootstrap_control_budget_is_bounded_and_late_discovery_remains_reusable(monkeypatch):
    monkeypatch.setenv("RELAX_INFERENCE_CONTROL_TIMEOUT_S", "0.02")
    manager = AdmissionManager(snapshot())
    finish = asyncio.Event()

    async def resources():
        await finish.wait()
        return {"snapshot": manager.fetch().to_dict(), "managers": {"default": manager}}

    client = InferenceRequestClient("rollout", resources)
    task = asyncio.create_task(client.request().__aenter__())
    try:
        done, _ = await asyncio.wait({task}, timeout=0.2)
        assert done, "bootstrap must not bypass the request control budget"
        with pytest.raises(ModelUnavailableError):
            await task
        assert manager.acquired == []
        finish.set()
        await asyncio.gather(*client._bootstraps.values())
        async with client.request():
            assert len(manager.gate._permits) == 1
        assert not manager.gate._permits
    finally:
        finish.set()
        outcome = await asyncio.gather(task, return_exceptions=True)
        if outcome and hasattr(outcome[0], "aclose"):
            await outcome[0].aclose()


@pytest.mark.parametrize("stall", ["acquire", "release"])
async def test_teacher_budget_cancels_control_wait_without_waiting_for_control_deadline(monkeypatch, stall):
    from relax.engine.rollout.on_policy_distillation import OpdManager

    monkeypatch.setenv("RELAX_INFERENCE_CONTROL_TIMEOUT_S", "1")
    manager = AdmissionManager(snapshot())
    finish, entered = asyncio.Event(), asyncio.Event()

    async def acquire(request_id, **selection):
        permit = manager.acquire_inference(request_id, **selection)
        if stall == "acquire":
            entered.set()
            await finish.wait()
        return permit

    async def release(permit):
        if stall == "release":
            entered.set()
            await finish.wait()
        manager.release_inference(permit)

    remote = SimpleNamespace(
        acquire_inference=SimpleNamespace(remote=acquire), release_inference=SimpleNamespace(remote=release)
    )

    async def resources():
        return {"snapshot": manager.fetch().to_dict(), "managers": {"default": remote}}

    client = InferenceRequestClient("teacher", resources)
    opd = object.__new__(OpdManager)
    opd.args = SimpleNamespace(opd_teacher_timeout_s=0.02)

    async def attempt():
        async with client.request():
            return {}

    task = asyncio.create_task(opd._retry_teacher_request(attempt))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        done, _ = await asyncio.wait({task}, timeout=0.2)
        assert done, "request cleanup overran the teacher's total retry budget"
        with pytest.raises(ModelUnavailableError, match="retry budget.*exhausted"):
            await task
        assert client._cleanup_tasks and manager.gate._permits
        with pytest.raises(AdmissionDrainError):
            manager.gate.drain(lambda: None, timeout=0)
    finally:
        finish.set()
        await asyncio.gather(task, *client._cleanup_tasks, return_exceptions=True)
    assert manager.released == manager.acquired


async def test_lost_release_ack_retry_cannot_return_a_new_generation_permit(monkeypatch):
    monkeypatch.setenv("RELAX_INFERENCE_HOLDER_POLL_S", "0.01")
    manager = AdmissionManager(snapshot())
    released, retry = asyncio.Event(), asyncio.Event()
    calls = []

    async def acquire(request_id, **selection):
        return manager.acquire_inference(request_id, **selection)

    async def release(permit):
        calls.append(permit)
        if len(calls) == 1:
            manager.release_inference(permit)
            released.set()
            raise ConnectionError("release executed, acknowledgement lost")
        await retry.wait()
        manager.release_inference(permit)

    remote = SimpleNamespace(
        acquire_inference=SimpleNamespace(remote=acquire), release_inference=SimpleNamespace(remote=release)
    )

    async def resources():
        return {"snapshot": manager.fetch().to_dict(), "managers": {"default": remote}}

    request = await InferenceRequestClient("rollout", resources).request().__aenter__()
    returning = asyncio.create_task(request.aclose())
    try:
        await asyncio.wait_for(released.wait(), 1)
        manager.gate.drain(lambda: None, timeout=0)
        manager.gate.confirm_drained()
        manager.gate.open()
        new = manager.gate.acquire(manager.fetch, "new-generation", dispatch_tracked=True)
        retry.set()
        await asyncio.wait_for(returning, 1)
        assert list(manager.gate._permits.values()) == [new]
        assert calls == [request._permit, request._permit]
        manager.gate.release(new)
    finally:
        retry.set()
        await returning
