# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""A cancellation delivered to the HTTP body does not restart cleanup wait."""

import asyncio
from types import SimpleNamespace

import pytest

from relax.engine.inference.admission import AdmissionDrainError
from relax.engine.inference.client import InferenceRequestClient
from relax.engine.inference.routing import ModelUnavailableError
from tests.engine.inference.helpers import AdmissionManager
from tests.engine.inference.test_request_dispatch import snapshot


@pytest.mark.parametrize("cause", ["teacher-timeout", "external-cancel"])
async def test_body_cancellation_does_not_restart_release_control_budget(monkeypatch, cause):
    from relax.engine.rollout.on_policy_distillation import OpdManager

    monkeypatch.setenv("RELAX_INFERENCE_CONTROL_TIMEOUT_S", "1")
    manager = AdmissionManager(snapshot())
    body_entered, body_cancelled = asyncio.Event(), asyncio.Event()
    release_entered, release_receipt = asyncio.Event(), asyncio.Event()
    returns = []

    async def acquire(request_id, **selection):
        return manager.acquire_inference(request_id, **selection)

    async def release(permit):
        returns.append(permit)
        release_entered.set()
        await release_receipt.wait()
        manager.release_inference(permit)

    remote = SimpleNamespace(
        acquire_inference=SimpleNamespace(remote=acquire), release_inference=SimpleNamespace(remote=release)
    )

    async def resources():
        return {"snapshot": manager.fetch().to_dict(), "managers": {"default": remote}}

    client = InferenceRequestClient("teacher", resources)
    opd = object.__new__(OpdManager)
    opd.args = SimpleNamespace(opd_teacher_timeout_s=0.03)

    async def attempt():
        async with client.request():
            body_entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                body_cancelled.set()

    task = asyncio.create_task(opd._retry_teacher_request(attempt) if cause == "teacher-timeout" else attempt())
    try:
        await asyncio.wait_for(body_entered.wait(), 1)
        if cause == "external-cancel":
            task.cancel()  # One cancellation, consumed in the HTTP body.
        await asyncio.wait_for(body_cancelled.wait(), 1)
        await asyncio.wait_for(release_entered.wait(), 1)
        done, _ = await asyncio.wait({task}, timeout=0.15)
        assert done, "body cancellation must not wait for a new one-second release budget"
        if cause == "teacher-timeout":
            with pytest.raises(ModelUnavailableError, match="retry budget.*exhausted"):
                await task
        else:
            with pytest.raises(asyncio.CancelledError):
                await task
        assert len(manager.gate._permits) == 1 and client._cleanup_tasks
        assert manager.released == [] and returns == manager.acquired
        with pytest.raises(AdmissionDrainError):
            manager.gate.drain(lambda: None, timeout=0)
        release_receipt.set()
        await asyncio.wait_for(asyncio.gather(*client._cleanup_tasks), 1)
        await asyncio.sleep(0)  # Deliver already-completed task callbacks.
        assert manager.released == manager.acquired and len(returns) == 1
        assert not manager.gate._permits and not client._cleanup_tasks
    finally:
        release_receipt.set()
        await asyncio.gather(task, *client._cleanup_tasks, return_exceptions=True)
