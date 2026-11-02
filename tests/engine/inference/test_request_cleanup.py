# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Permit return retries end with definitive owner death or a receipt."""

import asyncio
from types import SimpleNamespace

import pytest
from ray.exceptions import ActorDiedError, ActorUnavailableError

from relax.engine.inference.client import InferenceRequestClient
from tests.engine.inference.helpers import AdmissionManager
from tests.engine.inference.test_request_dispatch import snapshot


def client_with_release(manager, release):
    async def acquire(request_id, **selection):
        return manager.acquire_inference(request_id, **selection)

    remote = SimpleNamespace(
        acquire_inference=SimpleNamespace(remote=acquire), release_inference=SimpleNamespace(remote=release)
    )

    async def resources():
        return {"snapshot": manager.fetch().to_dict(), "managers": {"default": remote}}

    return InferenceRequestClient("rollout", resources)


async def test_owner_death_ends_permit_return_and_removes_cleanup_task(monkeypatch):
    monkeypatch.setenv("RELAX_INFERENCE_CONTROL_TIMEOUT_S", "0.04")
    monkeypatch.setenv("RELAX_INFERENCE_HOLDER_POLL_S", "0.01")
    manager = AdmissionManager(snapshot())
    calls = []
    warnings = []
    monkeypatch.setattr(
        "relax.engine.inference.client.logger.warning",
        lambda message, *args, **kwargs: warnings.append(message % args),
    )

    async def release(permit):
        calls.append(permit)
        raise ActorDiedError()

    client = client_with_release(manager, release)
    request = await client.request().__aenter__()
    try:
        await asyncio.wait_for(request.aclose(), 0.3)
        await asyncio.sleep(0)  # Deliver the cleanup task's completion callback.
        assert request._release_task.done()
        assert not client._cleanup_tasks
        assert calls == [request._permit]
        assert len([message for message in warnings if "owner" in message.lower()]) == 1
    finally:
        if request._release_task is not None and not request._release_task.done():
            request._release_task.cancel()
            await asyncio.gather(request._release_task, return_exceptions=True)


@pytest.mark.parametrize("error", [ConnectionError("lost reply"), ActorUnavailableError("partition", None)])
async def test_transient_return_error_retries_until_receipt_and_releases_tracking(monkeypatch, error):
    monkeypatch.setenv("RELAX_INFERENCE_HOLDER_POLL_S", "0.01")
    manager = AdmissionManager(snapshot())
    calls = []

    async def release(permit):
        calls.append(permit)
        if len(calls) == 1:
            raise error
        manager.release_inference(permit)

    client = client_with_release(manager, release)
    request = await client.request().__aenter__()
    await asyncio.wait_for(request.aclose(), 1)
    await asyncio.sleep(0)
    assert calls == [request._permit, request._permit]
    assert manager.released == manager.acquired
    assert not client._cleanup_tasks


@pytest.mark.parametrize("cancelled", [False, True])
async def test_finished_cleanup_tasks_are_logged_and_removed_even_without_receipt(monkeypatch, cancelled):
    client = InferenceRequestClient("rollout")
    entered, finish = asyncio.Event(), asyncio.Event()
    warnings = []
    monkeypatch.setattr(
        "relax.engine.inference.client.logger.warning",
        lambda message, *args, **kwargs: warnings.append(message % args),
    )

    async def cleanup():
        entered.set()
        await finish.wait()
        raise RuntimeError("cleanup receipt failed")

    task = client._track_cleanup(asyncio.create_task(cleanup()))
    await entered.wait()
    if cancelled:
        task.cancel()
    else:
        finish.set()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert task.done() and not client._cleanup_tasks
    assert len([message for message in warnings if "cleanup" in message.lower()]) == 1
