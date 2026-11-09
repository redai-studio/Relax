# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from aiohttp import web


@pytest_asyncio.fixture
async def engine():
    # The transport is real so cache-flush, URL and response-success checks run.
    from relax.distributed.ray.teacher_manager import SharedTeacherManager

    state = SimpleNamespace(
        current="/A",
        loads=[],
        scores=[],
        started=asyncio.Event(),
        release=asyncio.Event(),
        fail_update=False,
    )

    async def model_info(request):
        return web.json_response({"model_path": state.current})

    async def reload(request):
        body = await request.json()
        assert body["flush_cache"] is True
        state.loads.append(body["weight_version"])
        if state.fail_update:
            return web.json_response({"success": False, "message": "injected reload failure"})
        state.current = body["model_path"]
        return web.json_response({"success": True})

    async def generate(request):
        token = (await request.json())["input_ids"][0]
        state.scores.append((state.current, token))
        if token == 1:
            state.started.set()
            await state.release.wait()
        return web.json_response({"scored_by": state.current, "input": token})

    app = web.Application()
    app.add_routes([web.get("/model_info", model_info), web.post("/update_weights_from_disk", reload)])
    app.router.add_post("/generate", generate)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    physical = MagicMock()
    physical.offload.remote = AsyncMock()
    physical.onload.remote = AsyncMock()
    physical.shutdown.remote = AsyncMock()
    args = SimpleNamespace(opd_teacher_timeout_s=10, opd_teacher_connector_limit=5)
    cls = SharedTeacherManager.__ray_metadata__.modified_class
    url = f"http://127.0.0.1:{runner.addresses[0][1]}/generate"
    owner = cls(args, physical, url, {"A": "/A", "B": "/B"})
    try:
        await owner.initialize()
        yield owner, state, physical
    finally:
        state.release.set()
        if owner._worker is not None:
            await owner._worker
        await owner._session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_grouped_scores_use_the_right_checkpoint_after_active_request_finishes(engine):
    owner, state, _ = engine
    a1 = asyncio.create_task(owner.score("A", {"input_ids": [1]}))
    await asyncio.wait_for(state.started.wait(), 5)

    b1 = asyncio.create_task(owner.score("B", {"input_ids": [2]}))
    b2 = asyncio.create_task(owner.score("B", {"input_ids": [3]}))
    await asyncio.sleep(0)
    # Cancelling A1's caller does not finish its HTTP request.
    a1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await a1
    assert state.scores == [("/A", 1)]
    assert state.loads == []

    state.release.set()
    results = await asyncio.wait_for(asyncio.gather(b1, b2), 5)
    assert results == [{"scored_by": "/B", "input": 2}, {"scored_by": "/B", "input": 3}]
    assert state.loads == ["B"]


@pytest.mark.asyncio
async def test_training_waits_for_scores_and_restores_last_teacher_on_resume(engine):
    owner, state, physical = engine
    a1 = asyncio.create_task(owner.score("A", {"input_ids": [1]}))
    await asyncio.wait_for(state.started.wait(), 5)
    b1 = asyncio.create_task(owner.score("B", {"input_ids": [2]}))
    offload = asyncio.create_task(owner.offload())
    await asyncio.sleep(0)
    assert owner.get_status()["state"] == "draining"
    with pytest.raises(RuntimeError, match="draining"):
        await owner.score("A", {"input_ids": [3]})
    physical.offload.remote.assert_not_called()
    state.release.set()
    await asyncio.wait_for(asyncio.gather(a1, b1, offload), 5)
    physical.offload.remote.assert_awaited_once()
    assert owner.get_status()["state"] == "offloaded"
    state.current = "/A"  # A restarted physical engine comes up with its initial checkpoint.
    await owner.onload()
    await owner.onload()
    assert state.loads == ["B", "B"]
    assert (await owner.score("B", {"input_ids": [4]}))["scored_by"] == "/B"
    physical.onload.remote.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_reload_rejects_waiters_and_never_hands_memory_to_training(engine):
    owner, state, physical = engine
    owner._limit = 1  # A must wake up and fail if B fails to reload while it holds the only slot.
    state.fail_update = True
    results = await asyncio.gather(
        owner.score("B", {"input_ids": [2]}), owner.score("A", {"input_ids": [3]}), return_exceptions=True
    )
    assert all(isinstance(result, RuntimeError) for result in results)
    assert state.scores == []
    assert owner.get_status()["state"] == "failed"
    with pytest.raises(RuntimeError, match="failed"):
        await owner.score("A", {"input_ids": [4]})
    with pytest.raises(RuntimeError, match="failed"):
        await owner.offload()
    physical.offload.remote.assert_not_called()


@pytest.mark.asyncio
async def test_current_teacher_preference_expires_between_token_bounded_batches(engine, monkeypatch):
    from relax.distributed.ray import teacher_manager

    owner, state, _ = engine
    owner._batch_tokens = 2
    owner._limit = 6
    now = [0.0]
    monkeypatch.setattr(teacher_manager, "monotonic", lambda: now[0])
    batches = []
    original_take_batch = owner._take_batch

    def record_batch():
        teacher, batch = original_take_batch()
        tokens = [payload["input_ids"][0] for payload in batch.values()]
        batches.append((teacher, tokens))
        if tokens == [4, 5]:
            now[0] = 3.0  # B's oldest request reaches the fairness threshold.
        return teacher, batch

    monkeypatch.setattr(owner, "_take_batch", record_batch)
    a1 = asyncio.create_task(owner.score("A", {"input_ids": [1]}))
    await asyncio.wait_for(state.started.wait(), 5)
    requests = [
        asyncio.create_task(owner.score(teacher, {"input_ids": tokens}))
        for teacher, tokens in (("B", [2, 2, 2]), ("B", [3]), ("A", [4]), ("A", [5]), ("A", [6]))
    ]
    await asyncio.sleep(0)
    state.release.set()
    await asyncio.wait_for(asyncio.gather(a1, *requests), 5)
    assert batches == [("A", [1]), ("A", [4, 5]), ("B", [2]), ("B", [3]), ("A", [6])]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["offload", "onload", "checkpoint"])
async def test_failed_training_transition_keeps_admission_closed(engine, failure):
    owner, state, physical = engine
    if failure == "offload":
        physical.offload.remote.side_effect = RuntimeError("physical offload failed")
        transition = owner.offload
    else:
        await owner.offload()
        if failure == "onload":
            physical.onload.remote.side_effect = RuntimeError("physical onload failed")
        else:
            state.fail_update = True
        transition = owner.onload
    with pytest.raises(RuntimeError, match="failed"):
        await transition()
    assert owner.get_status() == {"state": "failed", "teacher_id": None, "pending": 0}
    with pytest.raises(RuntimeError, match="failed"):
        await owner.score("B", {"input_ids": [4]})


@pytest.mark.asyncio
async def test_capacity_waiter_replaces_cancelled_queued_score(engine):
    owner, state, _ = engine
    owner._limit = 2
    a1 = asyncio.create_task(owner.score("A", {"input_ids": [1]}))
    await asyncio.wait_for(state.started.wait(), 5)
    queued = asyncio.create_task(owner.score("B", {"input_ids": [2]}))
    waiting = asyncio.create_task(owner.score("B", {"input_ids": [3]}))
    b1 = asyncio.create_task(owner.score("B", {"input_ids": [4]}))
    await asyncio.sleep(0)
    assert owner.get_status()["pending"] == 2

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert owner.get_status()["pending"] == 2
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    await asyncio.sleep(0)
    assert owner.get_status()["pending"] == 2

    state.release.set()
    a, b = await asyncio.wait_for(asyncio.gather(a1, b1), 5)
    assert (a["input"], b["input"]) == (1, 4)
    assert state.scores == [("/A", 1), ("/B", 4)]
    assert owner.get_status()["pending"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["offload", "shutdown"])
async def test_training_or_shutdown_rejects_capacity_waiter_before_draining(engine, monkeypatch, transition):
    import ray

    monkeypatch.setattr(ray, "kill", MagicMock())
    owner, state, _ = engine
    owner._limit = 1
    a1 = asyncio.create_task(owner.score("A", {"input_ids": [1]}))
    await asyncio.wait_for(state.started.wait(), 5)
    b1 = asyncio.create_task(owner.score("B", {"input_ids": [2]}))
    await asyncio.sleep(0)
    assert not b1.done()
    close = asyncio.create_task(getattr(owner, transition)())
    with pytest.raises(RuntimeError, match="stopped accepting"):
        await asyncio.wait_for(b1, 5)
    assert not close.done()
    assert state.loads == []
    state.release.set()
    await asyncio.wait_for(asyncio.gather(a1, close), 5)
    assert state.scores == [("/A", 1)]
    if transition == "offload":
        await owner.onload()
        assert (await owner.score("B", {"input_ids": [3]}))["scored_by"] == "/B"
        assert state.scores == [("/A", 1), ("/B", 3)]


@pytest.mark.asyncio
async def test_initialize_rejects_wrong_initial_checkpoint(engine):
    owner, _, physical = engine
    other = type(owner)(SimpleNamespace(opd_teacher_timeout_s=10), physical, owner.url, {"B": "/B"})
    with pytest.raises(RuntimeError, match="expected checkpoint '/B', got '/A'"):
        await other.initialize()
