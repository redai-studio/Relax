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
        version=None,
        reported={},
        loads=[],
        scores=[],
        started=asyncio.Event(),
        release=asyncio.Event(),
        fail_update=False,
    )

    async def model_info(request):
        return web.json_response({"model_path": state.current, "weight_version": state.version, **state.reported})

    async def reload(request):
        body = await request.json()
        assert body["flush_cache"] is True
        state.loads.append(body["weight_version"])
        if state.fail_update:
            return web.json_response({"success": False, "message": "injected reload failure"})
        state.current = body["model_path"]
        state.version = body["weight_version"]
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
    owner._max_wait_s = 0
    a1 = asyncio.create_task(owner.score("A", {"input_ids": [1]}))
    await asyncio.wait_for(state.started.wait(), 5)
    b1, b2, a2, cancelled = [
        asyncio.create_task(owner.score(teacher, {"input_ids": [i]}))
        for teacher, i in (("B", 2), ("B", 3), ("A", 4), ("B", 5))
    ]
    await asyncio.sleep(0)
    waiting = asyncio.create_task(owner.score("A", {"input_ids": [6]}))
    await asyncio.sleep(0)
    assert not waiting.done()
    assert owner.get_status()["pending"] == 5
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert owner.get_status()["pending"] == 4
    a1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await a1
    assert state.loads == []
    state.release.set()
    results = await asyncio.wait_for(asyncio.gather(b1, b2, a2), 5)
    assert [(r["_relax_teacher_route"], r["scored_by"], r["input"]) for r in results] == [
        ("B", "/B", 2),
        ("B", "/B", 3),
        ("A", "/A", 4),
    ]
    assert state.loads == ["B", "A"]
    assert sorted(state.scores) == [("/A", 1), ("/A", 4), ("/B", 2), ("/B", 3)]
    assert owner.get_status()["pending"] == 0


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
async def test_cancelled_shutdown_drains_accepted_request_before_stopping_engine(engine, monkeypatch):
    import ray

    owner, state, physical = engine
    monkeypatch.setattr(ray, "kill", MagicMock())
    a1 = asyncio.create_task(owner.score("A", {"input_ids": [1]}))
    await asyncio.wait_for(state.started.wait(), 5)
    closing = asyncio.create_task(owner.shutdown())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    physical.shutdown.remote.assert_not_called()
    state.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, 5)
    assert (await a1)["scored_by"] == "/A"
    physical.shutdown.remote.assert_awaited_once()
    assert owner.get_status()["state"] == "closed"


@pytest.mark.asyncio
async def test_capacity_backpressure_accepts_next_teacher_after_active_score(engine):
    owner, state, _ = engine
    owner._limit = 1
    a1 = asyncio.create_task(owner.score("A", {"input_ids": [1]}))
    await asyncio.wait_for(state.started.wait(), 5)
    b1 = asyncio.create_task(owner.score("B", {"input_ids": [2]}))
    await asyncio.sleep(0)
    assert not b1.done()
    assert owner.get_status()["pending"] == 1
    assert state.scores == [("/A", 1)]
    state.release.set()
    a, b = await asyncio.wait_for(asyncio.gather(a1, b1), 5)
    assert [a["scored_by"], b["scored_by"]] == ["/A", "/B"]
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
@pytest.mark.parametrize("stale", [{"model_path": "/A"}, {"weight_version": "A"}])
async def test_successful_reload_with_stale_identity_cannot_score(engine, stale):
    owner, state, _ = engine
    state.reported = stale
    with pytest.raises(RuntimeError, match="Shared Teacher 'B'"):
        await owner.score("B", {"input_ids": [2]})
    assert state.loads == ["B"]
    assert state.scores == []
    assert owner.get_status() == {"state": "failed", "teacher_id": None, "pending": 0}
