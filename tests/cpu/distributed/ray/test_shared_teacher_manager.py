# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from aiohttp import ClientResponseError, web


async def _start_replica():
    # Use real HTTP so each replica has its own checkpoint and in-flight requests.
    state = SimpleNamespace(
        current="/A",
        loads=[],
        scores=[],
        started=asyncio.Event(),
        release=asyncio.Event(),
        score_finished=asyncio.Event(),
        reload_started=asyncio.Event(),
        reload_finished=asyncio.Event(),
        allow_reload=asyncio.Event(),
        fail_update=False,
    )
    state.allow_reload.set()

    async def model_info(request):
        return web.json_response({"model_path": state.current})

    async def reload(request):
        body = await request.json()
        assert body["flush_cache"] is True
        state.loads.append(body["weight_version"])
        state.reload_started.set()
        await state.allow_reload.wait()
        if not state.fail_update:
            state.current = body["model_path"]
        state.reload_finished.set()
        return web.json_response(
            {"success": not state.fail_update, "message": "injected reload failure" if state.fail_update else ""},
            status=400 if state.fail_update else 200,
        )

    async def generate(request):
        token = (await request.json())["input_ids"][0]
        state.scores.append((state.current, token))
        if token == 1:
            state.started.set()
            await state.release.wait()
        state.score_finished.set()
        return web.json_response({"scored_by": state.current, "input": token})

    app = web.Application()
    app.add_routes([web.get("/model_info", model_info), web.post("/update_weights_from_disk", reload)])
    app.router.add_post("/generate", generate)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, state


@pytest_asyncio.fixture
async def engine(request):
    from relax.distributed.ray.teacher_manager import SharedTeacherManager

    replicas = [await _start_replica() for _ in range(getattr(request, "param", 1))]
    states = [state for _, state in replicas]
    urls = [f"http://127.0.0.1:{runner.addresses[0][1]}/generate" for runner, _ in replicas]
    physical = MagicMock()
    physical.offload.remote = AsyncMock()
    physical.onload.remote = AsyncMock()
    physical.shutdown.remote = AsyncMock()
    args = SimpleNamespace(opd_teacher_timeout_s=10, opd_teacher_connector_limit=5)
    cls = SharedTeacherManager.__ray_metadata__.modified_class
    owner = cls(args, physical, urls, {"A": "/A", "B": "/B"})
    try:
        await owner.initialize()
        yield owner, states, physical
    finally:
        for state in states:
            state.release.set()
            state.allow_reload.set()
        if owner._worker is not None:
            await owner._worker
        await owner._session.close()
        for runner, _ in replicas:
            await runner.cleanup()


@pytest.mark.asyncio
async def test_grouped_scores_use_the_right_checkpoint_after_active_request_finishes(engine):
    owner, (state,), _ = engine
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
@pytest.mark.parametrize("engine", [2], indirect=True)
async def test_two_replicas_finish_scoring_and_switch_together(engine):
    owner, (first, slow), _ = engine
    slow.allow_reload.clear()

    a1 = asyncio.create_task(owner.score("A", {"input_ids": [2]}))
    a2 = asyncio.create_task(owner.score("A", {"input_ids": [1]}))
    await asyncio.wait_for(asyncio.gather(first.score_finished.wait(), slow.started.wait()), 5)
    b1 = asyncio.create_task(owner.score("B", {"input_ids": [3]}))
    await asyncio.sleep(0)
    assert not first.loads and not slow.loads  # The second replica is still scoring A.

    slow.release.set()
    assert await asyncio.wait_for(asyncio.gather(a1, a2), 5) == [
        {"scored_by": "/A", "input": 2},
        {"scored_by": "/A", "input": 1},
    ]
    await asyncio.wait_for(asyncio.gather(first.reload_finished.wait(), slow.reload_started.wait()), 5)
    assert (first.current, slow.current) == ("/B", "/A")
    # The fast HTTP reply can reach the owner, but it must not release B's batch.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(b1), 0.1)
    assert owner.get_status()["teacher_id"] == "A"
    assert first.scores == [("/A", 2)] and slow.scores == [("/A", 1)]

    slow.allow_reload.set()
    assert await asyncio.wait_for(b1, 5) == {"scored_by": "/B", "input": 3}
    # The next singleton batch starts on the next replica.
    assert await owner.score("B", {"input_ids": [4]}) == {"scored_by": "/B", "input": 4}
    assert first.scores == [("/A", 2), ("/B", 3)]
    assert slow.scores == [("/A", 1), ("/B", 4)]
    assert (await owner.score("A", {"input_ids": [5]}))["scored_by"] == "/A"
    assert first.loads == slow.loads == ["B", "A"]


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", [2], indirect=True)
async def test_training_waits_for_scores_and_restores_last_teacher_on_resume(engine):
    owner, (state, second), physical = engine
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
    second.current = "/A"  # A restarted physical engine comes up with its initial checkpoint.
    await owner.onload()
    await owner.onload()
    assert state.loads == second.loads == ["B", "B"]
    assert state.current == second.current == "/B"
    assert (await owner.score("B", {"input_ids": [4]}))["scored_by"] == "/B"
    physical.onload.remote.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", [2], indirect=True)
async def test_failed_reload_rejects_waiters_and_never_hands_memory_to_training(engine):
    owner, (failing, slow), physical = engine
    owner._limit = 1  # The waiting A request has not yet entered a Teacher queue.
    failing.fail_update = True
    slow.allow_reload.clear()
    b = asyncio.create_task(owner.score("B", {"input_ids": [2]}))
    await asyncio.wait_for(asyncio.gather(failing.reload_finished.wait(), slow.reload_started.wait()), 5)
    a = asyncio.create_task(owner.score("A", {"input_ids": [3]}))
    # An early failure must not resolve the callers before the slow update returns.
    done, _ = await asyncio.wait([b, a], timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
    assert not done
    slow.allow_reload.set()
    results = await asyncio.wait_for(asyncio.gather(b, a, return_exceptions=True), 5)
    assert isinstance(results[0], ClientResponseError) and results[0].status == 400
    assert isinstance(results[1], RuntimeError)
    assert (failing.current, slow.current) == ("/A", "/B")
    assert not failing.scores and not slow.scores
    assert owner.get_status() == {"state": "failed", "teacher_id": None, "pending": 0}
    with pytest.raises(RuntimeError, match="failed"):
        await owner.score("A", {"input_ids": [4]})
    with pytest.raises(RuntimeError, match="failed"):
        await owner.offload()
    physical.offload.remote.assert_not_called()


@pytest.mark.asyncio
async def test_current_teacher_preference_expires_between_token_bounded_batches(engine, monkeypatch):
    from relax.distributed.ray import teacher_manager

    owner, (state,), _ = engine
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
    owner, (state,), physical = engine
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
    with pytest.raises(ClientResponseError if failure == "checkpoint" else RuntimeError):
        await transition()
    assert owner.get_status() == {"state": "failed", "teacher_id": None, "pending": 0}
    with pytest.raises(RuntimeError, match="failed"):
        await owner.score("B", {"input_ids": [4]})


@pytest.mark.asyncio
async def test_capacity_waiter_replaces_cancelled_queued_score(engine):
    owner, (state,), _ = engine
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
    owner, (state,), _ = engine
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
@pytest.mark.parametrize("engine", [2], indirect=True)
async def test_initialize_rejects_wrong_initial_checkpoint(engine):
    owner, (_, second), physical = engine
    second.current = "/B"
    other = type(owner)(SimpleNamespace(opd_teacher_timeout_s=10), physical, owner.urls, {"A": "/A"})
    with pytest.raises(RuntimeError, match="expected checkpoint '/A', got '/B'"):
        await other.initialize()
