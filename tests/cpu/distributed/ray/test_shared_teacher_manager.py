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
        version=None,
        reported={},
        tensor_checkpoint=None,
        controls=[],
        score_transports=[],
        failure_status=400,
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
        return web.json_response({"model_path": state.current, "weight_version": state.version, **state.reported})

    async def reload(request):
        body = await request.json()
        assert body["flush_cache"] is True
        state.loads.append(body["weight_version"])
        state.reload_started.set()
        await state.allow_reload.wait()
        if not state.fail_update:
            state.current = body["model_path"]
            state.version = body["weight_version"]
            state.tensor_checkpoint = None
        state.reload_finished.set()
        return web.json_response(
            {"success": not state.fail_update, "message": "injected reload failure" if state.fail_update else ""},
            status=state.failure_status if state.fail_update else 200,
        )

    async def generate(request):
        token = (await request.json())["input_ids"][0]
        active = state.tensor_checkpoint or state.current
        state.scores.append((active, token))
        state.score_transports.append(request.transport)
        if token == 1:
            state.started.set()
            await state.release.wait()
        assert (state.tensor_checkpoint or state.current) == active, "weights changed during an in-flight score"
        state.score_finished.set()
        return web.json_response({"scored_by": active, "input": token})

    async def tensor_update(request):
        from sglang.srt.utils import MultiprocessingSerializer

        body = await request.json()
        markers = body["serialized_named_tensors"]
        assert len(markers) == state.tp_size and len(set(markers)) == 1
        marker = markers[0]
        ((name, tensor),) = MultiprocessingSerializer.deserialize(marker)
        prefix, operation, physical_id = name.split(":")
        assert prefix == "relax_adp" and tensor.numel() == 0
        assert body["flush_cache"] and body["weight_version"] == physical_id
        state.controls.append((operation, physical_id))
        if operation in ("switch", "restore"):
            state.reload_started.set()
            await state.allow_reload.wait()
            state.reload_finished.set()
        if state.fail_update:
            return web.json_response({"success": False, "message": "worker rejected the package"})
        state.version = physical_id
        state.tensor_checkpoint = {"Teacher1": "/A", "Teacher2": "/B"}[physical_id]
        return web.json_response({"success": True})

    app = web.Application()
    app.add_routes([web.get("/model_info", model_info), web.post("/update_weights_from_disk", reload)])
    app.router.add_post("/generate", generate)
    app.router.add_post("/update_weights_from_tensor", tensor_update)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, state


@pytest_asyncio.fixture
async def engine(request):
    from relax.distributed.ray.teacher_manager import SharedTeacherManager

    mode = getattr(request, "param", 1)
    delta = mode == "delta" or isinstance(mode, tuple)
    if isinstance(mode, tuple):
        count, tp = mode
    elif mode == "lockstep":
        count, tp = 2, 1
    else:
        count, tp = (1 if delta else mode), 1
    replicas = [await _start_replica() for _ in range(count)]
    states = [state for _, state in replicas]
    urls = [f"http://127.0.0.1:{runner.addresses[0][1]}/generate" for runner, _ in replicas]
    physical = MagicMock()
    physical.offload.remote = AsyncMock()
    physical.onload.remote = AsyncMock()
    physical.shutdown.remote = AsyncMock()
    args = SimpleNamespace(opd_teacher_timeout_s=10, opd_teacher_connector_limit=5)
    if mode == "lockstep":
        args.opd_teacher_scheduler = "lockstep"
    if delta:
        args.opd_teacher_delta_dir = "/packages"
        args.opd_teacher_delta_ids = {"A": "Teacher1", "B": "Teacher2"}
        args.teacher_num_gpus_per_engine = tp
        args.resource = {"teacher": [1, tp * count]}
        for state in states:
            state.tp_size = tp
        physical.engine_identities.remote = AsyncMock(return_value={url: str(i) for i, url in enumerate(urls)})
    cls = SharedTeacherManager.__ray_metadata__.modified_class
    # The existing TP1 validation tool supplies a scalar URL.
    routes = {"A": "/A", "B": "/B"} if delta else {"A": "/A", "B": "/B", "C": "/C"}
    owner = cls(args, physical, urls[0] if mode == "delta" else urls, routes)
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
async def test_borrowed_replica_drains_only_its_own_inflight_request(engine):
    owner, states, _ = engine
    a = asyncio.create_task(owner.score("A", {"input_ids": [1] * 131072}))
    b = asyncio.create_task(owner.score("B", {"input_ids": [1]}))
    await asyncio.wait_for(asyncio.gather(*(state.started.wait() for state in states)), 5)
    resident, borrowed = sorted(states, key=lambda state: state.current)
    assert (resident.current, borrowed.current) == ("/A", "/B")

    queued = asyncio.create_task(owner.score("A", {"input_ids": [2] * 131072}))
    b.cancel()  # The caller has gone, but B is still running on the borrowed replica.
    with pytest.raises(asyncio.CancelledError):
        await b
    assert borrowed.loads == ["B"]
    borrowed.release.set()
    assert await asyncio.wait_for(queued, 5) == {"scored_by": "/A", "input": 2}
    assert borrowed.scores == [("/B", 1), ("/A", 2)]
    assert borrowed.loads == ["B", "A"]
    assert not resident.release.is_set() and not a.done()
    resident.release.set()
    assert await asyncio.wait_for(a, 5) == {"scored_by": "/A", "input": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", [4], indirect=True)
async def test_spare_replica_follows_tokens_after_covering_each_waiting_teacher(engine):
    owner, states, _ = engine
    owner._limit = 8
    work = [("A", 2), ("B", 2), ("C", 80000), ("C", 80000), ("A", 2), ("B", 2)]
    scores = [asyncio.create_task(owner.score(route, {"input_ids": [1] * length})) for route, length in work]
    await asyncio.wait_for(asyncio.gather(*(state.started.wait() for state in states)), 5)
    # Each route has two requests. C carries 160k tokens, more than one replica-sized batch.
    assert sorted(state.current for state in states) == ["/A", "/B", "/C", "/C"]
    for state in states:
        state.release.set()
    results = await asyncio.wait_for(asyncio.gather(*scores), 5)
    assert [result["scored_by"] for result in results] == [f"/{route}" for route, _ in work]


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", [2], indirect=True)
async def test_offload_drains_all_replicas_and_restores_their_respective_teachers(engine):
    owner, states, physical = engine
    a = asyncio.create_task(owner.score("A", {"input_ids": [1]}))
    b = asyncio.create_task(owner.score("B", {"input_ids": [1]}))
    await asyncio.wait_for(asyncio.gather(*(state.started.wait() for state in states)), 5)
    resident, second = sorted(states, key=lambda state: state.current)
    offload = asyncio.create_task(owner.offload())
    await asyncio.sleep(0)
    assert owner.get_status()["state"] == "draining"
    with pytest.raises(RuntimeError, match="draining"):
        await owner.score("A", {"input_ids": [3]})
    resident.release.set()
    assert (await asyncio.wait_for(a, 5))["scored_by"] == "/A"
    physical.offload.remote.assert_not_called()
    second.release.set()
    await asyncio.wait_for(asyncio.gather(b, offload), 5)
    physical.offload.remote.assert_awaited_once()
    assert owner.get_status()["state"] == "offloaded"

    second.current = "/A"  # A rebuilt engine boots at the initial checkpoint.
    await owner.onload()
    await owner.onload()
    assert (resident.current, second.current) == ("/A", "/B")
    assert (await owner.score("B", {"input_ids": [4]}))["scored_by"] == "/B"
    physical.onload.remote.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", [2], indirect=True)
@pytest.mark.parametrize("failure_status", [200, 400])
async def test_failed_reload_rejects_waiters_and_never_hands_memory_to_training(engine, failure_status):
    owner, states, physical = engine
    owner._limit = 2
    switching = asyncio.Event()
    for state in states:
        state.fail_update = True
        state.failure_status = failure_status
        state.allow_reload.clear()
        state.reload_started = switching
    a = asyncio.create_task(owner.score("A", {"input_ids": [1]}))
    b = asyncio.create_task(owner.score("B", {"input_ids": [2]}))
    await asyncio.wait_for(switching.wait(), 5)
    failing, resident = sorted(states, key=lambda state: not state.loads)
    await asyncio.wait_for(resident.started.wait(), 5)
    waiting = asyncio.create_task(owner.score("A", {"input_ids": [3]}))
    await asyncio.sleep(0)
    assert owner.get_status()["pending"] == 2
    failing.allow_reload.set()
    await asyncio.wait_for(failing.reload_finished.wait(), 5)
    resident.release.set()
    _, failed, rejected = await asyncio.wait_for(asyncio.gather(a, b, waiting, return_exceptions=True), 5)
    if failure_status == 400:
        assert isinstance(failed, ClientResponseError) and failed.status == 400
    else:
        assert isinstance(failed, RuntimeError) and "injected reload failure" in str(failed)
    assert isinstance(rejected, RuntimeError)
    assert resident.scores == [("/A", 1)] and not failing.scores
    assert owner.get_status()["state"] == "failed" and owner.get_status()["teacher_id"] is None
    with pytest.raises(RuntimeError, match="failed"):
        await owner.score("A", {"input_ids": [4]})
    with pytest.raises(RuntimeError, match="failed"):
        await owner.offload()
    physical.offload.remote.assert_not_called()


@pytest.mark.asyncio
async def test_one_replica_serves_aged_routes_and_keeps_per_route_fifo(engine, monkeypatch):
    from relax.distributed.ray import teacher_manager

    owner, (state,), _ = engine
    now = [0.0]
    monkeypatch.setattr(teacher_manager, "monotonic", lambda: now[0])
    a1 = asyncio.create_task(owner.score("A", {"input_ids": [1]}))
    await asyncio.wait_for(state.started.wait(), 5)
    work = [("B", [2] * 131073), ("C", [3]), ("B", [4]), ("A", [5])]
    scores = []
    for route, tokens in work:
        now[0] += 0.1
        scores.append(asyncio.create_task(owner.score(route, {"input_ids": tokens})))
        await asyncio.sleep(0)
    now[0] = 3.0  # The oldest waiting request is on B; the oversized first request must still make progress.
    state.release.set()
    await asyncio.wait_for(asyncio.gather(a1, *scores), 5)
    assert state.scores[:2] == [("/A", 1), ("/B", 2)]
    assert [token for route, token in state.scores if route == "/B"] == [2, 4]
    expected = [("/A", 1)] + [(f"/{route}", ids[0]) for route, ids in work]
    assert sorted(state.scores) == sorted(expected)


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


@pytest.mark.asyncio
async def test_shared_teacher_replaces_idle_http_connection_before_server_timeout(engine):
    owner, (state,), _ = engine
    await owner.score("A", {"input_ids": [2]})
    await owner.score("A", {"input_ids": [3]})
    assert state.score_transports[0] is state.score_transports[1]

    # The server still accepts the idle connection. The client must retire it after 2s.
    await asyncio.sleep(2.1)
    result = await owner.score("A", {"input_ids": [4]})
    assert state.score_transports[2] is not state.score_transports[1]
    assert result["input"] == 4
    assert state.loads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", [2], indirect=True)
@pytest.mark.parametrize("stale", [{"model_path": "/A"}, {"weight_version": "A"}])
async def test_successful_reload_with_stale_identity_cannot_score(engine, stale):
    owner, states, _ = engine
    for state in states:
        state.reported = stale
    with pytest.raises(RuntimeError, match="Shared Teacher 'B'"):
        await owner.score("B", {"input_ids": [2]})
    assert sum(state.loads.count("B") for state in states) == 1
    assert not any(state.scores for state in states)
    status = owner.get_status()
    assert status["state"] == "failed" and status["teacher_id"] is None and status["pending"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["delta"], indirect=True)
async def test_delta_uses_worker_commands_and_restores_after_training(engine):
    owner, (state,), physical = engine
    assert state.controls == [("prepare", "Teacher1"), ("adopt", "Teacher2"), ("restore", "Teacher1")]
    assert state.loads == ["B"]
    assert await owner.score("B", {"input_ids": [2]}) == {"scored_by": "/B", "input": 2}
    await owner.offload()
    await owner.onload()
    assert (await owner.score("A", {"input_ids": [3]}))["scored_by"] == "/A"
    assert state.controls[-3:] == [("switch", "Teacher2"), ("restore", "Teacher2"), ("switch", "Teacher1")]
    assert state.loads == ["B"]
    # A replacement process starts at its original disk checkpoint with no pinned packages.
    await owner.offload()
    state.current, state.version = "/A", None
    physical.engine_identities.remote.return_value = {owner.urls[0]: "replacement"}
    await owner.onload()
    assert state.controls.count(("prepare", "Teacher1")) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["delta"], indirect=True)
async def test_failed_delta_stops_before_issuing_a_score(engine):
    owner, (state,), _ = engine
    state.fail_update = True
    with pytest.raises(RuntimeError, match="worker rejected"):
        await owner.score("B", {"input_ids": [2]})
    assert not state.scores and owner.get_status()["state"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", [(2, 4), (3, 4)], indirect=True)
async def test_delta_pool_restores_assigned_teacher_and_only_reprepares_rebuilt_replica(engine):
    owner, states, physical = engine
    initial = [("prepare", "Teacher1"), ("adopt", "Teacher2"), ("restore", "Teacher1")]
    assert all(state.controls == initial for state in states)
    assert (await owner.score("B", {"input_ids": [2]}))["scored_by"] == "/B"
    switched = [state for state in states if state.controls[-1] == ("switch", "Teacher2")]
    assert len(switched) == 1
    rebuilt = switched[0]
    assert all(state.controls == initial for state in states if state is not rebuilt)

    await owner.offload()
    rebuilt.current, rebuilt.version, rebuilt.tensor_checkpoint = "/A", None, None
    identities = physical.engine_identities.remote.return_value
    physical.engine_identities.remote.return_value = {**identities, owner.urls[states.index(rebuilt)]: "replacement"}
    await owner.onload()
    assert rebuilt.controls[-3:] == [("prepare", "Teacher1"), ("adopt", "Teacher2"), ("restore", "Teacher2")]
    for state in states:
        if state is not rebuilt:
            assert state.controls == [*initial, ("restore", "Teacher1")]
    assert owner.get_status()["state"] == "ready"
    assert (await owner.score("B", {"input_ids": [3]}))["scored_by"] == "/B"


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", [(3, 4)], indirect=True)
async def test_delta_onload_waits_for_peer_writes_before_reporting_partial_failure(engine):
    owner, states, _ = engine
    await owner.offload()
    failed, _, slow = states
    failed.fail_update = True
    failed.reload_finished.clear()
    slow.reload_started.clear()
    slow.allow_reload.clear()
    onload = asyncio.create_task(owner.onload())
    await asyncio.wait_for(asyncio.gather(failed.reload_finished.wait(), slow.reload_started.wait()), 5)
    assert not onload.done()
    with pytest.raises(RuntimeError):
        await owner.score("A", {"input_ids": [2]})
    slow.allow_reload.set()
    with pytest.raises(RuntimeError, match="worker rejected"):
        await asyncio.wait_for(onload, 5)
    with pytest.raises(RuntimeError, match="failed"):
        await owner.score("A", {"input_ids": [3]})
    assert all(not state.scores for state in states)
    assert owner.get_status()["teacher_id"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", [(2, 4)], indirect=True)
async def test_rebuilt_delta_replica_must_be_at_initial_checkpoint(engine):
    owner, (first, replacement), physical = engine
    await owner.offload()
    replacement.current = "/unexpected"
    physical.engine_identities.remote.return_value = {owner.urls[0]: "0", owner.urls[1]: "replacement"}
    with pytest.raises(RuntimeError, match="expected checkpoint"):
        await owner.onload()
    assert owner.get_status()["state"] == "failed"
    assert first.controls[-1] == ("restore", "Teacher1")
    assert replacement.controls.count(("prepare", "Teacher1")) == 1


@pytest.mark.asyncio
async def test_resident_replica_returns_and_refills_while_sibling_request_is_slow(engine):
    owner, (state,), _ = engine
    slow = asyncio.create_task(owner.score("A", {"input_ids": [1]}))
    try:
        await asyncio.wait_for(state.started.wait(), 5)
        for token in (2, 3):
            result = await asyncio.wait_for(owner.score("A", {"input_ids": [token]}), 5)
            assert result == {"scored_by": "/A", "input": token}
        assert not slow.done() and not state.loads
    finally:
        state.release.set()
        await slow


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["lockstep"], indirect=True)
async def test_lockstep_drains_the_pool_before_changing_teacher(engine):
    owner, (first, slow), _ = engine
    slow.allow_reload.clear()
    fast = asyncio.create_task(owner.score("A", {"input_ids": [2]}))
    blocked = asyncio.create_task(owner.score("A", {"input_ids": [1]}))
    await asyncio.wait_for(asyncio.gather(first.score_finished.wait(), slow.started.wait()), 5)
    switch = asyncio.create_task(owner.score("B", {"input_ids": [3]}))
    await fast
    assert not first.loads and not slow.loads
    slow.release.set()
    await blocked
    await asyncio.wait_for(asyncio.gather(first.reload_finished.wait(), slow.reload_started.wait()), 5)
    assert not switch.done() and all(route == "/A" for s in (first, slow) for route, _ in s.scores)
    slow.allow_reload.set()
    assert await asyncio.wait_for(switch, 5) == {"scored_by": "/B", "input": 3}
    assert first.loads == slow.loads == ["B"]
