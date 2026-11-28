# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import pickle
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from aiohttp import ClientConnectionError, ClientResponseError, web


@pytest_asyncio.fixture
async def pool(monkeypatch, unused_tcp_port_factory):
    from sglang.srt.utils import MultiprocessingSerializer

    from relax.distributed.ray import teacher_manager

    package_dir = "/packages/run:1"
    monkeypatch.setattr(teacher_manager.ray, "kill", lambda actor: None)
    physical = MagicMock()
    identities = {}
    physical.engine_identities.remote = AsyncMock(side_effect=lambda: dict(identities))
    for method in ("offload", "onload", "shutdown"):
        getattr(physical, method).remote = AsyncMock()

    released = defaultdict(asyncio.Event)
    arrived = defaultdict(asyncio.Event)
    endpoints = {}
    runners = []

    async def start_engine():
        switch_started, allow_switch = asyncio.Event(), asyncio.Event()
        allow_switch.set()
        state = SimpleNamespace(
            current=None,
            commands=[],
            scores=[],
            inflight=set(),
            stale=False,
            switch_started=switch_started,
            allow_switch=allow_switch,
        )

        async def update(request):
            body = await request.json()
            assert not state.inflight, "changed the weights under an active score"
            version = body["weight_version"]
            rank0, rank1 = body["serialized_named_tensors"]
            assert rank0 == rank1
            ((marker, _),) = MultiprocessingSerializer.deserialize(rank0)
            _, operation, target = marker.split(":", 2)
            assert target == (package_dir if operation == "prepare" else version)
            state.commands.append((operation, version))
            if operation == "switch":
                switch_started.set()
                await allow_switch.wait()
            if not state.stale:
                state.current = version
            return web.json_response({"success": True})

        async def version(request):
            return web.json_response({"weight_version": state.current})

        async def score(request):
            token = (await request.json())["input_ids"][0]
            state.scores.append(token)
            state.inflight.add(token)
            arrived[token].set()
            if token == -1:
                # Simulate a dropped connection while the engine still owns the request.
                request.transport.close()
                return web.Response()
            await released[token].wait()
            state.inflight.remove(token)
            if token <= -400:
                return web.Response(status=-token, text="rejected score")
            return web.json_response({"teacher": state.current, "token": token})

        app = web.Application()
        app.add_routes(
            [
                web.post("/update_weights_from_tensor", update),
                web.get("/get_weight_version", version),
                web.post("/generate", score),
            ]
        )
        runner = web.AppRunner(app)
        await runner.setup()
        port = unused_tcp_port_factory()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        url = f"http://127.0.0.1:{port}/generate"
        identities[url] = str(port)
        endpoints[url] = state
        runners.append(runner)

    await start_engine()
    await start_engine()
    cls = teacher_manager.SharedTeacherManager.__ray_metadata__.modified_class
    owner = cls(
        SimpleNamespace(opd_teacher_timeout_s=5, opd_teacher_delta_dir=package_dir),
        physical,
        list(endpoints),
        initial_teacher="A",
        teacher_to_adts_id={"A": "a", "B": "b", "B-alias": "b", "C": "c"},
        tp_size=2,
    )
    try:
        await owner.initialize()
        yield owner, physical, endpoints, identities, released, arrived
    finally:
        for engine in endpoints.values():
            engine.allow_switch.set()
        for gate in released.values():
            gate.set()
        await owner.shutdown()
        for runner in runners:
            await runner.cleanup()


def submit(owner, teacher, token):
    return asyncio.create_task(owner.score(teacher, {"input_ids": [token]}))


@pytest.mark.asyncio
async def test_independent_engine_drains_and_switches_while_peer_keeps_scoring(pool):
    owner, _, engines, _, released, arrived = pool
    a = submit(owner, "A", 10)
    b = submit(owner, "B", 20)
    await asyncio.wait_for(asyncio.gather(arrived[10].wait(), arrived[20].wait()), 3)
    c = submit(owner, "C", 30)
    released[20].set()
    assert (await b)["teacher"] == "b"
    await asyncio.wait_for(arrived[30].wait(), 3)
    first, second = engines.values()
    assert first.inflight == {10} and second.inflight == {30}
    assert second.commands == [("prepare", "a"), ("switch", "b"), ("switch", "c")]
    released[10].set()
    released[30].set()
    assert [r["teacher"] for r in await asyncio.gather(a, c)] == ["a", "c"]


@pytest.mark.asyncio
async def test_cancelled_client_still_drains_before_engine_can_switch(pool):
    owner, _, engines, _, released, arrived = pool
    a = submit(owner, "A", 10)
    b = submit(owner, "B", 20)
    await asyncio.wait_for(asyncio.gather(arrived[10].wait(), arrived[20].wait()), 3)
    b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await b

    c = submit(owner, "C", 30)
    released[11].set()
    assert (await asyncio.wait_for(owner.score("A", {"input_ids": [11]}), 3))["teacher"] == "a"
    second = list(engines.values())[1]
    assert second.inflight == {20}
    assert second.commands == [("prepare", "a"), ("switch", "b")]

    released[20].set()
    released[30].set()
    assert (await asyncio.wait_for(c, 3))["teacher"] == "c"
    assert second.commands[-1] == ("switch", "c")
    released[10].set()
    await a


@pytest.mark.asyncio
async def test_first_free_resident_refills_without_waiting_for_its_peer(pool):
    owner, _, _, _, released, arrived = pool
    owner._token_budget = 1
    slow, fast, next_score = (submit(owner, "A", token) for token in (1, 2, 3))
    await asyncio.wait_for(asyncio.gather(arrived[1].wait(), arrived[2].wait()), 3)
    released[2].set()
    released[3].set()
    for task in (fast, next_score):
        assert (await asyncio.wait_for(task, 3))["teacher"] == "a"
    assert not slow.done()
    released[1].set()
    await slow


@pytest.mark.asyncio
async def test_refill_balances_between_busy_residents(pool):
    owner, _, engines, _, released, arrived = pool
    initial = [submit(owner, "A", token) for token in (1, 2)]
    await asyncio.wait_for(asyncio.gather(arrived[1].wait(), arrived[2].wait()), 3)
    burst = [submit(owner, "A", token) for token in range(3, 11)]
    await asyncio.wait_for(asyncio.gather(*(arrived[token].wait() for token in range(3, 11))), 3)
    assert [len(engine.inflight) for engine in engines.values()] == [5, 5]
    for token in range(1, 11):
        released[token].set()
    await asyncio.gather(*initial, *burst)


@pytest.mark.asyncio
async def test_cold_replicas_share_a_burst_that_needs_two_switches(pool):
    owner, _, engines, _, released, arrived = pool
    owner._token_budget = 16
    burst = [submit(owner, "B", token) for token in range(17)]
    await asyncio.wait_for(asyncio.gather(*(arrived[token].wait() for token in range(17))), 3)
    assert sorted(len(engine.inflight) for engine in engines.values()) == [8, 9]
    for token in range(17):
        released[token].set()
    await asyncio.gather(*burst)


@pytest.mark.asyncio
async def test_resident_refills_until_expanded_tokens_require_borrowing(pool):
    owner, _, engines, _, released, arrived = pool
    owner._token_budget = 4
    a = asyncio.create_task(owner.score("A", {"input_ids": [10]}, token_count=2))
    await asyncio.wait_for(arrived[10].wait(), 3)
    released[20].set()
    await owner.score("B", {"input_ids": [20]})

    resident, spare = engines.values()
    released[11].set()
    await owner.score("A", {"input_ids": [11]})
    assert resident.scores == [10, 11]
    assert spare.current == "b"

    released[12].set()
    await owner.score("A", {"input_ids": [12]}, token_count=3)
    assert spare.scores == [20, 12]
    assert spare.current == "a"
    assert not a.done()
    released[10].set()
    await a


@pytest.mark.asyncio
async def test_cancel_during_switch_preserves_new_weights_and_skips_http_score(pool):
    owner, _, engines, _, released, _ = pool
    first, second = engines.values()
    first.allow_switch.clear()
    request = submit(owner, "B", 20)
    await asyncio.wait_for(first.switch_started.wait(), 3)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    released[20].set()
    released[21].set()
    first.allow_switch.set()
    await asyncio.wait_for(owner._worker, 3)
    assert (await owner.score("B", {"input_ids": [21]}))["teacher"] == "b"
    assert first.scores == [21]
    assert first.commands == [("prepare", "a"), ("switch", "b")]
    assert second.commands == [("prepare", "a")]


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_tokens, borrow", [(1, False), (2, True)])
async def test_queued_scores_use_remaining_budget_of_switching_replica(pool, pending_tokens, borrow):
    owner, _, engines, _, released, arrived = pool
    owner._token_budget = 4
    first, second = engines.values()
    first.allow_switch.clear()
    b = asyncio.create_task(owner.score("B", {"input_ids": [20]}, token_count=3))
    await asyncio.wait_for(first.switch_started.wait(), 3)
    next_score = asyncio.create_task(owner.score("B", {"input_ids": [21]}, token_count=pending_tokens))
    if borrow:
        await asyncio.wait_for(arrived[21].wait(), 3)
        assert second.inflight == {21}
    else:
        # The other engine can still serve A while both B requests await the same weight change.
        released[11].set()
        assert (await asyncio.wait_for(owner.score("A", {"input_ids": [11]}), 3))["teacher"] == "a"
    first.allow_switch.set()
    released[20].set()
    released[21].set()
    await asyncio.gather(b, next_score)
    assert second.scores == ([21] if borrow else [11])
    assert sorted(first.scores) == ([20] if borrow else [20, 21])


@pytest.mark.asyncio
async def test_offload_drains_and_onload_restores_or_reprepares_by_actor_identity(pool):
    owner, physical, engines, identities, released, arrived = pool

    def submit_offload():
        # Ray submits the RPC when .remote() is called, before its result is awaited.
        assert all(not engine.inflight for engine in engines.values()), "offload submitted with active scores"
        return asyncio.sleep(0)

    physical.offload.remote = MagicMock(side_effect=submit_offload)
    a = submit(owner, "A", 10)
    b = submit(owner, "B", 20)
    await asyncio.wait_for(asyncio.gather(arrived[10].wait(), arrived[20].wait()), 3)
    offload = asyncio.create_task(owner.offload())
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="draining"):
        await owner.score("B", {"input_ids": [21]})
    released[10].set()
    released[20].set()
    await asyncio.gather(a, b, offload)
    physical.offload.remote.assert_called_once_with()
    first_url, second_url = engines
    identities[second_url] = "rebuilt-at-same-url"
    await owner.onload()
    assert engines[first_url].commands[-1] == ("restore", "a")
    assert engines[second_url].commands[-2:] == [("prepare", "a"), ("switch", "b")]
    released[21].set()
    assert (await owner.score("B-alias", {"input_ids": [21]}))["teacher"] == "b"
    assert len(engines[second_url].commands) == 4


@pytest.mark.asyncio
async def test_unconfirmed_switch_cannot_score_or_offload_uncertain_weights(pool):
    owner, physical, engines, _, released, _ = pool
    for engine in engines.values():
        engine.stale = True
    released[33].set()
    with pytest.raises(RuntimeError, match="Expected Teacher b"):
        await owner.score("B", {"input_ids": [33]})
    assert all(not engine.scores for engine in engines.values())
    with pytest.raises(RuntimeError, match="failed"):
        await owner.offload()
    physical.offload.remote.assert_not_awaited()


@pytest.mark.asyncio
async def test_rejected_score_fails_only_its_caller_and_pool_remains_usable(pool):
    owner, _, _, _, released, arrived = pool
    b = submit(owner, "B", 20)
    await asyncio.wait_for(arrived[20].wait(), 3)
    released[-400].set()
    with pytest.raises(RuntimeError, match="400") as error:
        await owner.score("A", {"input_ids": [-400]})
    assert str(pickle.loads(pickle.dumps(error.value))) == str(error.value)
    released[20].set()
    assert (await b)["teacher"] == "b"
    released[21].set()
    assert (await owner.score("A", {"input_ids": [21]}))["teacher"] == "a"
    await owner.offload()
    await owner.onload()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token,error", [(-1, ClientConnectionError), (-408, ClientResponseError), (-503, ClientResponseError)]
)
async def test_uncertain_score_failure_stops_switching_before_reusing_engine(pool, token, error):
    owner, physical, engines, _, released, _ = pool
    released[token].set()
    with pytest.raises(error):
        await owner.score("A", {"input_ids": [token]})
    released[4].set()
    with pytest.raises(RuntimeError, match="failed"):
        await owner.score("C", {"input_ids": [4]})
    assert all(engine.commands == [("prepare", "a")] for engine in engines.values())
    physical.offload.remote.assert_not_awaited()
