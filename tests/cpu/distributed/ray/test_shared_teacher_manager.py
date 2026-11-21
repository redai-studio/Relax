# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from aiohttp import web


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
        state = SimpleNamespace(current=None, commands=[], scores=[], inflight=set(), stale=False)

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
async def test_offload_drains_and_onload_restores_or_reprepares_by_actor_identity(pool):
    owner, physical, engines, identities, released, arrived = pool
    a = submit(owner, "A", 10)
    b = submit(owner, "B", 20)
    await asyncio.wait_for(asyncio.gather(arrived[10].wait(), arrived[20].wait()), 3)
    offload = asyncio.create_task(owner.offload())
    await asyncio.sleep(0)
    physical.offload.remote.assert_not_awaited()
    with pytest.raises(RuntimeError, match="draining"):
        await owner.score("B", {"input_ids": [21]})
    released[10].set()
    released[20].set()
    await asyncio.gather(a, b, offload)
    physical.offload.remote.assert_awaited_once()
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
async def test_lost_score_connection_stops_switching_before_reusing_uncertain_engine(pool):
    from aiohttp import ClientConnectionError

    owner, physical, engines, _, released, _ = pool
    with pytest.raises(ClientConnectionError):
        await owner.score("A", {"input_ids": [-1]})
    released[4].set()
    with pytest.raises(RuntimeError, match="failed"):
        await owner.score("C", {"input_ids": [4]})
    assert all(engine.commands == [("prepare", "a")] for engine in engines.values())
    physical.offload.remote.assert_not_awaited()
