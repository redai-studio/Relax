# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import time

import pytest
from conftest import (
    AwaitableValue,
    create_test_manager,
    make_engine_group,
    make_mock_args,
    make_mock_engine,
    make_rollout_server,
)

from relax.distributed.ray.rollout_scale_types import ScaleOutRequest, ScaleOutStatus


@pytest.fixture
def pool():
    args = make_mock_args(colocate=True, fully_async=False, scale_in_drain_timeout=0)
    baseline = make_engine_group(args=args)
    server = make_rollout_server(engine_groups=[baseline])
    manager = create_test_manager(args=args, servers={"default": server})
    manager._colocate_candidates = {}
    manager.rollout_engine_lock = object()
    manager._colocate_scale_outs = set()
    manager._colocate_source = None
    manager._colocate_remote_group = None
    manager._colocate_disposing = False
    manager._health_monitors = []
    return manager, server, baseline


async def start_candidate(manager, request, rank):
    engine = make_mock_engine(weight_version="4")
    engine._actor_id.hex.return_value = f"candidate-{rank}"
    group = make_engine_group(args=manager.args, engines=[engine], is_scaled_out=True, rank_offset=rank)
    task = asyncio.create_task(manager._wait_for_colocate_full(request, group))
    await asyncio.sleep(0)
    return group, engine, task


@pytest.mark.asyncio
async def test_colocate_publishes_only_snapshot_members(pool):
    manager, server, baseline = pool
    request = ScaleOutRequest("scale", ScaleOutStatus.CREATING, num_replicas=2)
    manager._scale_out_requests[request.request_id] = request
    source = object()
    group1, engine1, first = await start_candidate(manager, request, 4)
    assert request.status is ScaleOutStatus.WAITING_FOR_WEIGHT_SYNC
    assert manager.create_scale_out_request(num_replicas=3)["status"] == "CONFLICT"
    assert server.engine_groups == [baseline]
    engine1.register_to_router.remote.assert_not_called()

    snapshot, obsolete = manager.acquire_colocate_full(source)
    assert obsolete is None
    assert [entry[0] for entry in snapshot["receivers"]] == [baseline.engines[0], engine1]
    assert snapshot["shared_count"] == 1
    assert request.status is ScaleOutStatus.WEIGHT_SYNCING
    group2, engine2, second = await start_candidate(manager, request, 5)
    assert request.status is ScaleOutStatus.WEIGHT_SYNCING

    assert await manager.publish_colocate_full(4, ["candidate-4"]) is None
    manager.release_colocate_full()
    assert (await first).success
    assert request.status is ScaleOutStatus.WAITING_FOR_WEIGHT_SYNC
    assert server.engine_groups == [baseline, group1]
    assert not second.done()
    engine1.register_to_router.remote.assert_called_once_with()
    engine1.register_dcs.remote.assert_not_called()
    engine2.register_to_router.remote.assert_not_called()

    assert manager.acquire_colocate_full(source) == (None, snapshot["group_name"])
    manager.acknowledge_colocate_group_removed(snapshot["group_name"])
    next_snapshot, obsolete = manager.acquire_colocate_full(source)
    assert obsolete is None
    assert next_snapshot["group_name"] != snapshot["group_name"]
    assert await manager.publish_colocate_full(4, ["candidate-4", "candidate-5"]) is None
    manager.release_colocate_full()
    assert (await second).success
    assert server.engine_groups == [baseline, group1, group2]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["expired", "evicted", "router_rejected"])
async def test_claimed_candidate_cannot_rollback_before_old_group_is_removed(pool, failure):
    manager, server, baseline = pool
    request = ScaleOutRequest("scale", ScaleOutStatus.CREATING, num_replicas=1)
    group, engine, waiting = await start_candidate(manager, request, 4)
    snapshot, _ = manager.acquire_colocate_full(object())
    if failure == "expired":
        manager._colocate_candidates[4].deadline = time.monotonic() - 1
    elif failure == "evicted":
        group.pg = (object(), [0, 1], [4, 5])
        manager._handle_evictions([("default", group, 0)])
    else:
        engine.register_to_router.remote.return_value = AwaitableValue(False)

    name = await manager.publish_colocate_full(4, ["candidate-4"])
    assert name == snapshot["group_name"]
    assert server.engine_groups == [baseline]
    assert not waiting.done()
    if failure == "router_rejected":
        engine.unregister_from_router.remote.assert_called_once()
    else:
        engine.register_to_router.remote.assert_not_called()
    with pytest.raises(RuntimeError, match="before disconnecting"):
        manager.release_colocate_full()
    manager.acknowledge_colocate_group_removed(name)
    manager.release_colocate_full()
    assert not (await waiting).success


@pytest.mark.asyncio
async def test_unclaimed_timeout_cannot_join_later_full(pool):
    manager, server, baseline = pool
    request = ScaleOutRequest("scale", ScaleOutStatus.CREATING, num_replicas=1)
    _, engine, waiting = await start_candidate(manager, request, 4)
    manager._colocate_candidates[4].deadline = time.monotonic() - 1
    snapshot, _ = manager.acquire_colocate_full(object())
    assert snapshot["group_name"] is None
    assert [entry[0] for entry in snapshot["receivers"]] == baseline.engines
    assert not (await waiting).success
    assert await manager.publish_colocate_full(1, []) is None
    manager.release_colocate_full()
    assert server.engine_groups == [baseline]
    engine.register_to_router.remote.assert_not_called()
