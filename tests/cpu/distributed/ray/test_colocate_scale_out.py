# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest
from conftest import (
    AwaitableValue,
    create_test_manager,
    make_engine_group,
    make_mock_args,
    make_mock_engine,
    make_rollout_server,
)
from ray.exceptions import RaySystemError

from relax.distributed.ray.rollout_scale_types import ScaleOutRequest, ScaleOutStatus


@pytest.fixture
def pool():
    args = make_mock_args(colocate=True, fully_async=False, scale_in_drain_timeout=0, scale_weight_sync_precheck=False)
    baseline = make_engine_group(args=args)
    server = make_rollout_server(engine_groups=[baseline])
    manager = create_test_manager(args=args, servers={"default": server})
    manager._colocate_candidates = {}
    manager.rollout_engine_lock = object()
    manager._colocate_scale_outs = set()
    manager._colocate_source = None
    manager._colocate_remote_group = None
    manager._colocate_retiring = None
    manager._colocate_disposing = False
    manager._colocate_failure = None
    manager._colocate_shutdown_pgs = set()
    manager._colocate_pending_pgs = {}
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
@pytest.mark.parametrize("second_arrives_during_full", [True, False], ids=["during-full", "after-release"])
async def test_colocate_publishes_only_snapshot_members(pool, second_arrives_during_full):
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
    if second_arrives_during_full:
        group2, engine2, second = await start_candidate(manager, request, 5)
        assert request.status is ScaleOutStatus.WEIGHT_SYNCING

    assert await manager.publish_colocate_full(4, ["candidate-4"]) is None
    manager.release_colocate_full()
    if not second_arrives_during_full:
        group2, engine2, second = await start_candidate(manager, request, 5)
    assert request.status is ScaleOutStatus.WAITING_FOR_WEIGHT_SYNC
    assert (await first).success
    assert server.engine_groups == [baseline, group1]
    assert not second.done()
    engine1.register_to_router.remote.assert_called_once_with(confirm=True)
    engine1.register_dcs.remote.assert_not_called()
    engine2.register_to_router.remote.assert_not_called()

    assert manager.acquire_colocate_full(source) == (None, snapshot["group_name"])
    manager.acknowledge_colocate_group_removed(snapshot["group_name"])
    next_snapshot, obsolete = manager.acquire_colocate_full(source)
    assert obsolete is None
    assert next_snapshot["group_name"] != snapshot["group_name"]
    assert request.status is ScaleOutStatus.WEIGHT_SYNCING
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
    assert not manager.training_weight_updating
    assert server.engine_groups == [baseline]
    engine.register_to_router.remote.assert_not_called()


@pytest.mark.asyncio
async def test_failed_nccl_precheck_keeps_the_current_full_group(pool):
    manager, server, baseline = pool
    source = object()
    manager.args.scale_weight_sync_precheck = True
    first = ScaleOutRequest("first", ScaleOutStatus.CREATING, num_replicas=1)
    active, active_engine, active_waiter = await start_candidate(manager, first, 4)
    assert manager.acquire_colocate_full(source)[0]["precheck"][0] == 4
    current, _ = manager.acquire_colocate_full(source, (4, True, True, ()))
    assert first.status is ScaleOutStatus.WEIGHT_SYNCING
    assert await manager.publish_colocate_full(4, ["candidate-4"]) is None
    manager.release_colocate_full()
    assert (await active_waiter).success

    second = ScaleOutRequest("second", ScaleOutStatus.CREATING, num_replicas=1)
    _, rejected_engine, rejected_waiter = await start_candidate(manager, second, 5)
    snapshot, obsolete = manager.acquire_colocate_full(source)
    assert obsolete is None
    assert snapshot["precheck"][:3] == (5, [rejected_engine], 2)
    assert second.status is ScaleOutStatus.WAITING_FOR_WEIGHT_SYNC
    assert not manager.training_weight_updating

    snapshot, obsolete = manager.acquire_colocate_full(source, (5, False, True, ()))
    assert obsolete is None
    assert snapshot["group_name"] == current["group_name"]
    assert [engine for engine, _, _ in snapshot["receivers"]] == [baseline.engines[0], active_engine]
    assert await manager.publish_colocate_full(4, ["candidate-4"]) is None
    manager.release_colocate_full()
    result = await rejected_waiter
    assert not result.success and result.reason.category.name == "NCCL_PRECHECK_FAIL"
    rejected_engine.register_to_router.remote.assert_not_called()
    assert server.engine_groups == [baseline, active]


@pytest.mark.parametrize("acknowledged", [True, False])
@pytest.mark.asyncio
async def test_late_receiver_probe_ack_does_not_interrupt_baseline_full(
    pool, patch_ray_get, monkeypatch, acknowledged
):
    from relax.distributed.ray import rollout

    manager, _, baseline = pool
    manager._eval_judge = None
    manager.args.scale_weight_sync_precheck = True
    request = ScaleOutRequest("scale", ScaleOutStatus.CREATING, num_replicas=1)
    candidate, engine, waiter = await start_candidate(manager, request, 4)
    candidate.pg = (SimpleNamespace(id="candidate-pg"), [0, 1], [4, 5])
    source = object()
    assert manager.acquire_colocate_full(source)[0]["precheck"][0] == 4
    probe = asyncio.get_running_loop().create_future()
    snapshot, obsolete = manager.acquire_colocate_full(source, (4, False, True, (probe,)))
    assert obsolete is None and snapshot["group_name"] is None
    assert [engine for engine, _, _ in snapshot["receivers"]] == baseline.engines
    assert not waiter.done()
    engine.register_to_router.remote.assert_not_called()

    if acknowledged:
        probe.set_result({"success": True, "stopped": True})
        result = await asyncio.wait_for(waiter, timeout=2)
        assert not result.success and result.reason.category.name == "NCCL_PRECHECK_FAIL"
    else:
        remove_pg = Mock()
        monkeypatch.setattr(rollout.ray.util, "remove_placement_group", remove_pg)
        try:
            with pytest.raises(RuntimeError, match="PGs were retained"):
                manager.dispose()
            remove_pg.assert_not_called()
            baseline.engines[0].shutdown.remote.assert_called_once()
        finally:
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter


def test_pure_baseline_full_does_not_fence_shutdown(pool, patch_ray_get):
    manager, _, baseline = pool
    manager._eval_judge = None

    snapshot, obsolete = manager.acquire_colocate_full(object())
    assert obsolete is None
    assert snapshot["group_name"] is None
    assert not manager.training_weight_updating

    # Simulate the actor unwinding after a failed IPC update, with no publication.
    manager.dispose()
    baseline.engines[0].shutdown.remote.assert_called_once()


@pytest.mark.asyncio
async def test_dispose_shuts_down_published_and_pending_replicas(pool, patch_ray_get, monkeypatch):
    from relax.distributed.ray import rollout

    manager, server, baseline = pool
    manager._eval_judge = None
    request = ScaleOutRequest("scale", ScaleOutStatus.CREATING, num_replicas=1)
    published = make_engine_group(args=manager.args, is_scaled_out=True, rank_offset=2)
    pending, pending_engine, waiting = await start_candidate(manager, request, 4)
    published.pg = (SimpleNamespace(id="published"), [0, 1], [2, 3])
    pending.pg = (SimpleNamespace(id="pending"), [0, 1], [4, 5])
    server.engine_groups.append(published)
    remove_pg = Mock()
    monkeypatch.setattr(rollout.ray.util, "remove_placement_group", remove_pg)

    manager.dispose()

    for engine in (baseline.engines[0], published.engines[0], pending_engine):
        engine.shutdown.remote.assert_called_once()
    remove_pg.assert_has_calls([call(published.pg[0]), call(pending.pg[0])], any_order=True)
    assert remove_pg.call_count == 2
    assert not (await waiting).success


@pytest.mark.parametrize("unconfirmed", [None, "receiver", "sender"])
@pytest.mark.asyncio
async def test_failed_full_blocks_publication_and_disposes_only_after_exit(pool, monkeypatch, unconfirmed):
    from relax.distributed.ray import rollout

    manager, _, baseline = pool
    manager._eval_judge = None
    request = ScaleOutRequest("scale", ScaleOutStatus.CREATING, num_replicas=1)
    pending, candidate, waiter = await start_candidate(manager, request, 4)
    pending.pg = (SimpleNamespace(id="candidate-pg"), [0, 1], [4, 5])
    waiting_pg = SimpleNamespace(id="unready-pg")
    manager._colocate_pending_pgs[waiting_pg.id] = waiting_pg
    source = Mock()
    snapshot, _ = manager.acquire_colocate_full(source)
    manager.fail_colocate_full(snapshot["group_name"])
    with pytest.raises(RuntimeError, match="ownership is unavailable"):
        manager.acquire_colocate_full(source)
    with pytest.raises(RuntimeError, match="No colocate FULL"):
        await manager.publish_colocate_full(4, ["candidate-4"])

    candidate_ref, sender_ref = object(), object()
    candidate.shutdown.options.return_value.remote.return_value = candidate_ref
    source.abort_colocate_weight_group.remote.return_value = sender_ref
    remove = Mock()
    monkeypatch.setattr(rollout.ray.util, "remove_placement_group", remove)
    monkeypatch.setattr(rollout.ray, "kill", Mock())

    def get(ref, **kwargs):
        if ref is candidate_ref and unconfirmed == "receiver":
            raise RuntimeError("candidate not stopped")
        if ref is sender_ref and unconfirmed == "sender":
            raise RuntimeError("sender abort not confirmed")

    monkeypatch.setattr(rollout.ray, "get", get)
    try:
        if unconfirmed is None:
            manager.dispose()
        else:
            with pytest.raises(RuntimeError, match="Could not confirm"):
                manager.dispose()
        source.abort_colocate_weight_group.remote.assert_called_once_with(snapshot["group_name"])
        if unconfirmed != "receiver":
            assert remove.call_args_list == [call(waiting_pg), call(pending.pg[0])]
            assert (await waiter).reason.category.name == "WEIGHT_SYNC_FAILED"
        else:
            remove.assert_called_once_with(waiting_pg)
        baseline.engines[0].shutdown.remote.assert_called_once()
        candidate.shutdown.options.assert_called_once_with(concurrency_group="colocate_shutdown")
        candidate.register_to_router.remote.assert_not_called()
    finally:
        if not waiter.done():
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter


@pytest.mark.asyncio
async def test_dispose_stops_colocate_receivers_when_group_retirement_fails(pool, monkeypatch):
    from relax.distributed.ray import rollout

    manager, _, baseline = pool
    manager._eval_judge = None
    request = ScaleOutRequest("scale", ScaleOutStatus.CREATING, num_replicas=1)
    group, candidate, waiter = await start_candidate(manager, request, 4)
    group.pg = (SimpleNamespace(id="candidate-pg"), [0, 1], [4, 5])
    source = Mock()
    snapshot, _ = manager.acquire_colocate_full(source)
    await manager.publish_colocate_full(4, ["candidate-4"])
    manager.release_colocate_full()
    assert (await waiter).success
    manager.fail_colocate_full("stale")
    assert manager._colocate_failure is None
    retire = source.disconnect_colocate_weight_group.remote.return_value

    def get(ref, **kwargs):
        if ref is retire:
            raise RuntimeError("receiver ACK failed")

    monkeypatch.setattr(rollout.ray, "get", get)
    remove_pg = Mock()
    monkeypatch.setattr(rollout.ray.util, "remove_placement_group", remove_pg)
    manager.dispose()

    source.abort_colocate_weight_group.remote.assert_called_once_with(snapshot["group_name"])
    baseline.engines[0].shutdown.remote.assert_called_once()
    candidate.shutdown.options.assert_called_once_with(concurrency_group="colocate_shutdown")
    remove_pg.assert_called_once_with(group.pg[0])


@pytest.mark.asyncio
async def test_dispose_reports_unconfirmed_sender_probe(pool, patch_ray_get, monkeypatch):
    from relax.distributed.ray import rollout

    manager, _, baseline = pool
    manager._eval_judge = None
    manager.args.scale_weight_sync_precheck = True
    request = ScaleOutRequest("scale", ScaleOutStatus.CREATING, num_replicas=1)
    candidate, _, waiter = await start_candidate(manager, request, 4)
    candidate.pg = (SimpleNamespace(id="candidate-pg"), [0, 1], [4, 5])
    assert manager.acquire_colocate_full(object())[0]["precheck"][0] == 4
    with pytest.raises(RuntimeError, match="sender probe"):
        manager.acquire_colocate_full(object(), (4, False, False, ()))
    remove_pg = Mock()
    monkeypatch.setattr(rollout.ray.util, "remove_placement_group", remove_pg)
    with pytest.raises(RuntimeError, match="sender probe"):
        manager.dispose()

    baseline.engines[0].shutdown.remote.assert_called_once()
    remove_pg.assert_called_once_with(candidate.pg[0])
    assert not (await waiter).success


@pytest.mark.asyncio
async def test_failed_full_still_releases_unready_pg_in_same_request(pool, monkeypatch):
    from relax.distributed.ray import rollout

    manager, _, _ = pool
    request_id = manager.create_scale_out_request(num_replicas=3)["request_id"]
    ready = asyncio.get_running_loop().create_future()
    ready.set_result(None)
    pending = asyncio.get_running_loop().create_future()
    claimed_pg = SimpleNamespace(id="claimed", ready=lambda: ready)
    waiting_pg = SimpleNamespace(id="waiting", ready=lambda: pending)
    monkeypatch.setattr(rollout.ray.util, "placement_group", Mock(side_effect=[claimed_pg, waiting_pg]))
    monkeypatch.setattr(rollout.asyncio, "wait", AsyncMock(return_value=({ready}, {pending})))
    remove_pg = Mock()
    monkeypatch.setattr(rollout.ray.util, "remove_placement_group", remove_pg)

    async def finish_failed_candidate(*, request, pg, **kwargs):
        assert pg is claimed_pg
        manager._colocate_disposing = True
        manager._colocate_failure = "mixed FULL"
        manager._colocate_shutdown_pgs.add(pg.id)
        return rollout.ScaleResult(False)

    monkeypatch.setattr(manager, "_bring_up_single_replica", finish_failed_candidate)
    try:
        await manager.execute_scale_out(request_id)
        remove_pg.assert_called_once_with(waiting_pg)
        assert manager._scale_out_requests[request_id].status is ScaleOutStatus.FAILED
    finally:
        pending.cancel()


@pytest.mark.parametrize(
    "overrides", [{"pp_size": 2}, {"base_gpu_id": 0}, {"host": "127.0.0.1"}, {"port": 32000}, {"nccl_port": 32001}]
)
def test_incompatible_engine_override_is_rejected_before_scale_request(pool, overrides):
    manager, _, baseline = pool
    baseline.sglang_overrides.update(overrides)

    with pytest.raises(ValueError, match="SGLang PP=DP=1"):
        manager.create_scale_out_request(num_replicas=2)

    assert not manager._scale_out_requests


def test_colocate_fault_tolerance_rejects_scale_out_before_creating_request(pool):
    manager, _, _ = pool
    manager.args.use_fault_tolerance = True

    with pytest.raises(ValueError, match="--use-fault-tolerance"):
        manager.create_scale_out_request(num_replicas=2)

    assert not manager._scale_out_requests


def test_rollout_only_colocate_rejects_scale_out_before_creating_request(pool):
    manager, _, _ = pool
    manager.args.debug_rollout_only = True

    with pytest.raises(ValueError, match="requires a training Actor"):
        manager.create_scale_out_request(num_replicas=2)

    assert not manager._scale_out_requests


@pytest.mark.asyncio
@pytest.mark.parametrize("delete_failures", [0, 2])
async def test_cancelled_colocate_scale_out_keeps_conflict_until_pg_removal_is_confirmed(
    pool, monkeypatch, delete_failures
):
    from relax.distributed.ray import rollout

    manager, _, _ = pool
    request_id = manager.create_scale_out_request(num_replicas=2)["request_id"]
    pg = SimpleNamespace(id="colocate-pg")

    def cancel_after_allocating_pg(*args, **kwargs):
        assert manager.cancel_scale_out(request_id)["status"] == "CANCELLED"
        return pg

    def query_pg(group):
        assert group is pg
        assert manager.create_scale_out_request(num_replicas=2)["status"] == "CONFLICT"
        if query_pg_mock.call_count <= 2:
            raise RaySystemError("GCS query failed")
        return {"state": "REMOVED"}

    def remove_pg(group):
        assert group is pg
        if delete_pg.call_count <= delete_failures:
            raise RaySystemError("GCS delete failed")

    query_pg_mock = Mock(side_effect=query_pg)
    delete_pg = Mock(side_effect=remove_pg)
    monkeypatch.setattr(rollout.ray.util, "placement_group", cancel_after_allocating_pg)
    monkeypatch.setattr(rollout.ray.util, "remove_placement_group", delete_pg)
    monkeypatch.setattr(rollout.ray.util, "placement_group_table", query_pg_mock)
    monkeypatch.setattr(rollout.asyncio, "sleep", AsyncMock())

    await manager.execute_scale_out(request_id)

    assert delete_pg.call_count == delete_failures + 1
    assert query_pg_mock.call_count == 3
    assert manager._scale_out_requests[request_id].status is ScaleOutStatus.CANCELLED
    assert manager.create_scale_out_request(num_replicas=2)["status"] == "PENDING"


@pytest.mark.asyncio
async def test_noncontiguous_pg_cannot_start_an_engine(pool, monkeypatch):
    from relax.distributed.ray import rollout

    manager, server, _ = pool
    request = ScaleOutRequest("scale", ScaleOutStatus.CREATING, num_replicas=1)
    probe = Mock()
    probe.get_ip_and_gpu_id.remote.side_effect = [AwaitableValue(("10.0.0.1", 4)), AwaitableValue(("10.0.0.1", 6))]
    probe_class = Mock()
    probe_class.options.return_value.remote.return_value = probe
    create_group = Mock()
    monkeypatch.setattr(rollout, "EngineGroup", create_group)
    monkeypatch.setattr(rollout.ray, "kill", Mock())

    result = await manager._bring_up_single_replica(
        request, server, object(), 0, 2, 2, 4, lambda bundle: bundle[2], probe_class
    )

    assert not result.success
    create_group.assert_not_called()
