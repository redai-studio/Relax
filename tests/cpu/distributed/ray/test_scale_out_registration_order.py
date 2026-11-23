# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for scale-out engine publication ordering."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from relax.distributed.ray.inference_manager import InferenceManager
from relax.distributed.ray.inference_pool import RolloutPool
from relax.engine.inference.types import LifecycleState


try:
    from relax.distributed.ray.policy_engine_pool import (
        EngineFinalizeResult,
        ScaleOutRequest,
        ScaleOutStatus,
        _EngineInitRecord,
    )
    from relax.utils.scale_utils import ScaleOutFailure, ScaleOutFailureCategory

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from conftest import (
    AwaitableValue,
    create_test_manager,
    make_engine_group,
    make_mock_args,
    make_mock_engine,
    make_rollout_server,
)


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


async def _sync_success(*args, publish, **kwargs):
    await publish()
    return True


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["published", "retry_rollback", "unconfirmed"])
async def test_scale_out_holds_training_lock_through_router_and_group_publication(patch_ray_get, outcome):
    registered = outcome == "published"
    seed = make_mock_engine(url="http://seed:1", weight_version="7")
    engine = make_mock_engine(url="http://elastic:1")
    seed._actor_id.hex.return_value = "seed"
    engine._actor_id.hex.return_value = "elastic"
    server = make_rollout_server(engine_groups=[make_engine_group(engines=[seed])])
    manager = create_test_manager(servers={"actor": server})
    for running in (seed, engine):
        running._get_local_state.return_value = manager._engine_actor_states.ALIVE
    seed_record = {"url": "http://seed:1", "pid": 1, "node_id": "seed-node"}
    new_record = {"url": "http://elastic:1", "pid": 2, "node_id": "elastic-node"}
    server.engine_groups[0].engine_init_records[0] = _EngineInitRecord(seed_record)
    pool = RolloutPool.__new__(RolloutPool)
    pool.backend, pool.synced_actor_ids = manager, {"seed"}
    pool.requires_weight_confirmation, pool.actor_weight_versions = True, {"seed": "7"}
    owner = InferenceManager.__ray_metadata__.modified_class()
    owner.pool, owner.memory_ready, owner.state = pool, True, LifecycleState.READY
    owner.required_weight_version = "7"
    lock = manager._weight_sync_lock
    held = False

    def acquire():
        nonlocal held
        if held:
            return AwaitableValue(False)
        held = True
        return AwaitableValue(True)

    def release():
        nonlocal held
        assert held
        assert (len(server.engine_groups) == 2) is registered
        if outcome == "retry_rollback":
            engine.shutdown.remote.assert_called_once()
        held = False
        return AwaitableValue(None)

    lock.acquire.remote.side_effect = acquire
    lock.release.remote.side_effect = release

    def register_router():
        assert not acquire().value  # training tries to advance the seed here
        return AwaitableValue(registered)

    engine.register_to_router.remote.side_effect = register_router
    if not registered:
        engine.unregister_from_router.remote.side_effect = [
            AwaitableValue(False),
            AwaitableValue(outcome == "retry_rollback"),
        ]
    manager._sync_single_engine_weights = AsyncMock(return_value=True)
    request = ScaleOutRequest(request_id="test", status=ScaleOutStatus.CREATING, model_name="actor")
    finalization = manager._finalize_engine_group_registration(
        request, server, [engine], engine_init_pairs=[(engine, new_record)], rank_offset=1
    )
    if outcome == "unconfirmed":
        with pytest.raises(RuntimeError, match="Rollback resource release is unconfirmed"):
            await asyncio.wait_for(finalization, 3)
        assert not (await owner.snapshot()).models[0].admission
        with pytest.raises(RuntimeError, match="cannot confirm scale-out rollback"):
            await asyncio.wait_for(owner.wait_ready("rollout"), 1)
        engine.shutdown.remote.assert_not_called()
        assert not held
        return
    result = await asyncio.wait_for(finalization, 3)
    assert result.success is registered
    assert not held
    snapshot = (await owner.snapshot()).models[0]
    assert snapshot.admission and snapshot.required_weight_version == "7"
    if registered:
        assert [(item.base_url, item.weight_version) for item in snapshot.replicas] == [
            ("http://seed:1", "7"),
            ("http://elastic:1", "7"),
        ]
    else:
        assert result.cleanup_confirmed
        assert result.reason.category is ScaleOutFailureCategory.ROUTER_REGISTRATION_FAILED
        assert engine.unregister_from_router.remote.call_count == 2
        assert len(snapshot.replicas) == 1


@pytest.mark.asyncio
async def test_weight_sync_failure_never_registers_engine_to_dcs():
    manager = create_test_manager()
    engine = make_mock_engine()
    server = make_rollout_server()
    request = ScaleOutRequest(request_id="test", status=ScaleOutStatus.CREATING)
    manager._health_check_engines = AsyncMock(return_value=True)
    manager._sync_weights_from_seed_engine = AsyncMock(return_value=False)

    result = await manager._finalize_engine_group_registration(
        request=request,
        srv=server,
        engines=[engine],
    )

    assert result.success is False
    assert result.group is None
    assert result.reason is not None and result.reason.category is ScaleOutFailureCategory.WEIGHT_SYNC_FAILED
    engine.register_dcs.remote.assert_not_called()
    engine.register_to_router.remote.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_scale_out_retains_execution_ownership_until_cleanup():
    server = make_rollout_server()
    manager = create_test_manager(servers={"default": server})
    cleanup_started, finish_cleanup = asyncio.Event(), asyncio.Event()

    async def bring_up(request):
        request.update_status(ScaleOutStatus.CREATING)
        cleanup_started.set()
        await finish_cleanup.wait()

    manager._scale_out_ray_native = bring_up
    first = manager.create_scale_out_request(num_replicas=2)
    running = asyncio.create_task(manager.execute_scale_out(first["request_id"]))
    await asyncio.wait_for(cleanup_started.wait(), 1)
    assert manager.cancel_scale_out(first["request_id"])["status"] == "CANCELLED"
    try:
        blocked = manager.create_scale_out_request(num_replicas=2)
        assert blocked["status"] == "CONFLICT"
        assert first["request_id"] in blocked["message"]
    finally:
        finish_cleanup.set()
        await running
    assert manager.create_scale_out_request(num_replicas=2)["status"] == "PENDING"


@pytest.mark.asyncio
async def test_failed_native_scale_out_reports_cleanup_failure_and_keeps_placement_group(monkeypatch, patch_ray_get):
    from relax.distributed.ray import policy_engine_pool as policy

    engine = make_mock_engine()
    pg = SimpleNamespace(ready=lambda: AwaitableValue(None))
    healthy_pg = SimpleNamespace(ready=lambda: AwaitableValue(None))
    server = make_rollout_server()
    manager = create_test_manager(servers={"default": server})
    manager._scale_out_requests["request"] = ScaleOutRequest(
        request_id="request", status=ScaleOutStatus.PENDING, num_replicas=2
    )

    async def fail_bringup(**kwargs):
        if kwargs["pg"] is healthy_pg:
            return SimpleNamespace(success=True, reason=None)
        assert kwargs["pg"] is pg
        await manager._rollback_engines([engine], confirm_router_removal=True)

    manager._bring_up_single_replica = fail_bringup
    create = MagicMock(side_effect=[healthy_pg, pg])
    monkeypatch.setattr(policy.ray.util, "placement_group", create)
    remove = MagicMock()
    monkeypatch.setattr(policy.ray.util, "remove_placement_group", remove)
    engine.shutdown.remote.side_effect = TimeoutError("exit pending")
    with pytest.raises(RuntimeError, match="did not confirm resource release"):
        await manager.execute_scale_out("request")
    status = manager.get_scale_out_status("request")
    assert status["status"] == "FAILED"
    assert status["engine_ids"] == ["engine_1"]
    assert status["failure_categories"][0] == "ROLLBACK_UNCONFIRMED"
    assert "rollback resource release is unconfirmed" in status["error_message"]
    remove.assert_not_called()
    engine.shutdown.remote.reset_mock()
    engine.shutdown.remote.side_effect = None
    engine.unregister_from_router.remote.return_value = AwaitableValue(False)
    with pytest.raises(RuntimeError, match="resource release is unconfirmed"):
        manager._shutdown_all_engines()
    engine.shutdown.remote.assert_not_called()
    remove.assert_not_called()
    engine.unregister_from_router.remote.return_value = AwaitableValue(True)
    manager._shutdown_all_engines()
    engine.shutdown.remote.assert_called_once()
    remove.assert_called_once_with(pg)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_during_wait", [False, True])
async def test_unused_scale_out_pg_is_retained_when_removal_fails(monkeypatch, patch_ray_get, cancel_during_wait):
    from relax.distributed.ray import policy_engine_pool as policy

    waiting, resume = asyncio.Event(), asyncio.Event()

    async def ready():
        waiting.set()
        await resume.wait()

    pg = SimpleNamespace(ready=ready)
    manager = create_test_manager(servers={"default": make_rollout_server()})
    create = MagicMock(side_effect=[pg] if cancel_during_wait else [pg, RuntimeError("PG allocation failed")])
    remove = MagicMock(side_effect=RuntimeError("PG removal failed"))
    monkeypatch.setattr(policy.ray.util, "placement_group", create)
    monkeypatch.setattr(policy.ray.util, "remove_placement_group", remove)
    request = manager.create_scale_out_request(num_replicas=2 if cancel_during_wait else 3)
    request_id = request["request_id"]
    running = asyncio.create_task(manager.execute_scale_out(request_id))
    if cancel_during_wait:
        await asyncio.wait_for(waiting.wait(), 1)
        assert manager.cancel_scale_out(request_id)["status"] == "CANCELLED"
        resume.set()
    with pytest.raises(RuntimeError, match="did not confirm resource release"):
        await asyncio.wait_for(running, 1)
    status = manager.get_scale_out_status(request_id)
    assert status["status"] == "FAILED"
    assert "ROLLBACK_UNCONFIRMED" in status["failure_categories"]
    assert manager.create_scale_out_request(num_replicas=3)["status"] == "CONFLICT"
    remove.assert_called_once_with(pg)
    remove.side_effect = None
    manager._shutdown_all_engines()
    assert remove.call_count == 2
    remove.assert_called_with(pg)


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_ok", [True, False])
async def test_healthy_replica_waits_for_live_rollback(patch_ray_get, cleanup_ok):
    seed = make_mock_engine(url="http://seed:1", weight_version="7")
    healthy, failed = make_mock_engine(), make_mock_engine()
    server = make_rollout_server(engine_groups=[make_engine_group(engines=[seed])])
    manager = create_test_manager(servers={"default": server})
    entered, finish = asyncio.Event(), asyncio.Event()

    async def shutdown():
        entered.set()
        await finish.wait()
        if not cleanup_ok:
            raise RuntimeError("unconfirmed shutdown")

    failed.shutdown.remote.side_effect = shutdown
    manager._sync_single_engine_weights = AsyncMock(return_value=True)
    rollback = asyncio.create_task(manager._rollback_engines([failed], confirm_router_removal=True))
    await asyncio.wait_for(entered.wait(), 1)
    sync = asyncio.create_task(manager._sync_weights_from_seed_engine([healthy], timeout=2))
    await asyncio.sleep(0)
    assert not sync.done()
    manager._weight_sync_lock.acquire.remote.assert_not_called()
    finish.set()
    if cleanup_ok:
        await rollback
    else:
        with pytest.raises(RuntimeError, match="unconfirmed"):
            await rollback
    assert await asyncio.wait_for(sync, 2) is cleanup_ok


@pytest.mark.asyncio
async def test_external_scale_out_reserves_ranks_after_another_model(monkeypatch):
    from relax.distributed.ray import policy_engine_pool as policy

    server = make_rollout_server()
    manager = create_test_manager(
        servers={"default": server, "other": make_rollout_server(engine_groups=[make_engine_group(rank_offset=8)])}
    )
    engine = make_mock_engine()
    engine.init.remote.return_value = AwaitableValue(None)
    actor_cls = MagicMock()
    actor_cls.options.return_value.remote.return_value = engine
    monkeypatch.setattr(policy.ray, "remote", lambda actor: actor_cls)
    manager._finalize_engine_group_registration = AsyncMock(return_value=EngineFinalizeResult(True))
    request = ScaleOutRequest(request_id="test", status=ScaleOutStatus.PENDING, engine_urls=["remote:30000"])

    await manager._scale_out_external(request)

    assert request.status is ScaleOutStatus.ACTIVE
    assert request.engine_ids == ["engine_9"]
    assert actor_cls.options.return_value.remote.call_args.kwargs["rank"] == 9
    assert manager._finalize_engine_group_registration.call_args.kwargs["rank_offset"] == 9


@pytest.mark.asyncio
async def test_ray_native_finalizer_failure_rolls_back_precreated_group():
    manager = create_test_manager()
    server = make_rollout_server()
    request = ScaleOutRequest(request_id="test", status=ScaleOutStatus.CREATING)
    engine = make_mock_engine()
    group = SimpleNamespace(engines=[engine])
    group.start_engines = MagicMock(return_value=([AwaitableValue(None)], {}))
    manager._finalize_engine_group_registration = AsyncMock(
        return_value=EngineFinalizeResult(False, reason=ScaleOutFailure(ScaleOutFailureCategory.WEIGHT_SYNC_FAILED))
    )
    manager._rollback_engines = AsyncMock()

    info_actor = MagicMock()
    info_actor.get_ip_and_gpu_id.remote.return_value = AwaitableValue(("10.0.0.1", 0))
    info_actor_class = MagicMock()
    info_actor_class.options.return_value.remote.return_value = info_actor

    with (
        patch("relax.distributed.ray.policy_engine_pool.EngineGroup", return_value=group),
        patch("relax.distributed.ray.policy_engine_pool.ray.kill"),
    ):
        result = await manager._bring_up_single_replica(
            request=request,
            srv=server,
            pg=object(),
            replica_idx=0,
            num_gpus=1,
            gpus_per_engine=1,
            engine_offset=1,
            sort_key=lambda item: item,
            InfoActor=info_actor_class,
        )

    assert result.success is False
    assert result.reason is not None and result.reason.category is ScaleOutFailureCategory.WEIGHT_SYNC_FAILED
    manager._rollback_engines.assert_awaited_once_with(group)


@pytest.mark.asyncio
async def test_ray_native_multinode_replica_allocates_one_slot_per_actor():
    args = make_mock_args(num_gpus_per_node=8, rollout_num_gpus_per_engine=16)
    manager = create_test_manager(args=args)
    server = make_rollout_server()
    request = ScaleOutRequest(request_id="test", status=ScaleOutStatus.CREATING)
    engines = [make_mock_engine(), make_mock_engine()]
    group = SimpleNamespace(
        engines=engines,
        start_engines=MagicMock(return_value=([AwaitableValue(None)] * 2, {})),
    )
    captured = {}

    def capture_group(**kwargs):
        captured.update(kwargs)
        return group

    manager._finalize_engine_group_registration = AsyncMock(return_value=EngineFinalizeResult(True, group=group))

    info_actor = MagicMock()
    info_actor.get_ip_and_gpu_id.remote.return_value = AwaitableValue(("10.0.0.1", 0))
    info_actor_class = MagicMock()
    info_actor_class.options.return_value.remote.return_value = info_actor

    with (
        patch("relax.distributed.ray.policy_engine_pool.EngineGroup", side_effect=capture_group),
        patch("relax.distributed.ray.policy_engine_pool.ray.kill"),
    ):
        result = await manager._bring_up_single_replica(
            request=request,
            srv=server,
            pg=object(),
            replica_idx=0,
            num_gpus=16,
            gpus_per_engine=16,
            engine_offset=2,
            sort_key=lambda item: item,
            InfoActor=info_actor_class,
        )

    assert result.success is True
    assert captured["all_engines"] == [None, None]
    manager._finalize_engine_group_registration.assert_awaited_once_with(
        request=request,
        srv=server,
        engines=engines,
        engine_group=group,
        replica_idx=0,
        log_prefix="[ScaleOut]",
    )


@pytest.mark.asyncio
async def test_ray_native_scale_out_inherits_canonical_model_override():
    args = make_mock_args(hf_checkpoint="/dev/shm/node-local-model")
    template = make_engine_group(args=args)
    template.sglang_overrides = {
        "model_path": "s3://model-bucket/canonical-model/",
        "load_format": "auto",
    }
    server = make_rollout_server(engine_groups=[template])
    manager = create_test_manager(args=args, servers={"default": server})
    request = ScaleOutRequest(request_id="test", status=ScaleOutStatus.CREATING)
    engine = make_mock_engine()
    group = SimpleNamespace(engines=[engine])
    group.start_engines = MagicMock(return_value=([AwaitableValue(None)], {}))
    manager._finalize_engine_group_registration = AsyncMock(
        return_value=EngineFinalizeResult(False, reason=ScaleOutFailure(ScaleOutFailureCategory.WEIGHT_SYNC_FAILED))
    )
    manager._rollback_engines = AsyncMock()

    info_actor = MagicMock()
    info_actor.get_ip_and_gpu_id.remote.return_value = AwaitableValue(("192.0.2.2", 0))
    info_actor_class = MagicMock()
    info_actor_class.options.return_value.remote.return_value = info_actor

    with (
        patch("relax.distributed.ray.policy_engine_pool.EngineGroup", return_value=group) as engine_group_cls,
        patch("relax.distributed.ray.policy_engine_pool.ray.kill"),
    ):
        await manager._bring_up_single_replica(
            request=request,
            srv=server,
            pg=object(),
            replica_idx=0,
            num_gpus=1,
            gpus_per_engine=1,
            engine_offset=4,
            sort_key=lambda item: item,
            InfoActor=info_actor_class,
        )

    inherited = engine_group_cls.call_args.kwargs["sglang_overrides"]
    assert inherited == template.sglang_overrides
    assert inherited is not template.sglang_overrides
    assert inherited["model_path"] != args.hf_checkpoint


@pytest.mark.asyncio
async def test_router_false_is_registration_failure():
    manager = create_test_manager()
    engine = make_mock_engine()
    engine.register_to_router.remote.return_value = AwaitableValue(False)
    server = make_rollout_server()
    request = ScaleOutRequest(request_id="test", status=ScaleOutStatus.CREATING)
    manager._health_check_engines = AsyncMock(return_value=True)
    manager._sync_weights_from_seed_engine = AsyncMock(side_effect=_sync_success)

    result = await manager._finalize_engine_group_registration(
        request=request,
        srv=server,
        engines=[engine],
    )

    assert result.success is False
    assert result.group is None
    assert result.reason is not None and result.reason.category is ScaleOutFailureCategory.ROUTER_REGISTRATION_FAILED


@pytest.mark.asyncio
async def test_finalize_propagates_precheck_root_cause_to_error_message():
    """A weight-sync/precheck failure must surface the specific root cause
    (e.g. NCCL wrong_type) all the way into ScaleOutRequest.error_message +
    failure_categories, so the monitor TUI can categorize it instead of showing
    a generic 'All N replicas failed'."""
    from relax.utils.autoscaler.monitor import _categorize_scale_error

    manager = create_test_manager()
    engine = make_mock_engine()
    server = make_rollout_server()
    request = ScaleOutRequest(request_id="test", num_replicas=1, status=ScaleOutStatus.CREATING)
    manager._health_check_engines = AsyncMock(return_value=True)

    async def _fake_sync(*args, reason_sink=None, **kwargs):
        # Emulate _run_scale_weight_sync_precheck surfacing a transport-type mismatch.
        if reason_sink is not None:
            reason_sink.append(ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_TRANSPORT_MISMATCH, "wrong_type"))
        return False

    manager._sync_weights_from_seed_engine = AsyncMock(side_effect=_fake_sync)

    # Step 1: the finalizer returns the specific classified root cause.
    result = await manager._finalize_engine_group_registration(
        request=request,
        srv=server,
        engines=[engine],
    )
    assert result.success is False
    assert result.group is None
    assert result.reason.category is ScaleOutFailureCategory.NCCL_PRECHECK_TRANSPORT_MISMATCH
    assert "wrong_type" in str(result.reason)

    # Step 2: the aggregated reason + structured categories land on the request.
    manager._update_scale_out_final_status(request, server, [], ["replica_0"], [result.reason])
    assert request.status == ScaleOutStatus.FAILED
    assert "wrong_type" in request.error_message
    assert "replica_0" in request.error_message
    assert not request.error_message.startswith("All 1 replicas failed")
    assert request.failure_categories == ["NCCL_PRECHECK_TRANSPORT_MISMATCH"]

    # Step 3: the monitor categorizes it via the structured category (no
    # error_message string guessing).
    assert _categorize_scale_error("", categories=request.failure_categories) == "NCCL transport mismatch"


@pytest.mark.asyncio
async def test_final_status_provision_timeout_is_categorizable():
    """A PG provision timeout reason must categorize as elastic node provision
    timeout (not the generic 'All N replicas failed')."""
    from relax.utils.autoscaler.monitor import _categorize_scale_error

    manager = create_test_manager()
    server = make_rollout_server()
    request = ScaleOutRequest(request_id="test", num_replicas=2, status=ScaleOutStatus.CREATING)

    reasons = [ScaleOutFailure(ScaleOutFailureCategory.PROVISION_TIMEOUT)]
    manager._update_scale_out_final_status(request, server, [], ["replica_0", "replica_1"], reasons)

    assert request.status == ScaleOutStatus.FAILED
    assert "provision" in request.error_message
    assert request.failure_categories == ["PROVISION_TIMEOUT"]
    assert _categorize_scale_error("", categories=request.failure_categories) == "elastic node provision timeout"


@pytest.mark.asyncio
async def test_engine_init_timeout_propagates_as_provision_timeout():
    """A wait_for timeout during engine init carries an empty message; the
    reason must still explicitly say 'timed out' + 'provision' so the monitor
    buckets it as an elastic-node provision timeout instead of 'other'."""
    from relax.utils.autoscaler.monitor import _categorize_scale_error

    manager = create_test_manager()
    server = make_rollout_server()
    request = ScaleOutRequest(request_id="test", num_replicas=1, status=ScaleOutStatus.CREATING)
    engine = make_mock_engine()
    group = SimpleNamespace(engines=[engine])
    group.start_engines = MagicMock(return_value=([AwaitableValue(None)], {}))
    manager._rollback_engines = AsyncMock()

    info_actor = MagicMock()
    info_actor.get_ip_and_gpu_id.remote.return_value = AwaitableValue(("10.0.0.1", 0))
    info_actor_class = MagicMock()
    info_actor_class.options.return_value.remote.return_value = info_actor

    async def _raise_timeout(*args, **kwargs):
        raise asyncio.TimeoutError()

    with (
        patch("relax.distributed.ray.policy_engine_pool.EngineGroup", return_value=group),
        patch("relax.distributed.ray.policy_engine_pool.ray.kill"),
        patch("relax.distributed.ray.policy_engine_pool.asyncio.wait_for", side_effect=_raise_timeout),
    ):
        result = await manager._bring_up_single_replica(
            request=request,
            srv=server,
            pg=object(),
            replica_idx=0,
            num_gpus=1,
            gpus_per_engine=1,
            engine_offset=1,
            sort_key=lambda item: item,
            InfoActor=info_actor_class,
        )

    assert result.success is False
    assert result.reason.category is ScaleOutFailureCategory.PROVISION_TIMEOUT
    manager._rollback_engines.assert_awaited_once_with(group)

    # The reason must reach error_message and be categorizable via the
    # structured category.
    manager._update_scale_out_final_status(request, server, [], ["replica_0"], [result.reason])
    assert "timed out" in request.error_message
    assert request.failure_categories == ["PROVISION_TIMEOUT"]
    assert _categorize_scale_error("", categories=request.failure_categories) == "elastic node provision timeout"


@pytest.mark.asyncio
async def test_external_outer_catch_does_not_leak_exception_args():
    """The outermost catch-all must surface only the exception *type* (never
    its args, which may carry addresses / secrets) and set a structured
    category."""
    from unittest.mock import PropertyMock

    manager = create_test_manager()
    request = ScaleOutRequest(
        request_id="test", status=ScaleOutStatus.PENDING, model_name="default", engine_urls=["h:1"]
    )

    secret = "sk-supersecret-0123456789 host=192.0.2.7"
    server = MagicMock()
    server.router_ip = "10.0.0.1"
    # Force the outer try/except to trip with a secret-bearing exception.
    type(server).router_port = PropertyMock(side_effect=RuntimeError(secret))
    manager._get_server = MagicMock(return_value=server)
    manager._rollback_engines = AsyncMock()

    await manager._scale_out_external(request)

    assert request.status == ScaleOutStatus.FAILED
    assert secret not in request.error_message
    assert "192.0.2.7" not in request.error_message
    assert "RuntimeError" in request.error_message
    assert request.failure_categories == ["UNKNOWN"]


def test_error_message_is_bounded_and_scrubbed():
    """Aggregated error_message must be length-bounded, cap the number of
    distinct reasons, and never carry raw exception text / tracebacks /
    addresses."""
    manager = create_test_manager()
    server = make_rollout_server()
    request = ScaleOutRequest(request_id="test", num_replicas=8, status=ScaleOutStatus.CREATING)

    secret = "sk-supersecret-token-abcdef0123456789"
    traceback_blob = "Traceback (most recent call last):\n  File x\n" + ("A" * 4000)
    reasons = [
        ScaleOutFailure(ScaleOutFailureCategory.ENGINE_INIT_FAILED, f"{traceback_blob} host=192.0.2.13 {secret}"),
        ScaleOutFailure(ScaleOutFailureCategory.WEIGHT_SYNC_FAILED, "NCCL sync failed for all engines"),
        ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_TRANSPORT_MISMATCH, "wrong_type"),
        ScaleOutFailure(ScaleOutFailureCategory.HEALTH_CHECK_FAILED),
        ScaleOutFailure(ScaleOutFailureCategory.DCS_REGISTRATION_FAILED, "some very long ray error " + ("B" * 500)),
        ScaleOutFailure(ScaleOutFailureCategory.ROUTER_REGISTRATION_FAILED, "Router rejected 3 engine(s)"),
    ]
    failed_ids = [f"replica_{i}" for i in range(6)]
    manager._update_scale_out_final_status(request, server, [], failed_ids, reasons)

    msg = request.error_message
    # Bounded: aggregated reasons cap (512) + failed-engine suffix stays small.
    assert len(msg) <= 512 + 200
    # No raw traceback / secret / address leaked.
    assert "Traceback" not in msg
    assert secret not in msg
    assert "192.0.2.13" not in msg
    assert "AAAA" not in msg
    # Distinct-reason cap applied.
    assert "and 3 more" in msg
    # Stable category prefixes are surfaced.
    assert "weight sync failed" in msg
