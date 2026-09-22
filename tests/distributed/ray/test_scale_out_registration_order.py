# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for scale-out engine publication ordering."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


try:
    from relax.distributed.ray.rollout import EngineFinalizeResult, ScaleOutRequest, ScaleOutStatus
    from relax.utils.scale_utils import ScaleOutFailure, ScaleOutFailureCategory

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from conftest import AwaitableValue, create_test_manager, make_mock_engine, make_rollout_server


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


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
        patch("relax.distributed.ray.rollout.EngineGroup", return_value=group),
        patch("relax.distributed.ray.rollout.ray.kill"),
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
async def test_router_false_is_registration_failure():
    manager = create_test_manager()
    engine = make_mock_engine()
    engine.register_to_router.remote.return_value = AwaitableValue(False)
    server = make_rollout_server()
    request = ScaleOutRequest(request_id="test", status=ScaleOutStatus.CREATING)
    manager._health_check_engines = AsyncMock(return_value=True)
    manager._sync_weights_from_seed_engine = AsyncMock(return_value=True)

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
        patch("relax.distributed.ray.rollout.EngineGroup", return_value=group),
        patch("relax.distributed.ray.rollout.ray.kill"),
        patch("relax.distributed.ray.rollout.asyncio.wait_for", side_effect=_raise_timeout),
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
    server.router_port = 8000
    # Force the outer try/except to trip with a secret-bearing exception.
    type(server).engine_groups = PropertyMock(side_effect=RuntimeError(secret))
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
