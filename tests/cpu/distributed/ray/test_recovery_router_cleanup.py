# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from conftest import (
    HAS_DEPS,
    create_test_manager,
    make_engine_group,
    make_mock_args,
    make_mock_engine,
    make_rollout_server,
)


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


@pytest.mark.parametrize("component_recovery", [False, True])
def test_component_recovery_prevents_engine_rebuild_from_legacy_callers(component_recovery):
    manager = create_test_manager(args=make_mock_args(use_fault_tolerance=True, inference_recovery=component_recovery))
    server = make_rollout_server()
    server.recover = MagicMock()
    manager.servers = {"default": server}
    manager.rollout_id = 2
    manager.rollout_engine_lock = object()
    manager.health_monitoring_pause = MagicMock()
    expected = manager.get_rollout_engines_and_lock()

    assert manager.recover_rollout_engines() == expected

    assert server.recover.call_count == int(not component_recovery)
    assert manager.health_monitoring_pause.call_count == int(not component_recovery)


@pytest.mark.asyncio
@pytest.mark.parametrize("component_recovery", [False, True])
async def test_generation_uses_legacy_fault_injection_only_with_engine_recovery(component_recovery):
    manager = create_test_manager(
        args=make_mock_args(
            ci_test=True,
            use_fault_tolerance=True,
            inference_recovery=component_recovery,
            partial_rollout=False,
        )
    )
    manager._try_ci_fault_injection = MagicMock()
    manager._generate_batch = MagicMock()

    await manager.generate(2)

    manager._generate_batch.assert_called_once_with(2)
    assert manager._try_ci_fault_injection.call_count == int(not component_recovery)


@pytest.mark.asyncio
@pytest.mark.parametrize("component_recovery", [False, True])
async def test_engine_registration_starts_monitor_only_with_engine_recovery(monkeypatch, component_recovery):
    from relax.distributed.ray import rollout

    manager = create_test_manager(args=make_mock_args(use_fault_tolerance=True, inference_recovery=component_recovery))
    server = make_rollout_server()
    engine = make_mock_engine()
    manager._health_check_engines = AsyncMock(return_value=True)
    manager._sync_weights_from_seed_engine = AsyncMock(return_value=True)
    monitor = MagicMock()
    monkeypatch.setattr(rollout, "RolloutHealthMonitor", monitor)
    monkeypatch.setattr(rollout.ray, "get", lambda value: value.value)
    request = rollout.ScaleOutRequest(request_id="test", status=rollout.ScaleOutStatus.CREATING)

    result = await manager._finalize_engine_group_registration(request, server, engines=[engine])

    assert result.success
    assert manager._health_monitors == ([] if component_recovery else [monitor.return_value])
    assert monitor.return_value.start.call_count == int(not component_recovery)


def test_recovery_removes_dead_route_before_creating_same_endpoint(monkeypatch):
    from relax.distributed.ray import rollout

    old_url = "http://worker:15000"
    group = make_engine_group(engines=[None], is_scaled_out=True)
    group.skip_router_registration = True
    group.args.fully_async = True
    group.pg = (object(), [0], [0])
    group.router_ip, group.router_port = "router", 30000
    record = rollout._EngineInitRecord(
        {"url": old_url, "pid": 1, "node_id": "node"}, router_cleanup=rollout.RouterWorkerCleanup("A")
    )
    group.engine_init_records[0] = record
    events = []

    def cleanup(*args, **kwargs):
        assert group.all_engines == [None]
        assert group.engine_init_records[0] is record
        events.append("old route removed")
        return True

    def start(cursors):
        assert events == ["old route removed"], "B can inherit A's healthy Router entry before receiving weights"
        group.all_engines[0] = SimpleNamespace()
        group.num_new_engines = 1
        events.append("B init submitted")
        return [], cursors

    monkeypatch.setattr(rollout, "remove_dead_router_worker", cleanup, raising=False)
    monkeypatch.setattr(group, "start_engines", start)
    make_rollout_server(engine_groups=[group]).recover()
    assert events == ["old route removed", "B init submitted"]


def test_recovery_cleanup_failure_preserves_generation_and_retries(monkeypatch):
    from relax.distributed.ray import rollout

    group = make_engine_group(engines=[None], is_scaled_out=True)
    group.skip_router_registration = True
    group.args.fully_async = True
    group.pg = (object(), [0], [0])
    group.router_ip, group.router_port = "router", 30000
    record = rollout._EngineInitRecord(
        {"url": "http://worker:15000", "pid": 1, "node_id": "node"}, router_cleanup=rollout.RouterWorkerCleanup("A")
    )
    group.engine_init_records[0] = record
    group.num_new_engines = 7  # Stale previous-round count must not cause reconnection.
    cleanup = MagicMock(return_value=False)
    start = MagicMock(return_value=([], {}))
    monkeypatch.setattr(rollout, "remove_dead_router_worker", cleanup, raising=False)
    monkeypatch.setattr(group, "start_engines", start)
    server = make_rollout_server(engine_groups=[group])

    server.recover()

    start.assert_not_called()
    assert group.engine_init_records[0] is record
    assert group.all_engines == [None]
    assert server.num_new_engines == 0

    cleanup.return_value = True

    def recovered(cursors):
        group.all_engines[0] = SimpleNamespace()
        group.num_new_engines = 1
        return [], cursors

    start.side_effect = recovered
    server.recover()
    assert cleanup.call_count == 2
    start.assert_called_once()


def test_recovery_retries_after_replacement_dies_during_init(monkeypatch):
    from relax.distributed.ray import rollout

    group = make_engine_group(engines=[None], is_scaled_out=True)
    group.args.fully_async = True
    group.skip_router_registration = True
    group.pg = (object(), [0], [0])
    failed_init_ref = object()
    group.engine_init_records[0] = rollout._EngineInitRecord(
        failed_init_ref,
        pending_router_registration=True,
        router_url="http://worker:15000",
        router_cleanup=rollout.RouterWorkerCleanup("B"),
    )
    get = MagicMock(side_effect=RuntimeError("B died before init could return metadata"))
    cleanup = MagicMock(return_value=True)
    monkeypatch.setattr(rollout.ray, "get", get)
    monkeypatch.setattr(rollout, "remove_dead_router_worker", cleanup)

    def start(cursors):
        group.all_engines[0] = SimpleNamespace()
        group.num_new_engines = 1
        return [], cursors

    start_mock = MagicMock(side_effect=start)
    monkeypatch.setattr(group, "start_engines", start_mock)
    make_rollout_server(engine_groups=[group]).recover()
    get.assert_not_called()
    cleanup.assert_called_once_with(
        "http://127.0.0.1:3000", "http://worker:15000", group.engine_init_records[0].router_cleanup
    )
    start_mock.assert_called_once()


@pytest.mark.parametrize("mode", ["sync", "dp", "legacy_router", "slime_router"])
def test_recovery_cleanup_leaves_other_protocols_unchanged(monkeypatch, mode):
    from relax.distributed.ray import rollout

    group = make_engine_group(engines=[None], is_scaled_out=True)
    group.args.fully_async = mode != "sync"
    group.args.sglang_dp_size = 2 if mode == "dp" else 1
    group.args.use_slime_router = mode == "slime_router"
    if mode == "legacy_router":
        monkeypatch.setattr(rollout.sglang_router, "__version__", "0.2.1")
    group.skip_router_registration = True
    group.pg = (object(), [0], [0])
    cleanup = MagicMock()
    monkeypatch.setattr(rollout, "remove_dead_router_worker", cleanup)

    def start(cursors):
        group.all_engines[0] = SimpleNamespace()
        group.num_new_engines = 1
        return [], cursors

    monkeypatch.setattr(group, "start_engines", start)
    make_rollout_server(engine_groups=[group]).recover()
    cleanup.assert_not_called()
    assert group.num_new_engines == 1
