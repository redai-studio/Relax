# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import requests
from conftest import (
    HAS_DEPS,
    create_test_manager,
    make_engine_group,
    make_mock_args,
    make_mock_engine,
    make_rollout_server,
    mock_ray_get,
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
async def test_manual_engine_recovery_reports_whether_it_can_run(component_recovery):
    from fastapi import HTTPException

    from relax.components.rollout import Rollout

    rpc = AsyncMock()
    replica = SimpleNamespace(
        config=make_mock_args(inference_recovery=component_recovery),
        rollout_manager=SimpleNamespace(recover_rollout_engines=SimpleNamespace(remote=rpc)),
        _logger=MagicMock(),
    )
    recover = Rollout.func_or_class.recover_rollout_engines
    if component_recovery:
        with pytest.raises(HTTPException) as error:
            await recover(replica)
        assert error.value.status_code == 409
        rpc.assert_not_awaited()
    else:
        assert await recover(replica) == {"status": "ok"}
        rpc.assert_awaited_once()


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

    def start(cursors, *, slots=None):
        assert slots is None  # Without autoscaler, legacy Router cleanup still runs.
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

    def recovered(cursors, *, slots=None):
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

    def start(cursors, *, slots=None):
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


@pytest.mark.parametrize("mode", ["sync", "dp", "legacy_router"])
def test_recovery_cleanup_leaves_other_protocols_unchanged(monkeypatch, mode):
    from relax.distributed.ray import rollout

    group = make_engine_group(engines=[None], is_scaled_out=True)
    group.args.fully_async = mode != "sync"
    group.args.sglang_dp_size = 2 if mode == "dp" else 1
    if mode == "legacy_router":
        monkeypatch.setattr(rollout.sglang_router, "__version__", "0.2.3")
    group.skip_router_registration = True
    group.pg = (object(), [0], [0])
    cleanup = MagicMock()
    monkeypatch.setattr(rollout, "remove_dead_router_worker", cleanup)

    def start(cursors, *, slots=None):
        group.all_engines[0] = SimpleNamespace()
        group.num_new_engines = 1
        return [], cursors

    monkeypatch.setattr(group, "start_engines", start)
    make_rollout_server(engine_groups=[group]).recover()
    cleanup.assert_not_called()
    assert group.num_new_engines == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("during_scale_in", [False, True])
async def test_health_monitor_cleans_dead_elastic_registration(monkeypatch, during_scale_in):
    from ray.exceptions import ActorDiedError

    from relax.distributed.checkpoint_service.config import RoleInfo
    from relax.distributed.checkpoint_service.coordinator.topology import TopologyManager
    from relax.distributed.ray import rollout

    engine = MagicMock()
    engine._actor_id.hex.return_value = "dead-actor"
    group = make_engine_group(engines=[engine], is_scaled_out=True)
    group.skip_router_registration = True
    group.args._relax_elastic_rollout_enabled = True
    group.pg = (object(), [0], [0])
    group.rank_offset = 5
    if during_scale_in:
        group.lifecycle_status = rollout.EngineGroupLifecycle.DRAINING
    group.args.coordinator_url = "http://coordinator/dcs_coordinator"
    group.args.rollout_health_check_first_wait = 0
    group.args.rollout_health_check_max_consecutive_failures = 2
    url = "http://worker:15000"
    group.engine_init_records[0] = rollout._EngineInitRecord(
        {"url": url}, router_cleanup=rollout.RouterWorkerCleanup("dead-actor")
    )
    topology = TopologyManager()
    topology.register(RoleInfo(role_name="rollout", rank=5, metadata={"actor_id": "dead-actor"}))

    fail_first_delete = during_scale_in

    def unregister(url, *, params, timeout):
        nonlocal fail_first_delete
        assert url == "http://coordinator/dcs_coordinator/unregister"
        if fail_first_delete:
            fail_first_delete = False
            raise requests.Timeout("coordinator temporarily unavailable")
        status = 200 if topology.unregister(params["role"], params["rank"], params["actor_id"]) else 404
        return MagicMock(status_code=status)

    monkeypatch.setattr(rollout.requests, "delete", unregister)
    remove_router = MagicMock(return_value=True)
    monkeypatch.setattr(rollout, "remove_dead_router_worker", remove_router)
    monkeypatch.setattr(rollout.ray, "get", MagicMock(side_effect=ActorDiedError()))
    monitor = rollout.RolloutHealthMonitor(group, group.args, on_actor_dead=group.cleanup_confirmed_dead_engine)

    monitor._run_health_checks()

    assert group.all_engines == [None]
    if during_scale_in:
        remove_router.assert_not_called()
        assert list(topology.get_role_nodes("rollout")) == [5]

        remove_pg = MagicMock()
        monkeypatch.setattr(rollout.ray.util, "remove_placement_group", remove_pg)
        server = make_rollout_server(engine_groups=[group])
        manager = create_test_manager(servers={"default": server})
        manager._cleanup_engine_groups(server)
        remove_pg.assert_not_called()
        assert server.engine_groups == [group]

        removed, failed = await manager._remove_live_engines(
            server, [(group, 0)], drain_timeout=0, shutdown_timeout=1, force=True
        )
        assert (removed, failed) == (["group_5_engine_0"], [])
        remove_pg.assert_called_once_with(group.pg[0])
        assert server.engine_groups == []

    assert topology.get_role_nodes("rollout") == {}
    remove_router.assert_called_once_with("http://127.0.0.1:3000", url, group.engine_init_records[0].router_cleanup)
    assert remove_router.call_args.args[2].actor_confirmed_dead


def test_dcs_cleanup_cannot_unregister_same_rank_replacement():
    from relax.distributed.checkpoint_service.config import RoleInfo
    from relax.distributed.checkpoint_service.coordinator.topology import TopologyManager

    topology = TopologyManager()
    topology.register(RoleInfo(role_name="rollout", rank=5, metadata={"actor_id": "old"}))
    replacement = RoleInfo(role_name="rollout", rank=5, metadata={"actor_id": "replacement"})
    topology.register(replacement)

    assert not topology.unregister("rollout", 5, "old")
    assert topology.get_role_nodes("rollout") == {5: replacement}
    assert topology.unregister("rollout", 5, "replacement")
    assert topology.get_role_nodes("rollout") == {}


def test_eviction_poll_retries_cleanup_after_claimed_actor_dies(monkeypatch):
    from relax.distributed.ray import rollout

    group = make_engine_group(engines=[None], is_scaled_out=True, rank_offset=5)
    group.pg = (object(), [0], [0])
    group.skip_router_registration = True
    group.args._relax_elastic_rollout_enabled = True
    group.lifecycle_status = rollout.EngineGroupLifecycle.DRAINING
    group.eviction_requested = True
    group.args.scale_in_drain_timeout = 0
    group.engine_init_records[0] = rollout._EngineInitRecord(
        {"url": "http://worker:15000"},
        router_cleanup=rollout.RouterWorkerCleanup("dead-actor", actor_confirmed_dead=True),
        dcs_unregistered=True,
    )
    cleanup = MagicMock(side_effect=[False, True])
    remove_pg = MagicMock()
    monkeypatch.setattr(rollout, "remove_dead_router_worker", cleanup)
    monkeypatch.setattr(rollout.ray.util, "remove_placement_group", remove_pg)
    server = make_rollout_server(engine_groups=[group])
    manager = create_test_manager(servers={"default": server}, args=group.args)

    manager._check_and_handle_evictions()
    assert manager.set_weight_updating(True) is False
    assert server.engine_groups == [group]
    remove_pg.assert_not_called()

    manager._check_and_handle_evictions()
    assert manager.set_weight_updating(True) is True
    assert server.engine_groups == []
    remove_pg.assert_called_once_with(group.pg[0])


def test_recovery_cleans_actor_dying_after_snapshot_before_next_rebuild(monkeypatch):
    from relax.distributed.ray import rollout

    group = make_engine_group(engines=[None, object()], is_scaled_out=True)
    group.pg = (object(), [0, 1], [0, 1])
    group.skip_router_registration = True
    group.args._relax_elastic_rollout_enabled = True
    group.args.coordinator_url = "http://coordinator/dcs_coordinator"
    first_url = "http://first:15000"
    late_url = "http://second:15000"
    group.engine_init_records.update(
        {
            0: rollout._EngineInitRecord({"url": first_url}, router_cleanup=rollout.RouterWorkerCleanup("first")),
            1: rollout._EngineInitRecord({"url": late_url}, router_cleanup=rollout.RouterWorkerCleanup("second")),
        }
    )
    events = []

    def cleanup(_router, url, record):
        events.append(("router", url, record.actor_id))
        if url == first_url:
            group.all_engines[1] = None
            group.engine_init_records[1].router_cleanup.actor_confirmed_dead = True
        return True

    def unregister(_url, *, params, timeout):
        events.append(("dcs", params["rank"], params["actor_id"]))
        return MagicMock(status_code=200)

    def start(cursors, *, slots):
        events.append(("start", slots))
        for slot in slots:
            group.all_engines[slot] = object()
        group.num_new_engines = len(slots)
        return [], cursors

    monkeypatch.setattr(rollout, "remove_dead_router_worker", cleanup)
    monkeypatch.setattr(rollout.requests, "delete", unregister)
    monkeypatch.setattr(group, "start_engines", start)
    server = make_rollout_server(engine_groups=[group])
    server.recover()

    assert group.all_engines[1] is None
    assert events == [("router", first_url, "first"), ("start", [0])]

    server.recover()

    first_cleanup, second_cleanup, rebuilt = events[2:]
    assert {first_cleanup, second_cleanup} == {("dcs", 1, "second"), ("router", late_url, "second")}
    assert rebuilt == ("start", [1])
    assert group.all_engines[1] is not None


def test_recovery_waits_for_all_nodes_of_elastic_engine(monkeypatch):
    from relax.distributed.ray import rollout

    args = make_mock_args(rollout_num_gpus=16, rollout_num_gpus_per_engine=16, _relax_elastic_rollout_enabled=True)
    peer = make_mock_engine()
    group = make_engine_group(args=args, engines=[None, peer], num_gpus_per_engine=16, is_scaled_out=True)
    group.pg = (object(), list(range(16)), list(range(8)) * 2)
    group.skip_router_registration = True
    group.engine_init_records[0] = rollout._EngineInitRecord(
        {"url": "http://old-head:15000"}, router_cleanup=rollout.RouterWorkerCleanup("old-head")
    )

    def finish_retiring_peer(*_):
        # The health monitor clears the peer after recover() captures its snapshot.
        group.all_engines[1] = None
        return True

    replacements = [make_mock_engine(), make_mock_engine()]
    for rank, engine in enumerate(replacements):
        engine._actor_id.hex.return_value = f"replacement-{rank}"
        engine._get_current_node_ip_and_free_port.remote.side_effect = (
            lambda start_port=15000, consecutive=1, rank=rank: (f"10.0.0.{rank + 1}", start_port)
        )
        engine.init.remote.return_value = object()
    actor_class = MagicMock()
    create_actor = actor_class.options.return_value.remote
    create_actor.side_effect = lambda *_, rank, **__: replacements[rank]
    monkeypatch.setattr(rollout, "remove_dead_router_worker", finish_retiring_peer)
    monkeypatch.setattr(rollout, "find_available_port", lambda _: 15000)
    monkeypatch.setattr(rollout.ray, "remote", lambda _: actor_class)
    monkeypatch.setattr(rollout.ray, "get", mock_ray_get)
    server = make_rollout_server(engine_groups=[group])

    server.recover()

    create_actor.assert_not_called()
    assert group.all_engines == [None, None]
    assert server.num_new_engines == 0

    server.recover()

    assert group.all_engines == replacements
    assert server.num_new_engines == 2
    head_address, peer_address = [engine.init.remote.call_args.kwargs["dist_init_addr"] for engine in replacements]
    assert head_address.startswith("10.0.0.1:")
    assert peer_address == head_address


@pytest.mark.parametrize("coordinator_failure", [None, "timeout", "proxy_404"])
def test_scale_in_owns_cleanup_after_router_removal(monkeypatch, coordinator_failure):
    from ray.exceptions import ActorDiedError

    from relax.distributed.ray import rollout

    engine = make_mock_engine()
    engine._actor_id.hex.return_value = "dead-actor"
    engine.unregister_dcs.remote.side_effect = ActorDiedError()
    group = make_engine_group(engines=[engine], is_scaled_out=True, rank_offset=5)
    group.pg = (object(), [0], [0])
    group.skip_router_registration = True
    group.args._relax_elastic_rollout_enabled = True
    group.args.coordinator_url = "http://coordinator/dcs_coordinator"
    group.args.scale_in_drain_timeout = 0
    group.engine_init_records[0] = rollout._EngineInitRecord(
        {"url": "http://worker:15000"}, router_cleanup=rollout.RouterWorkerCleanup("dead-actor")
    )
    server = make_rollout_server(engine_groups=[group])
    manager = create_test_manager(servers={"default": server}, args=group.args)
    monitor = MagicMock()
    monitor._engine_group = group
    # A health probe started before the intentional removal can report death after the Router ACK.
    monitor.mark_intentionally_removed.side_effect = lambda idx: group.cleanup_confirmed_dead_engine(idx, engine)
    manager._health_monitors.append(monitor)
    ok = MagicMock(status_code=200)
    deletion = MagicMock(return_value=ok)
    if coordinator_failure == "timeout":
        deletion.side_effect = [requests.Timeout(), ok]
    elif coordinator_failure == "proxy_404":
        proxy_response = requests.Response()
        proxy_response.status_code = 404
        proxy_response._content = b"Path '/dcs_coordinator/unregister' not found."
        coordinator_response = requests.Response()
        coordinator_response.status_code = 404
        coordinator_response._content = b'{"detail":"Node not found"}'
        deletion.side_effect = [proxy_response, coordinator_response]
    remove_pg = MagicMock()
    monkeypatch.setattr(rollout.requests, "delete", deletion)
    monkeypatch.setattr(rollout.ray.util, "remove_placement_group", remove_pg)
    request = rollout.ScaleInRequest(
        request_id="removal",
        status=rollout.ScaleInStatus.PENDING,
        engine_urls=["http://localhost:30000"],
    )
    manager._scale_in_requests[request.request_id] = request

    asyncio.run(manager._scale_in(request))

    engine.unregister_dcs.remote.assert_not_called()
    if coordinator_failure is not None:
        assert request.status is rollout.ScaleInStatus.FAILED
        assert request.removed_engines == []
        assert server.engine_groups == [group]
        assert manager.set_weight_updating(True) is False
        remove_pg.assert_not_called()
        manager._check_and_handle_evictions()
        assert deletion.call_count == 2
        assert manager.set_weight_updating(True) is True
    else:
        assert request.status is rollout.ScaleInStatus.COMPLETED
        assert request.removed_engines == ["group_5_engine_0"]
        deletion.assert_called_once()

    assert deletion.call_args.kwargs["params"] == {"role": "rollout", "rank": 5, "actor_id": "dead-actor"}
    assert server.engine_groups == []
    remove_pg.assert_called_once_with(group.pg[0])
    engine.unregister_from_router.remote.assert_called_once()
