# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""``RolloutManager.get_inference_snapshot``: the rollout role's topology in
the unified discovery schema."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


try:
    from relax.distributed.ray import rollout
    from relax.distributed.ray.rollout import EngineGroupLifecycle

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from conftest import (
    AwaitableValue,
    create_test_manager,
    make_engine_group,
    make_mock_engine,
    make_rollout_server,
    mock_ray_get,
)

from relax.engine.inference.discovery import EngineState, RoleSnapshot


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


def _manager(monkeypatch, groups, status=None):
    monkeypatch.setattr(rollout.ray, "get", mock_ray_get)
    for group in groups:
        group._engine_urls = {
            slot: mock_ray_get(engine.get_url.remote.return_value)
            for slot, engine in enumerate(group.all_engines)
            if engine is not None
        }
        from ray.core.generated.gcs_pb2 import ActorTableData

        for slot, url in group._engine_urls.items():
            group.all_engines[slot]._get_local_state.return_value = ActorTableData.ALIVE
            group.engine_init_records.setdefault(
                slot, rollout._EngineInitRecord({"url": url, "pid": 1, "node_id": "node"})
            )
    manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=groups)})
    manager.status = status
    manager.rollout_engine_lock = object()
    return manager


def _snapshot(manager) -> RoleSnapshot:
    return RoleSnapshot.from_dict(manager.get_inference_snapshot())


def _engines(*urls):
    return [make_mock_engine(url=url) for url in urls]


def test_rollout_snapshot_reports_router_and_logical_replicas(monkeypatch):
    single_node = make_engine_group(engines=_engines("http://n0:15000", "http://n0:15010"), num_gpus_per_engine=2)
    # 16 GPUs per engine on 8-GPU nodes: head and follower slots alternate.
    multi_node = make_engine_group(
        engines=_engines("http://n1:15000", "http://n2:15000", "http://n3:15000", "http://n4:15000"),
        num_gpus_per_engine=16,
        rank_offset=2,
    )
    snapshot = _snapshot(_manager(monkeypatch, [single_node, multi_node]))

    assert (snapshot.role, snapshot.default_model) == ("rollout", "default")
    model = snapshot.model("default")
    assert model.router_url == "http://127.0.0.1:3000"
    assert model.state.value == "ready"
    assert [(engine.engine_id, engine.base_url) for engine in model.engines] == [
        ("default/0", "http://n0:15000"),
        ("default/1", "http://n0:15010"),
        ("default/2", "http://n1:15000"),
        ("default/4", "http://n3:15000"),
    ]
    # Requests go through the router, never straight to a replica.
    assert not any(engine.direct_eligible for engine in model.engines)


def test_rollout_snapshot_reports_actual_served_name_from_engine_init(monkeypatch):
    group = make_engine_group(engines=_engines("http://engine:1"))
    group._served_model_names[0] = "/materialized/model"
    assert _snapshot(_manager(monkeypatch, [group])).models[0].served_model_name == "/materialized/model"


def test_rollout_observation_cannot_rewrite_lifecycle_intent(monkeypatch):
    group = make_engine_group(engines=_engines("http://engine:1"))
    manager = _manager(monkeypatch, [group], status="onload")
    pool = group._engine_pool()
    pool._active = False
    manager._admission.close()
    for _ in range(5):
        assert _snapshot(manager).models[0].state is not EngineState.READY
        assert pool.is_active() is False


@pytest.mark.parametrize("confirmed", [False, True])
def test_rollout_recovers_or_isolates_one_failed_slot_while_survivor_serves(monkeypatch, confirmed):
    from relax.engine.inference.routing import RoutingState, select_target

    group = make_engine_group(engines=_engines("http://old:1", "http://survivor:2"))
    group.pg = (object(), [0, 1], [0, 1])
    group.router_ip = group.router_port = None
    manager = _manager(monkeypatch, [group], status="onload")
    manager.servers["default"].router_ip = manager.servers["default"].router_port = None
    pool = group._engine_pool()
    group.health_failures = {0: pool.incarnations.get(0, 0)}
    failed = object()
    group.all_engines[0].shutdown.remote.return_value = failed

    def get(ref, **kwargs):
        if ref is failed:
            raise RuntimeError("actor died")
        return mock_ray_get(ref, **kwargs)

    monkeypatch.setattr(rollout.ray, "get", get)
    monkeypatch.setattr(rollout.ray, "kill", lambda *a, **kw: None)
    monkeypatch.setattr(pool, "_confirm_orphan_release", lambda slot: confirmed)

    def start(cursors):
        group.num_new_engines = int(group.all_engines[0] is None)
        if group.num_new_engines:
            group.all_engines[0] = make_mock_engine(url="http://new:3")
            pool._states[0] = EngineState.READY
            pool.incarnations[0] = 1
            group.engine_init_records[0] = rollout._EngineInitRecord(
                {"url": "http://new:3", "pid": 3, "node_id": "node"}
            )
        return [], cursors

    monkeypatch.setattr(group, "start_engines", start)
    manager.recover_rollout_engines()
    snapshot = _snapshot(manager)
    assert snapshot.models[0].state is EngineState.READY
    urls = {select_target(snapshot, RoutingState()).base_url for _ in range(5)}
    assert "http://old:1" not in urls
    assert group.all_engines[0] is not None
    assert snapshot.models[0].engines[0].state is (EngineState.READY if confirmed else EngineState.BLOCKED)


def test_rollout_failed_route_is_removed_using_init_record(monkeypatch):
    from ray.core.generated.gcs_pb2 import ActorTableData

    engine = make_mock_engine(url="http://old:1")
    engine._get_local_state.return_value = ActorTableData.DEAD
    group = make_engine_group(engines=[engine])
    group.engine_init_records[0] = rollout._EngineInitRecord(
        {"url": "http://old:1", "pid": 1, "node_id": "node"},
        router_cleanup=rollout.RouterWorkerCleanup("old-generation"),
    )
    removed = []
    monkeypatch.setattr(
        rollout,
        "remove_dead_router_worker",
        lambda router, url, cleanup, **kw: removed.append((url, cleanup.actor_id)) or True,
    )
    monkeypatch.setattr(rollout.ray, "get", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("actor died")))
    assert group.remove_failed_route(0)
    assert removed == [("http://old:1", "old-generation")]
    engine.get_url.remote.assert_not_called()


async def test_rollout_split_resume_keeps_admission_closed_until_all_tags_return(monkeypatch):
    from relax.engine.inference.routing import ModelUnavailableError

    group = make_engine_group(engines=_engines("http://engine:1"))
    manager = _manager(monkeypatch, [group], status="onload")
    await manager.offload()
    await manager.onload_kv()
    with pytest.raises(ModelUnavailableError):
        manager.acquire_inference("too-early")
    await manager.onload_weights()
    permit = manager.acquire_inference("ready")
    manager.release_inference(permit)
    assert manager.status == "onload"


@pytest.mark.parametrize("fail", [False, True])
async def test_rollout_partial_resume_is_released_before_next_owner(monkeypatch, fail):
    group = make_engine_group(engines=_engines("http://engine:1", "http://engine:2"))
    manager = _manager(monkeypatch, [group], status="onload")
    await manager.offload()
    if fail:
        failed_ref = object()
        group.all_engines[1].resume_memory_occupation.remote.return_value = failed_ref

        def get(ref, timeout=None):
            if ref is failed_ref:
                raise RuntimeError("partial onload")
            return mock_ray_get(ref, timeout)

        monkeypatch.setattr(rollout.ray, "get", get)
        with pytest.raises(RuntimeError, match="partial onload"):
            await manager.onload_weights()
    else:
        await manager.onload_weights()
    await manager.offload()
    for engine in group.all_engines:
        assert engine.release_memory_occupation.remote.call_count == 2
    assert _snapshot(manager).models[0].state.value == "sleeping"


def test_rollout_snapshot_excludes_pd_workers_from_replicas(monkeypatch):
    prefill = make_engine_group(engines=_engines("http://p0:15000"), worker_type="prefill")
    decode = make_engine_group(engines=_engines("http://d0:15000"), worker_type="decode", rank_offset=1)
    model = _snapshot(_manager(monkeypatch, [prefill, decode])).model("default")

    assert model.engines == ()
    assert model.state.value == "ready"
    assert [(worker.engine_id, worker.base_url) for worker in model.diagnostic_workers] == [
        ("default/prefill-0", "http://p0:15000"),
        ("default/decode-1", "http://d0:15000"),
    ]


def test_rollout_snapshot_state_follows_offload_and_dead_slots(monkeypatch):
    group = make_engine_group(engines=[make_mock_engine(url="http://n0:15000"), None], num_gpus_per_engine=2)
    manager = _manager(monkeypatch, [group])
    resident = _snapshot(manager)

    assert [engine.state.value for engine in resident.model("default").engines] == ["ready", "dead"]
    assert resident.model("default").engines[1].base_url is None

    manager.status = "offload"
    offloaded = _snapshot(manager)

    assert [engine.state.value for engine in offloaded.model("default").engines] == ["sleeping", "dead"]
    assert offloaded.model("default").state.value == "sleeping"
    assert offloaded.topology_revision == resident.topology_revision


def test_rollout_snapshot_tracks_scale_in(monkeypatch):
    base = make_engine_group(engines=_engines("http://n0:15000"), num_gpus_per_engine=2)
    scaled = make_engine_group(
        engines=_engines("http://n1:15000"), num_gpus_per_engine=2, rank_offset=1, is_scaled_out=True
    )
    manager = _manager(monkeypatch, [base, scaled])
    before = _snapshot(manager)

    scaled.lifecycle_status = EngineGroupLifecycle.DRAINING
    draining = _snapshot(manager)
    assert [engine.state.value for engine in draining.model("default").engines] == ["ready", "draining"]
    assert draining.topology_revision == before.topology_revision

    scaled.lifecycle_status = EngineGroupLifecycle.REMOVED
    removed = _snapshot(manager)
    assert [engine.engine_id for engine in removed.model("default").engines] == ["default/0"]
    assert removed.topology_revision == before.topology_revision + 1


def test_rollout_snapshot_leaves_legacy_engines_info_shape(monkeypatch):
    group = make_engine_group(engines=_engines("http://n0:15000"), num_gpus_per_engine=2)
    manager = _manager(monkeypatch, [group])

    legacy = manager.get_engines_info()

    # upstream/main already exposes observation in this legacy projection.
    assert set(legacy) == {"models", "total_engines", "observation"}
    assert set(legacy["models"]["default"]) == {"router_ip", "router_port", "engine_groups", "total_engines"}
    assert legacy["models"]["default"]["engine_groups"][0]["engines"][0]["url"] == "http://n0:15000"


def test_rollout_snapshot_shows_a_memory_switch_in_flight(monkeypatch):
    group = make_engine_group(engines=_engines("http://n0:15000"), num_gpus_per_engine=2)
    manager = _manager(monkeypatch, [group], status="onload")

    def states():
        return [engine.state.value for engine in _snapshot(manager).model("default").engines]

    # The release has been issued but the manager has not recorded it as done.
    group.offload()
    assert states() == ["draining"]
    manager.status = "offload"
    assert states() == ["sleeping"]

    # A partial resume leaves the manager's status alone until the last stage.
    group.onload(tags=["weights"])
    assert states() == ["onloading"]
    manager.status = "onload"
    assert states() == ["ready"]


@pytest.mark.parametrize("switch", ["offload", "onload"])
def test_rollout_snapshot_does_not_wait_for_busy_engines(monkeypatch, switch):
    engine = make_mock_engine(url="http://[2001:db8::1]:18000")
    group = make_engine_group(engines=[engine])
    manager = _manager(monkeypatch, [group], status="onload" if switch == "offload" else "offload")
    if switch == "offload":
        group.offload()
    else:
        group.onload(tags=["weights"])

    def unexpected_wait(*args, **kwargs):
        pytest.fail("Discovery must not wait for a busy engine actor")

    monkeypatch.setattr(rollout.ray, "get", unexpected_wait)
    model = _snapshot(manager).model("default")

    assert model.engines[0].base_url == "http://[2001:db8::1]:18000"
    assert model.state.value == ("draining" if switch == "offload" else "onloading")
    engine.get_url.remote.assert_not_called()


async def test_rollout_snapshot_external_scale_out_caches_only_successful_engines(monkeypatch):
    manager = _manager(monkeypatch, [make_engine_group()], status="onload")
    manager.args.scale_out_partial_success_policy = "keep_successful"
    manager._health_check_engines = AsyncMock(return_value=True)
    manager._sync_weights_from_seed_engine = AsyncMock(return_value=True)
    manager._rollback_engines = AsyncMock()
    engines = _engines("http://failed:18000", "http://second:18001", "http://third:18002")
    engines[0].init.remote.side_effect = RuntimeError("connection failed")
    for engine in engines[1:]:
        engine.init.remote.return_value = AwaitableValue(
            {"url": mock_ray_get(engine.get_url.remote.return_value), "pid": 1, "node_id": "node"}
        )
    actor = MagicMock()
    actor.options.return_value.remote.side_effect = engines
    monkeypatch.setattr(rollout.ray, "remote", lambda cls: actor)
    monkeypatch.setattr(rollout, "get_ray_accelerator_kwargs", lambda count: {})
    request = rollout.ScaleOutRequest(
        request_id="external",
        status=rollout.ScaleOutStatus.PENDING,
        model_name="default",
        engine_urls=["failed:18000", "second:18001", "third:18002"],
    )

    await manager._scale_out_external(request)

    assert request.status is rollout.ScaleOutStatus.ACTIVE
    added = manager.servers["default"].engine_groups[-1]
    assert added._engine_urls == {0: "http://second:18001", 1: "http://third:18002"}
    model = _snapshot(manager).model("default")
    assert [(engine.engine_id, engine.base_url) for engine in model.engines[1:]] == [
        ("default/1", "http://second:18001"),
        ("default/2", "http://third:18002"),
    ]
    engines[0].get_url.remote.assert_not_called()
    for engine in engines[1:]:
        engine.get_url.remote.assert_not_called()


async def test_rollout_snapshot_remains_available_during_scale_out_url_capture(monkeypatch):
    manager = _manager(monkeypatch, [make_engine_group()], status="onload")
    manager._health_check_engines = AsyncMock(return_value=True)
    manager._sync_weights_from_seed_engine = AsyncMock(return_value=True)
    entered, release = asyncio.Event(), asyncio.Event()

    async def initialization():
        entered.set()
        await release.wait()
        return {"url": "http://new:18000", "pid": 2, "node_id": "node"}

    engine = make_mock_engine(url="http://new:18000")
    init = asyncio.create_task(initialization())
    task = asyncio.create_task(
        manager._finalize_engine_group_registration(
            request=rollout.ScaleOutRequest(request_id="external", status=rollout.ScaleOutStatus.CREATING),
            srv=manager.servers["default"],
            engines=[engine],
            engine_init_pairs=[(engine, init)],
            rank_offset=1,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert not task.done()
        # No partially registered group is visible while URL capture is pending.
        assert len(_snapshot(manager).model("default").engines) == 1
    finally:
        release.set()
        result = await task
    assert result.success
    assert _snapshot(manager).model("default").engines[1].base_url == "http://new:18000"


def test_rollout_uses_survivor_directly_until_failed_route_is_removed(monkeypatch):
    from relax.engine.inference.routing import RoutingState, select_target

    group = make_engine_group(engines=_engines("http://old:1", "http://survivor:2"))
    manager = _manager(monkeypatch, [group], status="onload")
    group.health_failures = {0: 0}
    snapshot = _snapshot(manager)
    assert snapshot.models[0].router_url is None
    assert select_target(snapshot, RoutingState()).base_url == "http://survivor:2"
    assert group.remove_failed_route(0)
    assert _snapshot(manager).models[0].router_url == "http://127.0.0.1:3000"


def test_router_fallback_excludes_replacement_awaiting_weight_sync(monkeypatch):
    from relax.engine.inference.routing import RoutingState, select_target

    group = make_engine_group(engines=_engines("http://old:1", "http://unsynced:2", "http://survivor:3"))
    manager = _manager(monkeypatch, [group], status="onload")
    group.health_failures = {0: 0}
    group.engine_init_records[1].pending_router_registration = True
    snapshot = _snapshot(manager)
    routing = RoutingState()
    assert {select_target(snapshot, routing).base_url for _ in range(6)} == {"http://survivor:3"}


def test_rollout_admission_cache_observes_health_and_same_endpoint_replacement(monkeypatch):
    from relax.engine.inference.routing import ModelUnavailableError

    group = make_engine_group(engines=_engines("http://engine:1"))
    manager = _manager(monkeypatch, [group], status="onload")
    builds = []
    build = rollout.build_model_snapshot

    def record(*args, **kwargs):
        builds.append(True)
        return build(*args, **kwargs)

    monkeypatch.setattr(rollout, "build_model_snapshot", record)
    for i in range(4):
        permit = manager.acquire_inference(str(i))
        manager.release_inference(permit)
    assert len(builds) == 1
    before = _snapshot(manager).topology_revision
    group._engine_pool().incarnations[0] = 1
    assert _snapshot(manager).topology_revision > before
    group.health_failures = {0: 1}
    with pytest.raises(ModelUnavailableError):
        manager.acquire_inference("failed")
