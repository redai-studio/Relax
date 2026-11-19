# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from enum import Enum
from threading import RLock
from types import SimpleNamespace

import pytest
import ray

from relax.distributed.ray.inference_pool import RolloutPool, plan_rollout
from relax.engine.inference.config import EngineGroupSpec, InferenceModelSpec
from relax.engine.inference.placement import PlacementPlanner, ProbedPlacementGroup
from relax.engine.inference.types import LifecycleState


class GroupState(Enum):
    ACTIVE = "active"
    DRAINING = "draining"


def actor(identity):
    return SimpleNamespace(_actor_id=SimpleNamespace(hex=lambda: identity))


@pytest.fixture
def pool(monkeypatch):
    def metadata(group, slot, engine, record, states):
        if engine is None:
            return None, "ABSENT", None
        return {"url": "http://engine", "pid": 1, "node_id": "node"}, "ALIVE", None

    monkeypatch.setitem(
        sys.modules,
        "relax.distributed.ray.policy_engine_pool",
        SimpleNamespace(
            EngineGroupLifecycle=GroupState,
            _read_engine_metadata=metadata,
        ),
    )
    group = SimpleNamespace(
        worker_type="regular",
        all_engines=[actor("original")],
        rank_offset=0,
        nodes_per_engine=1,
        is_scaled_out=False,
        lifecycle_status=GroupState.ACTIVE,
        engine_init_records={0: SimpleNamespace(pending_router_registration=False)},
    )
    result = RolloutPool.__new__(RolloutPool)
    result.backend = SimpleNamespace(
        _engine_lifecycle_lock=RLock(),
        _engine_actor_states=None,
        _training_weight_updating=False,
        _scale_out_weight_updating=False,
        servers={"default": SimpleNamespace(engine_groups=[group], router_ip="router", router_port=1)},
    )
    result.synced_actor_ids = {"original"}
    return result


def observe(pool):
    return pool.observe(ready=True, state=LifecycleState.READY)[0]


def test_recovered_process_is_not_ready_until_its_weights_are_synced(pool):
    assert observe(pool).admission
    pool.backend.servers["default"].engine_groups[0].all_engines[0] = actor("replacement")
    assert not observe(pool).admission
    assert observe(pool).issues == (("default/engine-0", "weights_pending"),)
    pool.mark_synced({"original"})
    assert not observe(pool).admission
    pool.mark_synced({"replacement"})
    assert observe(pool).admission


def test_router_registration_and_scale_sync_are_readiness_requirements(pool):
    group = pool.backend.servers["default"].engine_groups[0]
    group.is_scaled_out = True
    group.engine_init_records[0].pending_router_registration = True
    assert not observe(pool).admission
    group.engine_init_records[0].pending_router_registration = False
    pool.backend._scale_out_weight_updating = True
    assert not observe(pool).admission
    pool.backend._scale_out_weight_updating = False
    assert observe(pool).admission
    group.lifecycle_status = GroupState.DRAINING
    assert not observe(pool).admission


def test_unknown_or_dead_engine_does_not_get_admission(pool):
    pool.backend.servers["default"].engine_groups[0].all_engines[0] = None
    assert not observe(pool).admission


def test_multinode_replica_requires_every_node_to_be_confirmed(pool):
    group = pool.backend.servers["default"].engine_groups[0]
    group.nodes_per_engine = 2
    group.all_engines.append(None)
    group.engine_init_records[1] = SimpleNamespace(pending_router_registration=False)
    assert not observe(pool).admission
    group.all_engines[1] = actor("second-node")
    assert not observe(pool).admission
    pool.mark_synced({"second-node"})
    assert observe(pool).admission


def test_planning_uses_original_probes_without_allocating_gpu_actors(monkeypatch):
    config = InferenceModelSpec("default", engine_groups=[EngineGroupSpec("regular", 2, 2)])
    monkeypatch.setitem(
        sys.modules,
        "relax.distributed.ray.policy_engine_pool",
        SimpleNamespace(
            _resolve_sglang_config=lambda args: SimpleNamespace(models=[config]),
        ),
    )
    pg = SimpleNamespace(bundle_count=2)
    binding = ProbedPlacementGroup(pg, [1, 0], [4, 7], ("node", "node"), (0, 1))
    restored = ray.cloudpickle.loads(ray.cloudpickle.dumps(binding))
    assert len(restored) == 3
    assert restored.node_ids == ("node", "node")
    planner = PlacementPlanner()
    plan_rollout(
        SimpleNamespace(
            num_gpus_per_node=2,
            rollout_num_gpus_per_engine=2,
            sglang_hf_checkpoint=None,
            hf_checkpoint="model",
        ),
        restored,
        planner,
    )
    assert planner.allocations()[0].bundle_indices == (1,)
    assert restored.local_gpu_ids == (0, 1)
    assert planner.allocations()[0].gpu_ids == (4,)
    assert planner.allocations()[0].reserved_size == 2
    assert planner.allocations()[0].gpus_per_slot == 2


def test_pool_cleanup_preserves_distributed_locks_when_backend_shutdown_fails(monkeypatch):
    pool = RolloutPool.__new__(RolloutPool)

    def failed_shutdown():
        raise RuntimeError("engine shutdown failed")

    pool.backend = SimpleNamespace(
        dispose=failed_shutdown, rollout_engine_lock="topology", _weight_sync_lock="weights"
    )
    released = []
    monkeypatch.setattr(ray, "kill", released.append)
    with pytest.raises(RuntimeError, match="engine shutdown failed"):
        pool.dispose()
    assert released == []


def test_pool_startup_failure_cleans_partial_engine_and_lock_ownership(monkeypatch):
    events = []

    class PartialBackend:
        def __init__(self, args, binding):
            self.servers = {"policy": object()}
            self.rollout_engine_lock = "topology-lock"
            raise RuntimeError("startup failed after engine creation")

        def _shutdown_all_engines(self):
            events.append("engines-stopped")

    monkeypatch.setitem(
        sys.modules,
        "relax.distributed.ray.policy_engine_pool",
        SimpleNamespace(
            PolicyEnginePool=PartialBackend,
            stop_launched_routers=lambda: events.append("routers-stopped"),
        ),
    )
    monkeypatch.setattr(ray, "kill", lambda actor: events.append(f"killed:{actor}"))
    with pytest.raises(RuntimeError, match="startup failed"):
        RolloutPool(SimpleNamespace(), None)
    assert set(events) == {"engines-stopped", "routers-stopped", "killed:topology-lock"}


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_secondary", [False, True])
async def test_manager_recovery_confirms_captured_multinode_generation(pool, replace_secondary):
    from relax.distributed.ray.inference_manager import InferenceManager

    group = pool.backend.servers["default"].engine_groups[0]
    group.nodes_per_engine = 2
    group.all_engines = [actor("recovered-head"), actor("recovered-secondary")]
    group.engine_init_records[1] = SimpleNamespace(pending_router_registration=False)
    pool.backend.recover_rollout_engines = lambda: ([group.all_engines[0]], None, 1, [], [])
    owner = InferenceManager.__ray_metadata__.modified_class()
    owner.pool = pool
    owner.memory_ready = True
    owner.state = LifecycleState.READY
    assert not (await owner.snapshot()).models[0].admission
    await owner.pool_call("recover_rollout_engines")
    token = await owner.begin_weight_update("test")
    assert not (await owner.snapshot()).models[0].admission
    if replace_secondary:
        group.all_engines[1] = actor("later-secondary")
    await owner.end_weight_update(token)
    assert (await owner.snapshot()).models[0].admission is (not replace_secondary)
    assert "recovered-secondary" in pool.synced_actor_ids
    assert "later-secondary" not in pool.synced_actor_ids


def test_direct_engine_readiness_does_not_require_or_publish_router(pool):
    pool.backend.route_mode = "direct"
    server = pool.backend.servers["default"]
    server.router_ip = server.router_port = None
    snapshot = observe(pool)
    assert snapshot.state is LifecycleState.READY
    assert snapshot.admission
    assert snapshot.router_url is None
    assert snapshot.replicas[0].direct_eligible
    pool.synced_actor_ids.clear()
    assert not observe(pool).admission
    assert not observe(pool).replicas[0].direct_eligible


def test_pd_observation_has_one_service_replica_and_diagnostic_workers(pool):
    from copy import copy

    prefill = pool.backend.servers["default"].engine_groups[0]
    prefill.worker_type = "prefill"
    decode = copy(prefill)
    decode.worker_type = "decode"
    decode.rank_offset = 1
    decode.all_engines = [actor("decode")]
    pool.synced_actor_ids.add("decode")
    pool.backend.servers["default"].engine_groups.append(decode)
    snapshot = observe(pool)
    assert snapshot.admission
    assert [replica.engine_id for replica in snapshot.replicas] == ["default/pd-service"]
    assert snapshot.replicas[0].base_url == snapshot.router_url
    assert not snapshot.replicas[0].direct_eligible
    assert {kind for kind, _ in snapshot.pd_workers} == {"prefill", "decode"}
    pool.synced_actor_ids.remove("decode")
    assert not observe(pool).admission


@pytest.mark.asyncio
async def test_dynamic_versions_require_all_nodes_and_preserve_topology_revision(pool):
    from relax.distributed.ray.inference_manager import InferenceManager

    group = pool.backend.servers["default"].engine_groups[0]
    group.nodes_per_engine = 2
    group.all_engines.append(actor("secondary"))
    group.engine_init_records[1] = SimpleNamespace(pending_router_registration=False)
    pool.requires_weight_confirmation = True
    pool.actor_weight_versions = {}
    owner = InferenceManager.__ray_metadata__.modified_class()
    owner.pool = pool
    owner.memory_ready = True
    owner.state = LifecycleState.READY
    owner._recipients = {"original", "secondary"}
    assert not (await owner.snapshot()).models[0].admission
    token = await owner.begin_weight_update("backend")
    await owner.end_weight_update(token, "transport-42")
    initial = await owner.snapshot()
    assert initial.models[0].required_weight_version == "transport-42"
    assert initial.models[0].replicas[0].weight_version == "transport-42"
    assert initial.models[0].admission
    pool.actor_weight_versions["secondary"] = "transport-41"
    mismatch = await owner.snapshot()
    assert not mismatch.models[0].admission
    assert mismatch.models[0].replicas[0].weight_version is None
    assert mismatch.topology_revision == initial.topology_revision
    token = await owner.begin_weight_update("backend")
    await owner.end_weight_update(token, "transport-43")
    updated = await owner.snapshot()
    assert updated.models[0].replicas[0].weight_version == "transport-43"
    assert updated.topology_revision == initial.topology_revision
    group.all_engines[1] = actor("replacement")
    pending = await owner.snapshot()
    assert not pending.models[0].admission
    assert pending.topology_revision == updated.topology_revision
    pool.mark_synced({"replacement"}, "transport-43")
    recovered = await owner.snapshot()
    assert recovered.models[0].admission
    assert recovered.topology_revision > pending.topology_revision


@pytest.mark.asyncio
async def test_scaled_out_confirmation_requires_transport_receipt_for_exact_generation(pool, monkeypatch):
    from relax.distributed.ray import inference_manager

    group = pool.backend.servers["default"].engine_groups[0]
    group.is_scaled_out = True
    group.nodes_per_engine = 2
    group.all_engines.append(actor("secondary"))
    group.engine_init_records[1] = SimpleNamespace(pending_router_registration=False)
    actual = {"version": "transport-6"}
    group.all_engines[0].get_weight_version = SimpleNamespace(remote=lambda: actual["version"])
    pool.requires_weight_confirmation = True
    pool.actor_weight_versions = {}
    owner = inference_manager.InferenceManager.__ray_metadata__.modified_class()
    owner.pool = pool
    owner.memory_ready = True
    owner.state = LifecycleState.READY
    owner.required_weight_version = "transport-7"
    monkeypatch.setattr(inference_manager.ray, "get", lambda refs, **kwargs: refs)
    await owner._confirm_scaled_out_versions()
    assert pool.actor_weight_versions == {}
    assert not (await owner.snapshot()).models[0].admission
    actual["version"] = "transport-7"
    await owner._confirm_scaled_out_versions()
    assert pool.actor_weight_versions == {}  # A head version is not a transport receipt.
    pool.backend._seed_sync_receipts = {"original": (("original", "secondary"), "transport-7")}
    await owner._confirm_scaled_out_versions()
    assert pool.actor_weight_versions == {"original": "transport-7", "secondary": "transport-7"}
    assert (await owner.snapshot()).models[0].admission
    group.all_engines[1] = actor("replacement")
    await owner._confirm_scaled_out_versions()
    assert "replacement" not in pool.actor_weight_versions
    assert not (await owner.snapshot()).models[0].admission


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_secondary", [False, True])
async def test_dcs_head_receipt_confirms_only_the_pre_sync_whole_replica(pool, replace_secondary):
    from relax.distributed.ray.inference_manager import InferenceManager

    group = pool.backend.servers["default"].engine_groups[0]
    group.nodes_per_engine = 2
    group.all_engines.append(actor("secondary"))
    group.engine_init_records[1] = SimpleNamespace(pending_router_registration=False)
    pool.requires_weight_confirmation = True
    pool.actor_weight_versions = {}
    published = []
    pool.backend.register_recovered_engines = lambda ids: published.append(set(ids))
    owner = InferenceManager.__ray_metadata__.modified_class()
    owner.pool = pool
    owner.memory_ready = True
    owner.state = LifecycleState.READY
    token = await owner.begin_weight_update("dcs")
    if replace_secondary:
        group.all_engines[1] = actor("replacement")
    await owner.pool_call("register_recovered_engines", ["original"], weight_version="dcs-5")
    await owner.end_weight_update(token, "dcs-5")
    model = (await owner.snapshot()).models[0]
    if replace_secondary:
        assert published == [set()]
        assert pool.actor_weight_versions == {}
        assert not model.admission
    else:
        assert published == [{"original", "secondary"}]
        assert pool.actor_weight_versions == {"original": "dcs-5", "secondary": "dcs-5"}
        assert model.admission
        assert model.replicas[0].weight_version == "dcs-5"
