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
        "relax.distributed.ray.rollout",
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
        "relax.distributed.ray.rollout",
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


def test_pool_cleanup_releases_distributed_locks_when_backend_shutdown_fails(monkeypatch):
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
    assert released == ["topology", "weights"]


def test_pool_startup_failure_cleans_partial_engine_and_lock_ownership(monkeypatch):
    events = []

    class PartialBackend:
        def __init__(self, args, binding, *, inference_only):
            assert inference_only
            self.servers = {"policy": object()}
            self.rollout_engine_lock = "topology-lock"
            raise RuntimeError("startup failed after engine creation")

        def _shutdown_all_engines(self):
            events.append("engines-stopped")

    monkeypatch.setitem(
        sys.modules,
        "relax.distributed.ray.rollout",
        SimpleNamespace(
            RolloutManager=SimpleNamespace(__ray_metadata__=SimpleNamespace(modified_class=PartialBackend)),
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
