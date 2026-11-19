# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.distributed.ray.static_inference_pool import StaticInferencePool
from relax.engine.inference.placement import PlacementPlanner, ProbedPlacementGroup
from relax.engine.inference.types import Role


@pytest.mark.parametrize(
    "role,shared", [(Role.GENRM, False), (Role.GENRM, True), (Role.TEACHER, False), (Role.TEACHER, True)]
)
def test_static_pool_constructs_shared_backend_and_records_actual_slice(monkeypatch, role, shared):
    from relax.distributed.ray.engine_pool import EnginePool

    args = SimpleNamespace(
        num_gpus_per_node=8,
        colocate=True,
        fully_async=False,
        rollout_num_gpus=4,
        genrm_model_path="judge",
        genrm_num_gpus_per_engine=2,
        genrm_num_gpus=2,
        genrm_engine_config={},
        genrm_sampling_config={},
        _genrm_instances_resolved=[{}] if role is Role.GENRM else [],
        _genrm_colocate_with_rollout=shared and role is Role.GENRM,
        defer_reward_to_post_process=shared,
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="teacher",
        hybrid=False,
        use_critic=False,
        resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 2]},
    )
    if role is Role.TEACHER and shared:
        args.rollout_num_gpus = 8
    binding = ProbedPlacementGroup(
        SimpleNamespace(bundle_count=8), list(range(8)), list(range(8)), ("node",) * 8, tuple(range(8))
    )
    implementation = EnginePool

    def initialize(backend, backend_role, backend_args, binding, options):
        backend.role = backend_role
        backend.args = backend_args
        if role is Role.GENRM:
            backend.pg = binding
            backend.bundle_offset = options.get("bundle_offset", 0)
            backend.dedicated_pg = options["dedicated_pg"]
            backend.num_gpu_per_engine = 2
        else:
            assert options["shared_pg"]
            assert getattr(backend_args, "_requires_weights_cpu_backup", False) is shared
            backend._shared_pg = True
            backend._shared_pg_tuple = options["pg"]
            backend._bundle_offset = options["bundle_offset"]
            backend.nodes_per_engine = 1
            backend.gpus_per_replica = options["gpus_per_replica"]
            backend._engine_placements = {0: (options["pg"], False)}

    monkeypatch.setattr(implementation, "__init__", initialize)
    monkeypatch.setattr(implementation, "offload", lambda backend: None)
    pool = StaticInferencePool(role, args, binding, {"num_replicas": 1, "gpus_per_replica": 2, "bundle_offset": 1})
    planner = PlacementPlanner()
    pool.record_placement(planner)
    allocation = planner.allocations()[0]
    expected = 1 if shared else 5
    assert allocation.bundle_indices == (expected,)
    assert args.rollout_num_gpus == (8 if role is Role.TEACHER and shared else 4)


@pytest.mark.asyncio
async def test_static_start_sleeping_never_health_generates_or_recovers(monkeypatch):
    from relax.distributed.ray import inference_manager as module
    from relax.engine.inference.types import LifecycleState

    def initialize(pool, *args):
        pool.start_sleeping = True

    monkeypatch.setattr(StaticInferencePool, "__init__", initialize)
    monkeypatch.setattr(StaticInferencePool, "record_placement", lambda *args: None)

    def unexpected(*args):
        raise AssertionError("sleeping engines must not be health generated")

    monkeypatch.setattr(StaticInferencePool, "unhealthy_ranks", unexpected)
    monkeypatch.setattr(StaticInferencePool, "recover", unexpected)
    owner = module.InferenceManager.__ray_metadata__.modified_class()
    await owner.start_static("teacher", None, None, {})
    assert owner.state is LifecycleState.SLEEPING
    assert not owner.memory_ready
    assert "memory" in owner.ledger.fences
    await owner.recover_static()
    assert owner.state is LifecycleState.SLEEPING
