# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.engine.inference.config import DeploymentSpec, EngineGroupSpec, InferenceModelSpec, InferenceRoleSpec
from relax.engine.inference.placement import PlacementConflictError, ProbedPlacementGroup
from relax.engine.inference.task_placement import StaticPlacement, plan_task_placement
from relax.engine.inference.types import DeploymentMode, Role, RoutingSpec, WorkloadType


@pytest.mark.parametrize("phase,offset,allowed", [("genrm", 0, True), ("inference", 0, False), ("inference", 2, True)])
def test_task_plan_checks_cross_role_overlap_before_startup(monkeypatch, phase, offset, allowed):
    from relax.distributed.ray import policy_engine_pool as rollout

    monkeypatch.setattr(
        rollout,
        "_resolve_sglang_config",
        lambda args: SimpleNamespace(
            models=[
                SimpleNamespace(
                    name="policy",
                    model_path="policy",
                    num_gpus_per_engine=2,
                    engine_groups=[EngineGroupSpec("regular", 2, 2, {})],
                )
            ]
        ),
    )
    args = SimpleNamespace(num_gpus_per_node=4)
    binding = ProbedPlacementGroup(
        SimpleNamespace(bundle_count=4), list(range(4)), list(range(4)), ("node",) * 4, tuple(range(4))
    )
    spec = InferenceRoleSpec(
        Role.GENRM,
        WorkloadType.REWARD,
        DeploymentSpec(DeploymentMode.DEFER, phase),
        RoutingSpec(default_model="judge"),
        (
            InferenceModelSpec(
                "judge",
                model_path="judge",
                num_gpus_per_engine=2,
                engine_groups=[EngineGroupSpec("regular", 2, 2, {})],
            ).resolved(args),
        ),
    )
    if not allowed:
        with pytest.raises(PlacementConflictError, match="overlap"):
            plan_task_placement(args, binding, [StaticPlacement(spec, binding, offset)])
    else:
        allocations = plan_task_placement(args, binding, [StaticPlacement(spec, binding, offset)]).allocations()
        assert len(allocations) == 2
        assert allocations[1].bundle_indices == (offset,)


@pytest.mark.parametrize("teacher_phase,allowed", [("inference", False), ("teacher", True)])
def test_task_plan_checks_teacher_and_genrm_against_each_other(monkeypatch, teacher_phase, allowed):
    from relax.distributed.ray import policy_engine_pool as rollout

    model = InferenceModelSpec(
        "policy", model_path="policy", num_gpus_per_engine=2, engine_groups=[EngineGroupSpec("regular", 2, 2, {})]
    )
    monkeypatch.setattr(rollout, "_resolve_sglang_config", lambda args: SimpleNamespace(models=[model]))
    args = SimpleNamespace(num_gpus_per_node=4)
    binding = ProbedPlacementGroup(
        SimpleNamespace(bundle_count=4), list(range(4)), list(range(4)), ("node",) * 4, tuple(range(4))
    )

    def static(role, phase):
        name = role.value
        spec = InferenceRoleSpec(
            role,
            WorkloadType.REWARD if role is Role.GENRM else WorkloadType.DISTILLATION,
            DeploymentSpec(DeploymentMode.SPLIT, phase),
            RoutingSpec(default_model=name),
            (
                InferenceModelSpec(
                    name, model_path=name, num_gpus_per_engine=2, engine_groups=[EngineGroupSpec("regular", 2, 2, {})]
                ).resolved(args),
            ),
        )
        return StaticPlacement(spec, binding, 2)

    roles = [static(Role.GENRM, "inference"), static(Role.TEACHER, teacher_phase)]
    if allowed:
        assert len(plan_task_placement(args, binding, roles).allocations()) == 3
    else:
        with pytest.raises(PlacementConflictError, match="overlap"):
            plan_task_placement(args, binding, roles)
