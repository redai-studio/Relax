# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Adapt existing auxiliary flags without adding a second user
configuration."""

from typing import Any

from relax.engine.inference.config import DeploymentSpec, EngineGroupSpec, InferenceModelSpec, InferenceRoleSpec
from relax.engine.inference.phase_plans import PHASE_GENERATE, placement_phase
from relax.engine.inference.types import (
    WORKLOAD_BY_ROLE,
    DeploymentMode,
    Role,
    RouteMode,
    RoutingSpec,
    WeightSource,
)


def static_role_spec(role: Role, args: Any, options: dict[str, Any]) -> InferenceRoleSpec:
    if role is Role.GENRM:
        name = options.get("model_id", "default")
        path = args.genrm_model_path
        tp = args.genrm_num_gpus_per_engine
        total = args.genrm_num_gpus
        overrides = dict(args.genrm_engine_config or {})
        sampling = dict(args.genrm_sampling_config or {})
    elif role is Role.TEACHER:
        from relax.utils.opd.opd_utils import build_teacher_overrides

        name = options.get("model_id", "default")
        path = args.teacher_hf_checkpoint
        tp = options["gpus_per_replica"]
        total = options["num_replicas"] * tp
        overrides = build_teacher_overrides(args, colocate_sync=False)
        sampling = {}
    else:
        raise ValueError("Static configuration only supports GenRM and Teacher")
    if not path:
        raise ValueError("Static inference requires a checkpoint")
    model = InferenceModelSpec(
        name,
        model_path=path,
        num_gpus_per_engine=tp,
        engine_groups=[EngineGroupSpec("regular", total, tp, overrides)],
        weight_source=WeightSource.STATIC,
        route_mode=RouteMode.DIRECT,
        sampling_defaults=sampling,
    ).resolved(args)
    phase = placement_phase(args, role.value)
    mode = DeploymentMode.DECOUPLED
    if phase != PHASE_GENERATE:
        mode = DeploymentMode.DEFER
    elif role is Role.GENRM and getattr(args, "colocate", False) and not getattr(args, "hybrid", False):
        mode = DeploymentMode.SPLIT
    elif role is Role.TEACHER:
        from relax.utils.opd.opd_utils import is_managed_opd_teacher_colocate

        if is_managed_opd_teacher_colocate(args):
            mode = DeploymentMode.SPLIT
    return InferenceRoleSpec(
        role,
        WORKLOAD_BY_ROLE[role],
        DeploymentSpec(mode, phase),
        RoutingSpec(default_model=name, route_key_to_model=((name, name),)),
        (model,),
    )
