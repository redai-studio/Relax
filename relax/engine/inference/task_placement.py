# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate a complete task layout before starting any inference engines."""

from dataclasses import dataclass
from typing import Any, Sequence

from relax.engine.inference.config import InferenceRoleSpec
from relax.engine.inference.placement import PlacementGroupView, PlacementOwner, PlacementPlanner, PlacementRequest


@dataclass(frozen=True)
class StaticPlacement:
    spec: InferenceRoleSpec
    binding: Any
    bundle_offset: int = 0
    owner: PlacementOwner = PlacementOwner.CONTROLLER


def plan_task_placement(args: Any, rollout_binding: Any, static: Sequence[StaticPlacement]) -> PlacementPlanner:
    """Use one planner for all roles, including shared phases and split slices.

    Bindings are probed placement metadata; this function neither starts
    engines nor acquires placement groups. A rejected plan cannot publish
    partial state.
    """
    from relax.distributed.ray.inference_pool import plan_rollout

    planner = PlacementPlanner()
    plan_rollout(args, rollout_binding, planner)
    for item in static:
        plan_static_placement(args, item, planner)
    return planner


def plan_static_placement(
    args: Any, item: StaticPlacement, planner: PlacementPlanner, *, dry_run: bool = False
) -> None:
    """Validate a role spec against the task ledger before constructing
    engines."""
    pg, indices, gpu_ids = item.binding
    view = PlacementGroupView(
        tuple(indices),
        tuple(int(gpu) for gpu in gpu_ids),
        item.owner,
        identity=pg,
        node_ids=getattr(item.binding, "node_ids", None),
        local_gpu_ids=getattr(item.binding, "local_gpu_ids", None),
        bundle_count=getattr(pg, "bundle_count", None),
    )
    offset = item.bundle_offset
    for model in item.spec.models:
        for index, group in enumerate(model.engine_groups):
            planner.plan(
                [
                    PlacementRequest(
                        f"{item.spec.role.value}/{model.model_id}/{index}",
                        group.worker_type,
                        group.num_gpus,
                        group.num_gpus_per_engine,
                        args.num_gpus_per_node,
                        bundle_offset=offset,
                        phase=item.spec.deployment.phase,
                        active=group.worker_type != "placeholder",
                    )
                ],
                view,
                dry_run=dry_run,
            )
            offset += group.num_gpus
