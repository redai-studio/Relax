# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Derive shared inference phases from existing layout flags.

Separate phases describe deferred scorers that reuse generation bundles. Split
layouts keep scoring concurrently in the generation phase.
"""

from typing import Any


# Planner phase labels. Rollout engines use the planner's default phase.
PHASE_GENERATE = "inference"
PHASE_GENRM = "genrm"
PHASE_TEACHER = "teacher"


def genrm_shares_rollout_bundles(args: Any) -> bool:
    """Shared placement requires a GenRM phase, independent of scoring
    policy."""
    return bool(getattr(args, "_genrm_instances_resolved", None)) and bool(
        getattr(args, "_genrm_colocate_with_rollout", False)
    )


def deferred_genrm_enabled(args: Any) -> bool:
    """Whether the workload automatically stages shared GenRM scoring."""
    return genrm_shares_rollout_bundles(args) and bool(getattr(args, "defer_reward_to_post_process", False))


def uses_managed_genrm_hook(args: Any) -> bool:
    """An opted-in hook scores samples while the coordinator owns GPU
    residency."""
    from relax.utils.misc import load_function

    path = getattr(args, "custom_reward_post_process_path", None)
    return bool(path and getattr(load_function(path), "managed_genrm_scoring", False))


def deferred_opd_enabled(args: Any) -> bool:
    """Whether the managed OPD teacher scores in its own stage.

    Only a teacher that shares the rollout bundles has to wait for the
    student's memory. A teacher in its own slice or placement group is a split
    layout and keeps scoring inline, concurrently with generation.
    """
    from relax.utils.opd.opd_utils import is_managed_opd_teacher_enabled, teacher_shares_rollout_bundles

    return is_managed_opd_teacher_enabled(args) and teacher_shares_rollout_bundles(args)


def placement_phase(args: Any, phase_id: str) -> str:
    """The planner phase a role's engines occupy."""
    deferred = {PHASE_GENRM: genrm_shares_rollout_bundles, PHASE_TEACHER: deferred_opd_enabled}
    return phase_id if deferred[phase_id](args) else PHASE_GENERATE
