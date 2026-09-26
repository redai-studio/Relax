# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Which inference roles take turns on shared GPUs, derived from existing
flags.

A role that scores in its own deferred stage (GenRM with ``--defer-reward-to-
post-process``, a managed OPD teacher that shares the rollout bundles) owns a
separate planner phase and may reuse GPUs the generation phase holds; the
Manager switches the two with ``InferenceManager.switch``. Every other role
serves while generation runs, so it shares the generation phase and the
placement planner rejects any GPU it would share with rollout.
"""

from typing import Any


# Planner phase labels. Rollout engines use the planner's default phase.
PHASE_GENERATE = "inference"
PHASE_GENRM = "genrm"
PHASE_TEACHER = "teacher"


def deferred_genrm_enabled(args: Any) -> bool:
    """Whether GenRM scores in its own stage instead of inline with generation.

    Only a GenRM that shares the rollout bundles has to take turns with
    generation. Split and fully-async layouts keep it resident, so a deferred
    post-process hook scores inline without a phase switch.
    """
    return (
        bool(getattr(args, "_genrm_instances_resolved", None))
        and bool(getattr(args, "defer_reward_to_post_process", False))
        and bool(getattr(args, "_genrm_colocate_with_rollout", False))
    )


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
    deferred = {PHASE_GENRM: deferred_genrm_enabled, PHASE_TEACHER: deferred_opd_enabled}
    return phase_id if deferred[phase_id](args) else PHASE_GENERATE


def reject_shared_co_resident(args: Any) -> None:
    """Refuse a GenRM that shares the rollout bundles without deferring.

    The planner rejects the same layout once both roles are placed; this check
    runs before the first engine of the task starts.
    """
    if getattr(args, "_genrm_colocate_with_rollout", False) and not deferred_genrm_enabled(args):
        raise ValueError(
            "GenRM shares the rollout GPUs while both stay resident (shared co-resident), which is not supported. "
            "Split the GPUs (--rollout-num-gpus + GenRM GPUs == actor GPUs) or add --defer-reward-to-post-process."
        )
