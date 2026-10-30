# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Scoring that generation leaves for the score phase.

With ``--defer-reward-to-post-process`` a judge that shares bundles with
rollout stays asleep while rollout generates; with ``--opd-teacher-defer`` so
does the managed OPD teacher. Once a batch is complete, and before it is
converted and published for training, ``run_deferred_scoring`` enters the
score phase, fills in the rewards that are still missing with the configured
reward function, asks the teacher for its log-probs, and leaves the phase
again. A batch whose scoring fails is not published.

A run that passes ``--custom-reward-post-process-path`` owns the reward swap
itself (see ``examples/generate_reward_model/post_process_genrm_swap.py``); the
framework then leaves rewards alone.
"""

from __future__ import annotations

import asyncio
import json
import weakref
from typing import Any, Iterable

from relax.engine.inference.lifecycle import (
    LifecycleCoordinator,
    local_rollout_participant,
    manager_participant,
)
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# One batch is scored at a time: two batches sharing the score phase would
# have the first one to finish put the judge to sleep under the other.
_SCORING_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = weakref.WeakKeyDictionary()


def is_framework_deferred_reward(args: Any) -> bool:
    """Whether reward computation is left for the score phase and run by the
    framework, rather than by a user-supplied post-process hook."""
    return bool(getattr(args, "defer_reward_to_post_process", False)) and (
        getattr(args, "custom_reward_post_process_path", None) is None
    )


def is_deferred_teacher(args: Any) -> bool:
    """Whether the managed OPD teacher is asked in the score phase, after a
    batch has been generated, instead of while it is generated."""
    return (
        bool(getattr(args, "opd_teacher_defer", False))
        and bool(getattr(args, "use_opd", False))
        and getattr(args, "opd_type", None) == "sglang"
    )


def validate_deferred_scoring_args(args: Any) -> None:
    """Reject a run the framework cannot defer scoring for.

    Raises:
        ValueError: the rollout path scores somewhere the deferral does not
            reach.
    """
    deferred_reward = is_framework_deferred_reward(args)
    if deferred_reward and getattr(args, "dynamic_sampling_filter_path", None) is not None:
        raise ValueError(
            "--defer-reward-to-post-process without --custom-reward-post-process-path is not supported with "
            "--dynamic-sampling-filter-path: the filter decides on a prompt group as soon as it has generated, "
            "but its rewards only exist after the batch has been scored. Drop one of the two flags."
        )
    if (
        is_deferred_teacher(args)
        and getattr(args, "opd_kl_coef", 0)
        and getattr(args, "opd_token_selection", None) in {"teacher_topk", "union"}
    ):
        batch_size = getattr(args, "rollout_batch_size", 0)
        oversampling = getattr(args, "over_sampling_batch_size", None)
        partial = getattr(args, "partial_rollout", False)
        # The partial-rollout path masks every carried response token before
        # its completed-sample fast path or continuation. Only new tokens then
        # contribute to OPD, so their student scores use the current policy.
        masks_carryover = partial and getattr(args, "mask_offpolicy_in_partial_rollout", False)
        if (
            partial
            or (oversampling is not None and oversampling > batch_size)
            or getattr(args, "dynamic_sampling_filter_path", None) is not None
        ) and not masks_carryover:
            raise ValueError(
                "--opd-teacher-defer with teacher_topk/union in advantage mode requires fresh samples from one "
                "student policy: disable --partial-rollout and --dynamic-sampling-filter-path, and set "
                "--over-sampling-batch-size equal to --rollout-batch-size, or enable --partial-rollout with "
                "--mask-offpolicy-in-partial-rollout. Buffered samples may cross a weight "
                "update before deferred student prefill; their generation-time student probabilities would "
                "then be combined with scores from a different policy."
            )
    if not getattr(args, "use_agentic_rollout", False):
        return
    if deferred_reward:
        raise ValueError(
            "--defer-reward-to-post-process without --custom-reward-post-process-path is not supported with "
            "--use-agentic-rollout: the agentic pipeline scores inside its own stages, so the judge would be "
            "asked while it is asleep. Drop the flag, or pass a custom post-process function that owns the swap."
        )
    if is_deferred_teacher(args):
        raise ValueError(
            "--opd-teacher-defer is not supported with --use-agentic-rollout: the agentic pipeline asks the "
            "teacher inside its own stages, while the teacher would be asleep. Drop the flag."
        )


def _samples(group: Any) -> list:
    if not isinstance(group, list):
        return [group]
    return [sample for item in group for sample in _samples(item)]


def _pending_groups(groups: Iterable[Any]) -> list:
    groups = list(groups)
    if groups and not isinstance(groups[0], list):
        groups = [groups]
    return [group for group in groups if any(sample.reward is None for sample in _samples(group))]


async def _score_group(args: Any, group: list) -> None:
    """Compute what the inline reward step would have, for one prompt group."""
    # Deferred: the reward package starts its executor machinery on import.
    from relax.engine.rewards import async_rm, batched_async_rm

    if args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward
        return

    async def score(item: Any) -> None:
        if isinstance(item, list):
            # A multi-sample generation; some of its rewards may have been
            # assigned while it generated.
            missing = [sample for sample in item if sample.reward is None]
            rewards = await batched_async_rm(args, missing)
            for sample, reward in zip(missing, rewards, strict=False):
                sample.reward = reward
        elif item.reward is None:
            item.reward = await async_rm(args, item)

    await asyncio.gather(*(score(item) for item in group))


def _teacher_manager_actor_names(args: Any) -> dict[str, str]:
    """``{model: actor name}`` of the managed teachers sharing the actor pool."""
    if not getattr(args, "use_opd", False):
        return {}
    # Deferred: opd_utils pulls in torch.
    from relax.distributed.ray.placement_planner import DEFAULT_MODEL_KEY
    from relax.utils.opd.opd_utils import TEACHER_MANAGER_ACTOR_NAME, is_managed_opd_teacher_colocate

    if not is_managed_opd_teacher_colocate(args):
        return {}
    routes_json = getattr(args, "opd_teacher_routes", None)
    if routes_json is None:
        return {DEFAULT_MODEL_KEY: TEACHER_MANAGER_ACTOR_NAME}
    return {key: f"{TEACHER_MANAGER_ACTOR_NAME}_{key}" for key in json.loads(routes_json)}


def build_local_coordinator(args: Any) -> LifecycleCoordinator:
    """Coordinator for switches driven from inside the rollout manager process.

    Rollout is switched in-process; the judges and teachers through their
    managers, found by actor name.
    """
    import ray

    from relax.distributed.ray.placement_group import genrm_manager_actor_name
    from relax.distributed.ray.placement_planner import GENRM_ROLE, TEACHER_ROLE, plan_placement
    from relax.distributed.ray.rollout import get_local_rollout_manager

    participants = [local_rollout_participant(get_local_rollout_manager())]
    instance_keys = list(getattr(args, "_genrm_instances_resolved", None) or {})
    for key in instance_keys:
        manager = ray.get_actor(genrm_manager_actor_name(instance_keys, key))
        participants.append(manager_participant(GENRM_ROLE, key, manager))
    for key, actor_name in _teacher_manager_actor_names(args).items():
        participants.append(manager_participant(TEACHER_ROLE, key, ray.get_actor(actor_name)))
    # The layout was validated when the run started; here only residency matters.
    return LifecycleCoordinator(plan_placement(args, validate=False), participants)


def _scoring_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _SCORING_LOCKS.get(loop)
    if lock is None:
        lock = _SCORING_LOCKS[loop] = asyncio.Lock()
    return lock


async def run_deferred_scoring(
    args: Any,
    groups: Iterable[Any],
    *,
    evaluation: bool = False,
    coordinator: LifecycleCoordinator | None = None,
    opd_manager: Any = None,
) -> None:
    """Do the scoring that generation left for the score phase.

    Fills in missing rewards (framework-deferred GenRM scoring) and the
    teacher's log-probs (``--opd-teacher-defer``). Does nothing when neither is
    active or nothing is left to score.

    The teacher is asked once rollout has released the bundles they share. If
    the token selection also needs the student on the teacher's top-k tokens,
    that runs afterwards, with the teacher asleep again and rollout back.

    Args:
        groups: Finished prompt groups (or one flat list of samples). Results
            are written onto the samples.
        evaluation: The batch is an evaluation: no distillation, and rollout is
            brought back afterwards because generation may follow right away.
            A training batch does not need that: the next weight sync wakes
            rollout anyway.
        coordinator: Lifecycle coordinator to switch with; built from ``args``
            when omitted.
        opd_manager: OPD manager to run the teacher stages with; built from
            ``args`` when omitted.

    Raises:
        Exception: a phase switch, the reward function or the teacher failed.
            Results may then be partly written; the caller must not publish
            the batch.
    """
    groups = list(groups)
    reward_groups = _pending_groups(groups) if is_framework_deferred_reward(args) else []
    teacher_samples = _samples(groups) if is_deferred_teacher(args) and not evaluation else []
    if not reward_groups and not teacher_samples:
        return

    async with _scoring_lock():
        if coordinator is None:
            coordinator = build_local_coordinator(args)
        if teacher_samples and opd_manager is None:
            from relax.engine.rollout.on_policy_distillation import OpdManager

            opd_manager = OpdManager(args)
        needs_student = bool(teacher_samples) and opd_manager.needs_student_stage

        logger.info(
            f"Deferred scoring: entering the score phase "
            f"({len(reward_groups)} group(s) to reward, {len(teacher_samples)} sample(s) for the teacher)"
        )
        # The switches block on engine calls; keep them off the event loop.
        await asyncio.to_thread(coordinator.enter_score)
        scored = False
        try:
            await asyncio.gather(*(_score_group(args, group) for group in reward_groups))
            if teacher_samples:
                await opd_manager.teacher_stage(teacher_samples)
            scored = True
        finally:
            # Also on failure: a scorer left awake would sit in the GPU memory
            # the next phase needs.
            await asyncio.to_thread(coordinator.leave_score)
            if evaluation or (scored and needs_student):
                await asyncio.to_thread(coordinator.enter_generate)
        logger.info("Deferred scoring: left the score phase")

        if teacher_samples:
            if needs_student:
                # Deferred: the rollout module needs sglang.
                from relax.engine.rollout.sglang_rollout import _encode_multimodal_inputs

                await opd_manager.student_stage(teacher_samples, encode_multimodal_inputs=_encode_multimodal_inputs)
            opd_manager.assemble_stage(teacher_samples)
