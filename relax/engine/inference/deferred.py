# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Scoring that generation leaves for the score phase.

With ``--defer-reward-to-post-process`` a judge that shares bundles with
rollout stays asleep while rollout generates. Once a batch is complete, and
before it is converted and published for training, ``run_deferred_scoring``
enters the score phase, fills in the rewards that are still missing with the
configured reward function, and leaves the phase again. A batch whose scoring
fails is not published.

A run that passes ``--custom-reward-post-process-path`` owns that swap itself
(see ``examples/generate_reward_model/post_process_genrm_swap.py``); the
framework then does none of this.
"""

from __future__ import annotations

import asyncio
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


def validate_deferred_scoring_args(args: Any) -> None:
    """Reject a run the framework cannot defer scoring for.

    Raises:
        ValueError: the rollout path computes its rewards somewhere the
            deferral does not reach.
    """
    if not is_framework_deferred_reward(args):
        return
    if getattr(args, "use_agentic_rollout", False):
        raise ValueError(
            "--defer-reward-to-post-process without --custom-reward-post-process-path is not supported with "
            "--use-agentic-rollout: the agentic pipeline scores inside its own stages, so the judge would be "
            "asked while it is asleep. Drop the flag, or pass a custom post-process function that owns the swap."
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


def build_local_coordinator(args: Any) -> LifecycleCoordinator:
    """Coordinator for switches driven from inside the rollout manager process.

    Rollout is switched in-process; the judges through their managers, found by
    actor name.
    """
    import ray

    from relax.distributed.ray.placement_group import genrm_manager_actor_name
    from relax.distributed.ray.placement_planner import GENRM_ROLE, plan_placement
    from relax.distributed.ray.rollout import get_local_rollout_manager

    participants = [local_rollout_participant(get_local_rollout_manager())]
    instance_keys = list(getattr(args, "_genrm_instances_resolved", None) or {})
    for key in instance_keys:
        manager = ray.get_actor(genrm_manager_actor_name(instance_keys, key))
        participants.append(manager_participant(GENRM_ROLE, key, manager))
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
    restore_rollout: bool = False,
    coordinator: LifecycleCoordinator | None = None,
) -> None:
    """Fill in the rewards that generation left for the score phase.

    Does nothing unless the framework owns deferred scoring and some sample of
    ``groups`` still has no reward.

    Args:
        groups: Finished prompt groups (or one flat list of samples). Rewards
            are written onto the samples.
        restore_rollout: Bring rollout back afterwards, for a caller that may
            be followed by generation right away. A training batch does not
            need it: the next weight sync wakes rollout anyway.
        coordinator: Lifecycle coordinator to switch with; built from ``args``
            when omitted.

    Raises:
        Exception: a phase switch or the reward function failed. Rewards may
            then be partly filled in; the caller must not publish the batch.
    """
    if not is_framework_deferred_reward(args):
        return
    pending = _pending_groups(groups)
    if not pending:
        return

    async with _scoring_lock():
        if coordinator is None:
            coordinator = build_local_coordinator(args)
        logger.info(f"Deferred scoring: entering the score phase for {len(pending)} group(s)")
        # The switches block on engine calls; keep them off the event loop.
        await asyncio.to_thread(coordinator.enter_score)
        try:
            await asyncio.gather(*(_score_group(args, group) for group in pending))
        finally:
            # Also on failure: a judge left awake would sit in the GPU memory
            # the next phase needs.
            await asyncio.to_thread(coordinator.leave_score)
            if restore_rollout:
                await asyncio.to_thread(coordinator.enter_generate)
        logger.info("Deferred scoring: left the score phase")
