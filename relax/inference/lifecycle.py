# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Serial activation leases for inference phases that share physical GPUs."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Any


class LifecycleCoordinator:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.phase: str | None = None

    async def run_phase(
        self,
        name: str,
        activate: Callable[[], Awaitable],
        deactivate: Callable[[], Awaitable],
        work: Callable[[], Awaitable],
    ) -> Any:
        async with self._lock:
            if self.phase is not None:
                raise RuntimeError(f"GPU lease retained by failed phase {self.phase!r}")
            self.phase = name
            try:
                await activate()
                return await work()
            finally:
                # Even a partial activation must roll back. A failed drain
                # retains the lease; no other model may acquire these GPUs.
                await deactivate()
                self.phase = None


async def run_deferred_batch(
    args: Any,
    samples: list,
    opd: Any,
    encode_mm: Callable,
    publish: Callable[[], Awaitable],
    *,
    evaluation: bool = False,
    reward_groups: list[Iterable] | None = None,
) -> None:
    """Score retained Sample objects completely before any TQ publication."""
    from relax.distributed.ray.rollout import get_local_rollout_manager
    from relax.engine.rewards import batched_async_rm

    rollout = get_local_rollout_manager()
    teachers = list(getattr(args, "_inference_teacher_managers", {}).values())
    judges = list(getattr(rollout, "_inference_genrm_managers", []))
    coordinator = rollout.lifecycle_coordinator

    async def call_all(managers: list, method: str) -> None:
        # Serial calls also cover partial activation failure: run_phase's
        # finally deactivates every manager, including those already awake.
        failures = []
        for manager in managers:
            try:
                await getattr(manager, method).remote()
            except Exception as exc:
                if method != "offload":
                    raise
                failures.append(exc)
        if failures:
            raise failures[0]

    async def rollout_on() -> None:
        await rollout.onload()

    async def rollout_off() -> None:
        await rollout.offload()

    await rollout_off()
    await call_all(teachers + judges, "offload")
    if getattr(args, "opd_teacher_defer", False):
        if opd is None or not teachers:
            raise RuntimeError("Deferred Teacher is missing its OPD workload or managed engines")
        await coordinator.run_phase(
            "teacher",
            lambda: call_all(teachers, "onload"),
            lambda: call_all(teachers, "offload"),
            lambda: opd.prefill_teacher(samples, strict=True),
        )
        if opd.topk_worker is not None and opd.topk_worker.spec.student_at_teacher:
            await coordinator.run_phase(
                "student_prefill", rollout_on, rollout_off, lambda: opd.prefill_student(samples, encode_mm)
            )
        opd.finish_prefill(samples, strict=True)

    async def score_rewards() -> None:
        from relax.utils.utils import post_process_rewards

        if not args.custom_reward_post_process_path:
            groups = reward_groups if getattr(args, "group_rm", False) else [samples]
            if groups is None:
                raise ValueError("Deferred group rewards require the original prompt groups")
            for group in groups:
                # Rollout may supply flattened iterators. Retain the same
                # Sample objects for both scoring and result writeback.
                group = list(group)
                rewards = await batched_async_rm(args, group)
                for sample, reward in zip(group, rewards, strict=True):
                    sample.reward = reward
        if evaluation:
            if args.custom_reward_post_process_path:
                import copy

                eval_args = copy.copy(args)
                eval_args.rewards_normalization = False
                raw, _ = await asyncio.to_thread(post_process_rewards, eval_args, samples)
                for sample, reward in zip(samples, raw, strict=True):
                    key = getattr(args, "eval_reward_key", None) or getattr(args, "reward_key", None)
                    if key:
                        previous = sample.reward if isinstance(sample.reward, dict) else {}
                        sample.reward = {**previous, key: reward}
                    else:
                        sample.reward = reward
            return
        raw, processed = await asyncio.to_thread(post_process_rewards, args, samples)
        for sample, raw_reward, processed_reward in zip(samples, raw, processed, strict=True):
            sample._inference_reward_result = (raw_reward, processed_reward)

    if getattr(args, "defer_reward_to_post_process", False):
        if not judges:
            raise RuntimeError("Deferred GenRM has no managed engines")
        await coordinator.run_phase(
            "genrm", lambda: call_all(judges, "onload"), lambda: call_all(judges, "offload"), score_rewards
        )
    # All overlapping inference models have relinquished their GPUs before
    # ready samples can wake the training consumer.
    await publish()
