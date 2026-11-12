# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Defer explicitly supported eval judges without changing training rewards.

Custom scorers opt in with ``reward_func.supports_async_eval = True``. This
promises that scoring only needs the exported Sample and independent CPU or
remote resources, not a live agent or the policy/colocated GenRM engines.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import ray

from relax.utils.types import Sample


@dataclass
class DeferredReward:
    samples: Sample | list[Sample]
    kwargs: dict[str, Any]
    route: Any = None


@dataclass
class DeferredEvalRewards:
    calls: list[DeferredReward] = field(default_factory=list)
    sample_ids: set[int] = field(default_factory=set)


_deferred_eval: ContextVar[DeferredEvalRewards | None] = ContextVar("deferred_eval_rewards", default=None)


def eval_reward_deferral_enabled() -> bool:
    return _deferred_eval.get() is not None


def defer_eval_reward(sample: Sample | list[Sample], kwargs: dict[str, Any], *, route: Any = None) -> bool:
    """Register an eligible call, preserving single/batch input semantics."""
    pending = _deferred_eval.get()
    if pending is None:
        return False
    pending.calls.append(DeferredReward(sample, dict(kwargs), route))
    pending.sample_ids.update(id(item) for item in (sample if isinstance(sample, list) else [sample]))
    return True


def is_eval_reward_pending(sample: Sample) -> bool:
    pending = _deferred_eval.get()
    return pending is not None and id(sample) in pending.sample_ids


@contextmanager
def collect_eval_rewards(args: Any, *, enabled: bool = True) -> Iterator[DeferredEvalRewards | None]:
    """Collect eligible rewards in this context; other calls still run
    inline."""
    pending = DeferredEvalRewards() if enabled and not getattr(args, "disable_eval_async_judge", False) else None
    token = _deferred_eval.set(pending)
    try:
        yield pending
    finally:
        _deferred_eval.reset(token)


def call_eval_with_deferred_rewards(fn: Any, args: Any, *positional: Any) -> tuple[Any, list[DeferredReward]]:
    """Capture supported built-in eval calls in the generation thread only."""
    from relax.engine.rollout.base_types import call_rollout_fn

    supported_paths = {
        "relax.agentic.rollout.generate_rollout",
        "relax.engine.rollout.sglang_rollout.generate_rollout",
    }
    with collect_eval_rewards(args, enabled=args.eval_function_path in supported_paths) as pending:
        result = call_rollout_fn(fn, args, *positional, evaluation=True)
    if pending is None:
        return result, []

    # A failed/dropped group must not enqueue orphan work. For a custom batch,
    # preserve its entire original input if any of its outputs are retained.
    retained = {id(sample) for dataset in result.data.values() for sample in dataset.get("samples", [])}
    calls = [
        call
        for call in pending.calls
        if any(
            id(sample) in retained for sample in (call.samples if isinstance(call.samples, list) else [call.samples])
        )
    ]
    return result, calls


@ray.remote(num_cpus=0.25)
class EvalJudgeWorker:
    """Long-lived owner of an isolated reward pool and its event loop."""

    def __init__(self, args: Any) -> None:
        from relax.engine.rewards import RewardExecutor

        self._executor = RewardExecutor(
            max_concurrency=getattr(args, "reward_max_concurrency", None) or 64,
            num_workers=getattr(args, "reward_num_workers", 16),
            worker_name_prefix=f"eval_reward_worker_{uuid.uuid4().hex}",
        )
        self._lock = asyncio.Lock()

    async def score(self, args: Any, payload: tuple[Any, list[DeferredReward]]) -> Any:
        """Score snapshots in submission order; parallelize calls within a
        job."""
        # Keep data and calls in ONE serialized payload so their Sample refs
        # remain shared. Passing them as separate Ray arguments loses aliases.
        result, calls = payload
        async with self._lock:
            tasks = [asyncio.create_task(self._score_call(args, call)) for call in calls]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            reward_key = args.eval_reward_key or args.reward_key
            for dataset in result.data.values():
                if "samples" in dataset:
                    dataset["rewards"] = [
                        sample.reward if not reward_key else sample.reward[reward_key] for sample in dataset["samples"]
                    ]
            return result

    async def _score_call(self, args: Any, call: DeferredReward) -> None:
        if isinstance(call.samples, list):
            rewards = await self._executor._execute_custom_with_concurrency(args, call.samples, **call.kwargs)
            for sample, reward in zip(call.samples, rewards, strict=True):
                sample.reward = reward
        else:
            call.samples.reward = await self._executor.execute(
                args, call.samples, _resolved_route=call.route, **call.kwargs
            )

    async def close(self) -> None:
        from relax.engine.rewards import _shared_session

        for worker in self._executor._workers:
            ray.kill(worker)
        self._executor._workers.clear()
        if _shared_session is not None and not _shared_session.closed:
            await _shared_session.close()
