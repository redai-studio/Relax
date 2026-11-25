# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Defer explicitly supported eval judges without changing training rewards.

Custom scorers opt in with ``reward_func.supports_async_eval = True``. This
promises that scoring only needs the exported Sample and independent CPU or
remote resources, not a live agent or the policy/colocated GenRM engines.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypeVar

import ray

from relax.engine.rollout.base_types import EvalJudgeError
from relax.utils.types import Sample


if sys.version_info >= (3, 11):
    from builtins import BaseExceptionGroup
else:
    from exceptiongroup import BaseExceptionGroup


_Params = ParamSpec("_Params")
_Result = TypeVar("_Result")
_eval_judge_failures: ContextVar[list[EvalJudgeError] | None] = ContextVar("eval_judge_failures", default=None)
_score_sources: ContextVar[dict[int, Exception] | None] = ContextVar("score_sources", default=None)


@contextmanager
def prefer_eval_judge_error(enabled: bool) -> Iterator[None]:
    """Share a scoring failure across sibling tasks in one evaluation."""
    failures: list[EvalJudgeError] = []
    token = _eval_judge_failures.set(failures if enabled else None)
    try:
        yield
    except Exception as exc:
        if failures and failures[0] is not exc:
            raise failures[0]
        raise
    finally:
        _eval_judge_failures.reset(token)


@contextmanager
def raw_score_errors() -> Iterator[dict[int, Exception]]:
    sources: dict[int, Exception] = {}
    token = _score_sources.set(sources)
    try:
        yield sources
    finally:
        sources.clear()
        _score_sources.reset(token)


@contextmanager
def eval_judge_errors(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    sources = _score_sources.get()
    # Even a standalone scoring call can invoke a plugin which handles nested scorers itself.
    with raw_score_errors() if sources is None else nullcontext():
        try:
            yield
        except EvalJudgeError:
            raise
        except Exception as exc:
            if sources is None:
                raise EvalJudgeError(str(exc)) from exc
            sources[id(exc)] = exc
            raise


def _escaped_score_error(exc: BaseException, sources: dict[int, Exception]) -> BaseException | None:
    pending = [exc]
    seen: set[int] = set()
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        if isinstance(error, EvalJudgeError) or id(error) in sources:
            return error
        if isinstance(error, BaseExceptionGroup):
            pending.extend(reversed(error.exceptions))
        # A plugin can add context with "raise ... from ...". Its implicit
        # __context__ can instead be a handled score followed by a failed fallback.
        if error.__cause__ is not None:
            pending.append(error.__cause__)
    return None


def observe_eval_task(
    fn: Callable[_Params, Coroutine[Any, Any, _Result]],
) -> Callable[_Params, Coroutine[Any, Any, _Result]]:
    """Remember scoring errors that escape a scheduled evaluation task."""
    failures = _eval_judge_failures.get()
    if failures is None:
        return fn

    async def observed(*args: _Params.args, **kwargs: _Params.kwargs) -> _Result:
        with raw_score_errors() as sources:
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                score_error = _escaped_score_error(exc, sources)
                if score_error is None:
                    raise
                if isinstance(exc, EvalJudgeError):
                    error = exc
                else:
                    message = str(exc) if exc is score_error else f"{exc}: {score_error}"
                    error = EvalJudgeError(message)
                if not failures:
                    failures.append(error)
                if error is exc:
                    raise
                raise error from exc

    return observed


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

    async def reload_custom_reward(self, path: str) -> None:
        """Refresh the owner and its workers between complete eval jobs."""
        from relax.utils.reload_utils import reload_function

        def refresh() -> None:
            function = reload_function(path)
            if function is None or not callable(function):
                raise RuntimeError(f"Failed to reload custom reward {path!r} in EvalJudgeWorker.")
            self._executor.refresh_custom_reward(path)

        async with self._lock:
            task = asyncio.create_task(asyncio.to_thread(refresh))
            cancelled = False
            # Cancelling the RPC cannot stop its thread. Keep scoring excluded
            # until every worker finishes refreshing, even on repeated cancel.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
            task.result()
            if cancelled:
                raise asyncio.CancelledError

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
        # The manager classifies errors that escape this worker. Plugins may handle nested scorer errors first.
        with raw_score_errors():
            if isinstance(call.samples, list):
                rewards = await self._executor.execute_batch(args, call.samples, **call.kwargs)
                for sample, reward in zip(call.samples, rewards, strict=True):
                    sample.reward = reward
            else:
                call.samples.reward = await self._executor.execute(
                    args, call.samples, _resolved_route=call.route, **call.kwargs
                )

    async def close(self) -> None:
        from relax.engine.rewards import _shared_session

        self._executor.close_workers()
        if _shared_session is not None and not _shared_session.closed:
            await _shared_session.close()
