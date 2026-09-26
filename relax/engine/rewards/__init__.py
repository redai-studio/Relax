# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import inspect
import random
from typing import Any, Callable

import aiohttp
import ray

from relax.utils.logging_utils import get_logger
from relax.utils.misc import load_function
from relax.utils.types import Sample

from .dapo_genrm import async_compute_score_genrm
from .math_utils import extract_answer as extract_boxed_answer
from .registry import get_reward_spec, list_reward_types, register_reward, resolve_rm_type


logger = get_logger(__name__)
_shared_session: aiohttp.ClientSession | None = None
_MAX_CONTEXT_VALUE_LENGTH = 64
_MAX_GROUP_CONTEXT_SAMPLES = 4
_MAX_GROUP_CONTEXT_LENGTH = 2048


class RewardExecutionError(RuntimeError):
    """Error raised when a custom reward fails for a sample or batch."""

    def __init__(self, *, custom_rm_path: str, sample_context: str, cause: Exception) -> None:
        self.custom_rm_path = custom_rm_path
        self.sample_context = sample_context
        self.cause = cause
        super().__init__(
            f"Custom reward {custom_rm_path!r} failed for {sample_context}: {type(cause).__name__}: {cause}"
        )


def _bounded_repr(value: Any) -> str:
    value_repr = repr(value)
    if len(value_repr) <= _MAX_CONTEXT_VALUE_LENGTH:
        return value_repr
    return f"{value_repr[: _MAX_CONTEXT_VALUE_LENGTH - 3]}..."


def _is_async_callable(function: Callable[..., Any]) -> bool:
    try:
        unwrapped = inspect.unwrap(function)
    except ValueError:
        unwrapped = function
    if inspect.iscoroutinefunction(unwrapped):
        return True

    call = getattr(unwrapped, "__call__", None)
    if call is None:
        return False
    try:
        call = inspect.unwrap(call)
    except ValueError:
        pass
    return inspect.iscoroutinefunction(call)


def _sample_context(sample: Sample | list[Sample]) -> str:
    if isinstance(sample, list):
        displayed_samples = sample[:_MAX_GROUP_CONTEXT_SAMPLES]
        sample_contexts = "; ".join(
            f"sample[{sample_index}]=({_sample_context(group_sample)})"
            for sample_index, group_sample in enumerate(displayed_samples)
        )
        omitted_count = len(sample) - len(displayed_samples)
        omitted_context = f", omitted_samples={omitted_count}" if omitted_count > 0 else ""
        context = f"batch_size={len(sample)}, samples=[{sample_contexts}]{omitted_context}"
        if len(context) > _MAX_GROUP_CONTEXT_LENGTH:
            return f"{context[: _MAX_GROUP_CONTEXT_LENGTH - 3]}..."
        return context
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    request_id = metadata.get("request_id")
    context = [
        f"session_id={_bounded_repr(sample.session_id)}",
        f"index={_bounded_repr(sample.index)}",
        f"group_index={_bounded_repr(sample.group_index)}",
    ]
    if request_id is not None:
        context.append(f"request_id={_bounded_repr(request_id)}")
    return ", ".join(context)


def _get_shared_session() -> aiohttp.ClientSession:
    global _shared_session
    if _shared_session is None or _shared_session.closed:
        connector = aiohttp.TCPConnector(
            limit=64,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=120)
        _shared_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _shared_session


# ---------------------------------------------------------------------------
# RewardWorker: Ray Actor for process-isolated reward computation
# ---------------------------------------------------------------------------
# Solves three problems at once:
# 1. CPU-intensive reward functions no longer block the async event loop.
# 2. Thread-unsafe libraries (e.g. math_verify) are safely isolated inside
#    their own process – each Actor is single-threaded by default.
# 3. Per-process concurrency is bounded by the number of workers in the pool
#    combined with the asyncio.Semaphore in RewardExecutor.
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=0.25)
class RewardWorker:
    """Worker that executes synchronous reward functions in a dedicated
    process.

    Built-in rewards are dispatched by ``rm_type``. Custom reward functions are
    loaded by import path once per worker and then reused for subsequent calls.
    """

    def __init__(self) -> None:
        self._custom_reward_functions: dict[str, Callable[..., Any]] = {}

    def _load_custom_reward(self, path: str) -> Callable[..., Any]:
        function = self._custom_reward_functions.get(path)
        if function is None:
            function = load_function(path)
            if not callable(function):
                raise TypeError(f"Custom reward {path!r} is not callable.")
            self._custom_reward_functions[path] = function
        return function

    def reload_custom_reward(self, path: str) -> None:
        """Reload and cache a custom reward function in this worker process."""
        from relax.utils.reload_utils import reload_function

        function = reload_function(path)
        if function is None or not callable(function):
            raise RuntimeError(f"Failed to reload custom reward {path!r} in RewardWorker.")
        self._custom_reward_functions[path] = function

    def compute_custom(
        self,
        path: str,
        args: Any,
        sample: Sample | list[Sample],
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a synchronous custom reward payload."""
        function = self._load_custom_reward(path)
        result = function(args, sample, **(kwargs or {}))
        if inspect.isawaitable(result):
            close = getattr(result, "close", None)
            if close is not None:
                close()
            raise TypeError(f"Custom reward {path!r} returned an awaitable; declare it with async def.")
        return result

    def compute(self, rm_type: str, response: str, label, metadata: dict | None = None):
        """Dispatch to the registered synchronous reward function.

        Returns the same value the original function would return.
        """
        spec = get_reward_spec(rm_type)
        if spec is None or spec.mode != "sync":
            raise NotImplementedError(
                f"RewardWorker: unknown rm_type={rm_type!r}. Available: {list_reward_types('sync')}"
            )
        fn = spec.resolve()
        if spec.pass_metadata:
            return fn(response, label, metadata=metadata)
        return fn(response, label)


# ---------------------------------------------------------------------------
# RewardExecutor: manages the worker pool and per-process concurrency
# ---------------------------------------------------------------------------


class RewardExecutor:
    """Singleton that manages a pool of RewardWorker actors and an
    asyncio.Semaphore for per-process concurrency control."""

    _instance: "RewardExecutor | None" = None

    def __init__(self, max_concurrency: int = 64, num_workers: int = 16):
        self._max_concurrency = max_concurrency
        self._num_workers = num_workers
        self._semaphore: asyncio.Semaphore | None = None
        self._workers: list = []
        self._worker_init_lock = None
        self._worker_index = 0
        self._custom_reward_functions: dict[str, Callable[..., Any]] = {}
        self._custom_reward_is_async: dict[str, bool] = {}
        self._submission_cleanup_tasks: set[asyncio.Task] = set()

    # -- singleton access -----------------------------------------------------

    @classmethod
    def get_or_create(cls, max_concurrency: int = 64, num_workers: int = 16) -> "RewardExecutor":
        if cls._instance is None:
            cls._instance = cls(max_concurrency=max_concurrency, num_workers=num_workers)
        return cls._instance

    @classmethod
    def reload_custom_reward(cls, path: str) -> None:
        """Refresh a custom reward in the executor and all worker processes."""
        executor = cls._instance
        if executor is None:
            return
        executor._reload_custom_reward(path)

    # -- lazy init (must happen inside an event loop) -------------------------

    def _ensure_semaphore(self):
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)

    async def _ensure_workers(self):
        if self._workers:
            return
        if self._worker_init_lock is None:
            self._worker_init_lock = asyncio.Lock()
        async with self._worker_init_lock:
            if self._workers:
                return
            self._workers = await asyncio.to_thread(
                lambda: [
                    RewardWorker.options(
                        name=f"reward_worker_{i}",
                        get_if_exists=True,
                    ).remote()
                    for i in range(self._num_workers)
                ]
            )
            logger.info(
                "RewardExecutor: created %d RewardWorker actors (max_concurrency=%d)",
                self._num_workers,
                self._max_concurrency,
            )

    def _next_worker(self):
        worker = self._workers[self._worker_index % self._num_workers]
        self._worker_index += 1
        return worker

    def _load_custom_reward(self, path: str) -> Callable[..., Any]:
        function = self._custom_reward_functions.get(path)
        if function is None:
            function = load_function(path)
            if not callable(function):
                raise TypeError(f"Custom reward {path!r} is not callable.")
            self._custom_reward_functions[path] = function
            self._custom_reward_is_async[path] = _is_async_callable(function)
        return function

    def _reload_custom_reward(self, path: str) -> None:
        self._custom_reward_functions.pop(path, None)
        self._custom_reward_is_async.pop(path, None)

        if not self._workers:
            return
        ray.get([worker.reload_custom_reward.remote(path) for worker in self._workers])

    @staticmethod
    def _cancel_worker_call(ref, *, custom_rm_path: str) -> None:
        try:
            ray.cancel(ref, force=False)
        except Exception as exc:
            logger.warning("Failed to request cancellation for custom reward %r: %s", custom_rm_path, exc)

    @classmethod
    async def _cancel_after_submission(cls, submission_task: asyncio.Task, *, custom_rm_path: str) -> None:
        try:
            ref = await submission_task
        except Exception as exc:
            logger.warning(
                "Custom reward %r submission failed while handling cancellation: %s",
                custom_rm_path,
                exc,
            )
            return
        cls._cancel_worker_call(ref, custom_rm_path=custom_rm_path)

    async def _submit_worker_call(self, submit: Callable[[], Any], *, custom_rm_path: str) -> Any:
        submission_task = asyncio.create_task(asyncio.to_thread(submit))
        try:
            return await asyncio.shield(submission_task)
        except asyncio.CancelledError:
            cleanup_task = asyncio.create_task(
                self._cancel_after_submission(submission_task, custom_rm_path=custom_rm_path)
            )
            self._submission_cleanup_tasks.add(cleanup_task)
            cleanup_task.add_done_callback(self._submission_cleanup_tasks.discard)
            raise

    async def _execute_custom(
        self,
        args: Any,
        sample: Sample | list[Sample],
        kwargs: dict[str, Any],
    ) -> Any:
        path = args.custom_rm_path
        sample_context = _sample_context(sample)
        try:
            function = self._load_custom_reward(path)
            if self._custom_reward_is_async[path]:
                return await function(args, sample, **kwargs)

            await self._ensure_workers()
            worker = self._next_worker()
            ref = await self._submit_worker_call(
                lambda: worker.compute_custom.remote(path, args, sample, kwargs),
                custom_rm_path=path,
            )
            try:
                return await ref
            except asyncio.CancelledError:
                self._cancel_worker_call(ref, custom_rm_path=path)
                raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RewardExecutionError(
                custom_rm_path=path,
                sample_context=sample_context,
                cause=exc,
            ) from exc

    # -- public API -----------------------------------------------------------

    async def _execute_custom_with_concurrency(self, args, sample: Sample | list[Sample], **kwargs):
        """Execute one custom reward call with concurrency control."""
        self._ensure_semaphore()
        async with self._semaphore:
            return await self._execute_custom(args, sample, kwargs)

    async def execute(self, args, sample: Sample, **kwargs):
        """Execute a single reward computation with concurrency control.

        - Custom rm paths bypass the format-aware router: async customs are
          awaited in the caller process; sync customs use ``compute_custom``.
        - Non-custom paths keep ``resolve_rm_type`` / registry dispatch.
        - Async rm_types (remote_rm, dapo-genrm) run in the event loop.
        - Sync rm_types are dispatched to the Ray worker pool.
        """
        if args.custom_rm_path is not None and not kwargs.get("ignore_custom", False):
            return await self._execute_custom_with_concurrency(args, sample, **kwargs)

        self._ensure_semaphore()
        async with self._semaphore:
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            route = resolve_rm_type(
                metadata_rm_type=metadata.get("rm_type"),
                args_rm_type=args.rm_type,
                label=sample.label,
                infer=getattr(args, "rm_type_infer", False),
                fallback=getattr(args, "rm_type_fallback", None),
                sample_index=getattr(sample, "index", None),
            )
            if route.zero_reward:
                # Fallback "zero": no dispatch. Reuse the dummy reward so the
                # 0.0 is reward-key aware and cannot mix dict/scalar shapes.
                return await _dummy_reward(args)

            rm_type = route.rm_type
            response = sample.response
            label = sample.label
            if route.boxed:
                response = extract_boxed_answer(response) or ""

            # --- async rm types: run in event loop -----------------------
            spec = get_reward_spec(rm_type) if rm_type else None
            if spec is not None and spec.mode == "async":
                return await spec.resolve()(args, sample)

            # --- sync rm types: dispatch to worker pool ------------------
            # Default to sync path for any non-empty rm_type not registered
            # as async; unknown types keep failing inside the worker.
            if rm_type:
                await self._ensure_workers()
                worker = self._next_worker()
                ref = worker.compute.remote(rm_type, response, label, metadata=metadata)
                return await ref

            # --- no rm_type specified -----------------------------------------
            raise NotImplementedError("Rule-based RM type is not specified.")


# ---------------------------------------------------------------------------
# Public API (backward-compatible)
# ---------------------------------------------------------------------------


def _get_reward_executor(args):
    max_concurrency = getattr(args, "reward_max_concurrency", None)
    if max_concurrency is None:
        max_concurrency = 64
    num_workers = getattr(args, "reward_num_workers", 16)
    return RewardExecutor.get_or_create(
        max_concurrency=max_concurrency,
        num_workers=num_workers,
    )


async def _dummy_reward(args):
    # No-op reward. Paired with --custom-reward-post-process-path when real
    # scoring is deferred to a post-rollout batch pass. Returns a dict when
    # reward_key is set so downstream sample.reward[reward_key] access does
    # not KeyError before the post-process step overwrites everything.
    reward_key = getattr(args, "reward_key", None)
    if reward_key:
        return {reward_key: 0.0}
    return 0.0


async def remote_rm(args, sample: Sample, max_retries: int = 10):
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    session = _get_shared_session()
    for attempt in range(max_retries):
        try:
            async with session.post(args.rm_url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            if attempt + 1 >= max_retries:
                logger.warning(f"remote_rm failed after {attempt + 1} attempts: {e}")
                raise
            backoff = min(2**attempt, 30) + random.random()
            logger.info(f"remote_rm: {type(e).__name__}, retrying in {backoff:.1f}s ({attempt + 1}/{max_retries})")
            await asyncio.sleep(backoff)


# Async rm_types run in the event loop (not dispatched to the worker pool).
register_reward("remote_rm", remote_rm, mode="async")
register_reward("dapo-genrm", async_compute_score_genrm, mode="async")
# `dummy` returns 0 without any computation. Use it when the real reward is
# produced elsewhere (e.g., --custom-reward-post-process-path does batched
# GenRM scoring after all rollout finishes).
register_reward("dummy", lambda args, sample: _dummy_reward(args), mode="async")


async def async_rm(args, sample: Sample, **kwargs):
    """Single-sample reward computation.

    Delegates to RewardExecutor which handles concurrency control and process
    isolation for CPU-bound / thread-unsafe reward functions.
    """
    return await _get_reward_executor(args).execute(args, sample, **kwargs)


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float | dict[str, Any]]:
    """Score multiple samples.

    Custom rewards receive the complete sample payload in one invocation.
    Built-in rewards fan out through per-sample ``async_rm`` calls.
    """
    if args.custom_rm_path is not None and not kwargs.get("ignore_custom", False):
        return await _get_reward_executor(args)._execute_custom_with_concurrency(args, samples, **kwargs)
    return await asyncio.gather(*(async_rm(args, sample, **kwargs) for sample in samples))
