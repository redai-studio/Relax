# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rollout workload: generation, evaluation, data source, reward post-
processing and queue transfer.

The workload owns no GPU engine, no placement group and no engine actor
handle.  Everything it needs from the inference side goes through
:class:`RolloutInferencePort`, so the engine pool can move into the unified
``InferenceManager`` without touching this module.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

import ray

from relax.core.node_group_affinity import with_control_plane_affinity
from relax.engine.rewards.async_eval import DeferredReward, EvalJudgeWorker, call_eval_with_deferred_rewards
from relax.engine.rollout.base_types import call_rollout_fn
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_checker import MetricChecker
from relax.utils.misc import load_function
from relax.utils.reload_utils import ReloadableMixin
from relax.utils.s3_model_loader import prepare_model_maybe_update_args
from relax.utils.training.train_dump_utils import save_debug_rollout_data


logger = get_logger(__name__)


class RolloutInferencePort(Protocol):
    """The only inference surface a workload is allowed to use."""

    async def resume_health_monitoring(self) -> None:
        """Re-arm engine health monitoring before a generation pass."""

    async def inject_ci_fault(self) -> None:
        """Crash one engine on purpose; CI fault-tolerance coverage only."""

    async def onload_kv(self) -> None:
        """Make the KV cache and cuda graphs resident before generating."""

    def router_base_url(self, model_name: str = "default") -> str:
        """Base URL of the router that fronts the model's engines."""


class RolloutWorkload(ReloadableMixin):
    """Generation, evaluation and data plumbing for one rollout role.

    ``ReloadableMixin`` resolves ``ReloadScope.ROLLOUT_MANAGER`` attributes on
    this object, so the reloadable rollout/eval/reward functions live here
    rather than next to the engine pool.
    """

    def __init__(self, args, data_source: Any, data_system_client: Any, inference: RolloutInferencePort) -> None:
        self.args = args
        self.data_source = data_source
        self.data_system_client = data_system_client
        self.inference = inference

        logger.info(f"import {args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {args.eval_function_path} as eval_generate_rollout function.")
        self.generate_rollout = load_function(args.rollout_function_path)
        self.eval_generate_rollout = load_function(args.eval_function_path)
        self.custom_reward_post_process_func = None
        if args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(args.custom_reward_post_process_path)
        self.custom_convert_samples_to_train_data_func = None
        if args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                args.custom_convert_samples_to_train_data_path
            )

        self.rollout_id = -1
        self.train_parallel_config: dict | None = None
        self._dynamic_global_batch_size = None
        self._metric_checker = MetricChecker.maybe_create(args)
        self._tokenizer = None  # Lazy-initialized tokenizer for debug data saving
        self._eval_judge = None
        self._eval_judge_init_lock = asyncio.Lock()
        self._eval_judge_tasks: set[asyncio.Task] = set()
        self._eval_judge_errors: list[str] = []

    # ----------------------------- data source -----------------------------

    def get_num_rollout_per_epoch(self) -> int:
        import ray

        assert self.args.rollout_global_dataset
        return ray.get(self.data_source.lengths.remote()) // self.args.rollout_batch_size

    async def save(self, rollout_id) -> None:
        await self.data_source.save.remote(rollout_id)

    async def load(self, rollout_id=None) -> None:
        try:
            await self.data_source.load.remote(rollout_id)
        except Exception as e:
            logger.warning(f"Failed to load data source: {e}")

    # ------------------------------ generation -----------------------------

    def get_dynamic_global_batch_size(self):
        """Return the actual sample count from the last rollout step.

        Used by training side to compute correct batch_size for TQ fetch when
        use_dynamic_global_batch_size is enabled.
        """
        assert self._dynamic_global_batch_size is not None, (
            "get_dynamic_global_batch_size called before first generate()"
        )
        return self._dynamic_global_batch_size

    async def generate(self, rollout_id) -> None:
        self.rollout_id = rollout_id
        await self.inference.resume_health_monitoring()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            await self.inference.inject_ci_fault()
        output = await asyncio.to_thread(
            call_rollout_fn,
            self.generate_rollout,
            self.args,
            rollout_id,
            self.data_source,
            self.data_system_client,
            evaluation=False,
        )
        if self.args.partial_rollout and self.args.use_dynamic_global_batch_size:
            self._dynamic_global_batch_size = len(
                {sample.index for sample_group in output.samples for sample in sample_group}
            )

    async def eval(self, rollout_id) -> None:
        await self.inference.resume_health_monitoring()

        # TODO: add fault tolerance to eval
        result, pending = await asyncio.to_thread(
            call_eval_with_deferred_rewards,
            self.eval_generate_rollout,
            self.args,
            rollout_id,
            self.data_source,
            self.data_system_client,
        )
        await self.submit_eval_result(rollout_id, result, pending)

    def _record_eval_result(self, rollout_id: int, result: Any) -> None:
        from relax.distributed.ray.rollout import _log_eval_rollout_data

        data = result.data
        self.save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

    def _on_custom_reward_reloaded(self, module_path: str) -> None:
        """Wait for this manager's background judge to refresh as well."""
        if self._eval_judge is not None:
            ray.get(self._eval_judge.reload_custom_reward.remote(module_path))

    async def submit_eval_result(self, rollout_id: int, result: Any, pending: list[DeferredReward]) -> None:
        """Record inline scores or enqueue captured judges from RL or SFT."""
        if pending:
            async with self._eval_judge_init_lock:
                if self._eval_judge is None:
                    self._eval_judge = await asyncio.to_thread(
                        lambda: EvalJudgeWorker.options(**with_control_plane_affinity(self.args)).remote(self.args)
                    )
            # Ray serializes the snapshot before returning the ObjectRef. Keep
            # result + pending together to preserve their shared Sample refs.
            ref = await asyncio.to_thread(self._eval_judge.score.remote, self.args, (result, pending))
            task = asyncio.create_task(self._finish_eval_judge(rollout_id, ref))
            self._eval_judge_tasks.add(task)
            task.add_done_callback(self._eval_judge_tasks.discard)
            logger.info(
                "Queued eval judge for rollout %s (%d pending evaluations)", rollout_id, len(self._eval_judge_tasks)
            )
            return
        self._record_eval_result(rollout_id, result)

    async def _finish_eval_judge(self, rollout_id: int, ref: Any) -> None:
        start = time.monotonic()
        try:
            result = await ref
            result.metrics = dict(result.metrics or {})
            result.metrics.update({"eval/async_judge": 1, "eval/async_judge_seconds": time.monotonic() - start})
            await asyncio.to_thread(self._record_eval_result, rollout_id, result)
        except Exception as exc:
            self._eval_judge_errors.append(f"rollout {rollout_id}: {exc}")
            logger.exception("Background eval judge failed for rollout %s", rollout_id)

    async def drain_eval_judges(self) -> None:
        """Finish scoring and reporting before the normal job shutdown."""
        while self._eval_judge_tasks:
            logger.info("Waiting for %d pending eval judges at training completion", len(self._eval_judge_tasks))
            await asyncio.gather(*tuple(self._eval_judge_tasks))
        if self._eval_judge_errors:
            raise RuntimeError("Async eval judge failures: " + "; ".join(self._eval_judge_errors))

    async def close(self) -> None:
        if self._eval_judge is not None:
            try:
                await asyncio.wait_for(self._eval_judge.close.remote(), timeout=10)
            finally:
                ray.kill(self._eval_judge)
                self._eval_judge = None

    # ------------------------------ debug dump -----------------------------

    @property
    def tokenizer(self):
        """Lazy-initialized tokenizer for debug data saving."""
        if self._tokenizer is None:
            try:
                from relax.utils.data.processing_utils import load_tokenizer

                prepare_model_maybe_update_args(self.args, completeness="metadata")
                self._tokenizer = load_tokenizer(self.args.hf_checkpoint, trust_remote_code=True)
                logger.info(f"Loaded tokenizer from {self.args.hf_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
        return self._tokenizer

    def save_debug_rollout_data(self, data, rollout_id, evaluation: bool) -> None:
        """Save debug rollout data using shared utility function."""
        save_debug_rollout_data(self.args, data, rollout_id, evaluation, tokenizer=self.tokenizer)

    def set_train_parallel_config(self, config: dict) -> None:
        self.train_parallel_config = config
