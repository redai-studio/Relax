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
from typing import Any, Optional, Protocol

from relax.engine.rollout.base_types import call_rollout_fn
from relax.utils.http_utils import get, post, router_worker_base_urls
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
        from relax.distributed.ray.rollout import _log_eval_rollout_data

        await self.inference.resume_health_monitoring()

        # TODO: add fault tolerance to eval
        result = await asyncio.to_thread(
            call_rollout_fn,
            self.eval_generate_rollout,
            self.args,
            rollout_id,
            self.data_source,
            self.data_system_client,
            evaluation=True,
        )
        data = result.data
        self.save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

    # ------------------------------ SFT predict ----------------------------

    async def run_predict(self, train_step: int) -> None:
        """Periodic SFT predict pass entry point.

        Ensures KV/cuda-graph is onloaded (no-op if already on), then delegates
        to ``run_predict_loop`` which renders the eval set, batches calls to
        ``self.generate_predict``, and writes
        ``<args.save>/predict/predictions_step_<train_step>.jsonl``.

        Mirrors the ``eval`` method: does NOT proactively offload afterward —
        the next training step's actor↔rollout coordination drives state
        transitions, same as PPL eval.
        """
        from relax.engine.sft.predict.loop import run_predict_loop

        await self.inference.onload_kv()
        await run_predict_loop(self, self.args, train_step)

    async def generate_predict(
        self,
        prompts: list[str],
        multimodal_inputs_list: Optional[list[dict | None]] = None,
    ) -> list[str]:
        """Generate completions for ``prompts`` concurrently.

        POSTs all prompts at once in round-robin order directly to engine
        workers, bypassing the router. Predict prompts share a long fixed
        prefix (``<|vision_start|><|image_pad|><|vision_end|>...``); cache-
        aware routing would pin every request to the engine that first
        cached the prefix, defeating multi-engine throughput.

        ``multimodal_inputs_list`` is a parallel list of dicts (or ``None``)
        carrying images/videos/audio for each prompt; encoded inline and
        merged into the payload, mirroring the RL ``generate()`` path.
        """
        import sglang_router
        from packaging.version import parse

        from relax.engine.rollout.sglang_rollout import _encode_multimodal_inputs

        await self.inference.resume_health_monitoring()

        router_base = self.inference.router_base_url()
        if parse(sglang_router.__version__) <= parse("0.2.1") or getattr(self.args, "use_slime_router", False):
            response = await get(f"{router_base}/list_workers")
            worker_urls = response["urls"]
        else:
            response = await get(f"{router_base}/workers")
            worker_urls = [w["url"] for w in response["workers"]]
        worker_urls = router_worker_base_urls(worker_urls)
        if not worker_urls:
            worker_urls = [router_base]

        # Reuse the shared --eval-* sampling args (no SFT-predict-specific
        # flags). Defaults preserve the original SFT predict behaviour
        # (greedy, max_new_tokens=512) when --eval-* is not provided.
        eval_temperature = getattr(self.args, "eval_temperature", None)
        eval_max_response_len = getattr(self.args, "eval_max_response_len", None)
        eval_top_p = getattr(self.args, "eval_top_p", None)
        sampling_params = {
            "temperature": 0.0 if eval_temperature is None else eval_temperature,
            "max_new_tokens": 512 if eval_max_response_len is None else eval_max_response_len,
            "top_p": 1.0 if eval_top_p is None else eval_top_p,
        }
        if multimodal_inputs_list is None:
            multimodal_inputs_list = [None] * len(prompts)

        async def _one(idx: int, prompt: str, mm: dict | None) -> str:
            url = f"{worker_urls[idx % len(worker_urls)]}/generate"
            payload: dict[str, Any] = {"text": prompt, "sampling_params": sampling_params}
            if mm:
                encoded_mm, _ = await _encode_multimodal_inputs(mm)
                payload.update(encoded_mm)
            output = await post(url, payload)
            if isinstance(output, dict) and "text" in output:
                return output["text"]
            if isinstance(output, str):
                return output
            return str(output)

        return await asyncio.gather(
            *[_one(i, p, m) for i, (p, m) in enumerate(zip(prompts, multimodal_inputs_list, strict=True))]
        )

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
