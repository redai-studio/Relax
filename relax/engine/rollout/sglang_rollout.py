# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import contextvars
import copy
import inspect
import uuid
from argparse import Namespace
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, contextmanager
from time import monotonic
from typing import Any

import numpy as np
import pybase64
import ray
import sglang_router
import torch
from packaging.version import parse
from tqdm import tqdm

from relax.distributed.ray.rollout import _log_rollout_data
from relax.engine.filters.base_types import MetricGatherer, call_dynamic_filter
from relax.engine.rewards import async_rm, batched_async_rm
from relax.engine.rollout import on_policy_distillation as opd
from relax.engine.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from relax.engine.rollout.request_permit import GenerationAborted, InferencePermitManager
from relax.utils.async_utils import run
from relax.utils.data.data import Dataset
from relax.utils.data.processing_utils import (
    async_encode_audio_for_rollout_engine,
    async_encode_image_for_rollout_engine,
    async_encode_video_tensor_for_rollout_engine,
    configure_encode_executor,
    load_processor,
    load_tokenizer,
)
from relax.utils.data.processor_pool import ProcessorPool, prepare_mm_inputs_for_ipc, process_sample_in_worker
from relax.utils.http_utils import get, post, router_worker_base_urls
from relax.utils.logging_utils import get_logger
from relax.utils.misc import SingletonMeta, load_function
from relax.utils.profile_utils import start_sglang_profile, stop_sglang_profile
from relax.utils.s3_model_loader import prepare_model_maybe_update_args
from relax.utils.timer import Timer
from relax.utils.training.eval_config import EvalDatasetConfig
from relax.utils.training.train_dump_utils import save_debug_rollout_data
from relax.utils.types import Sample
from relax.utils.utils import CURRENT_ROLLOUT_BATCH, compute_dp_size, transfer_batch_to_data_system


__all__ = ["generate_rollout"]

logger = get_logger(__name__)


# Misuse guard for the per-request permit contract. Set while the session-level
# lock (GenerateState.semaphore) is held so that a legacy custom function that
# wrongly calls inference_permit()/post_generate() fails loudly instead of
# deadlocking on the same non-reentrant semaphore. A ContextVar is task-local
# and propagates down the await chain (including create_task children).
_holding_session_lock: contextvars.ContextVar[bool] = contextvars.ContextVar("holding_session_lock", default=False)


def _ensure_not_holding_session_lock() -> None:
    if _holding_session_lock.get():
        raise RuntimeError(
            "inference_permit()/post_generate() was called while the session-level lock is held. "
            "A custom generate function must declare `manages_inference_permit = True` to use "
            "per-request permits; without it the function runs under the session lock (the same "
            "non-reentrant semaphore) and acquiring a permit would deadlock. "
            "See docs/{zh,en}/guide/customize-training.md."
        )


class GenerateState(metaclass=SingletonMeta):
    """The global state for the generation process."""

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        prepare_model_maybe_update_args(args, completeness="metadata")
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        # OPD manager (singleton — one OpdManager per GenerateState)
        self.opd_manager = opd.OpdManager(args) if opd.is_opd_enabled(args) else None

        # Media-encoding thread pool for this rollout worker process, sized by
        # --encode-max-workers (falls back to $RELAX_ENCODE_MAX_WORKERS, then an
        # affinity-aware default). Held on state so consumers use it explicitly.
        self.encode_executor = configure_encode_executor(getattr(args, "encode_max_workers", None))

        # Process pool for running HuggingFace processor without GIL contention.
        # Controlled by --mm-processor-pool-size (0 = disabled).
        self.processor_pool = None
        if self.processor is not None:
            pool_size = getattr(args, "mm_processor_pool_size", 0)
            if pool_size > 0:
                try:
                    self.processor_pool = ProcessorPool(
                        model_path=args.hf_checkpoint,
                        pool_size=pool_size,
                        trust_remote_code=True,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create ProcessorPool, falling back to ThreadPoolExecutor: {e}")

        capacity = args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        self.permit_manager = InferencePermitManager(capacity)
        # Backward-compatible alias: same underlying semaphore, so the session-level
        # lock path behaves exactly as before. Custom multi-turn rollouts should use
        # inference_permit()/post_generate() instead (see docs customize-training).
        self.semaphore = self.permit_manager.semaphore
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0

        self.reset()

    @contextmanager
    def dp_rank_context(self):
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        dp_rank = int(np.random.choice(candidates))
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def inference_permit(self) -> AbstractAsyncContextManager[None]:
        """Per-request permit for custom multi-turn rollouts.

        Acquire one permit per model request and release it right after; do not
        hold it during env/tool execution. Only for custom generate functions
        that declare ``manages_inference_permit = True`` -- see
        docs/{zh,en}/guide/customize-training.md.
        """
        _ensure_not_holding_session_lock()
        return self.permit_manager.permit(abort_check=lambda: self.aborted)

    async def post_generate(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        """Acquire a per-request permit, honor abort, then POST to the engine.

        Recommended entry point for custom multi-turn rollouts: it scopes the
        permit to exactly one in-flight request. Raises ``GenerationAborted``
        if rollout abort has been signalled by the time the permit is acquired.
        """
        async with self.inference_permit():
            return await post(url, payload, headers=headers)

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.protected_pendings = (
            set()
        )  # tasks that should not be aborted (abort_count >= partial_rollout_max_aborted_count)
        self.aborted = False
        self.evaluating = getattr(self, "evaluating", 0)  # preserve eval state across resets
        # Pre-fetched data ObjectRef for cross-step overlap.
        # Persisted across reset() calls so the ref submitted at the end of
        # step N is consumed at the beginning of step N+1.
        if not hasattr(self, "prefetched_samples_ref"):
            self.prefetched_samples_ref: ray.ObjectRef | None = None
        # How many groups the previous step left short of its transfer-queue target
        # (rollout_batch_size - committed_current). The next step backfills exactly this
        # many into rollout_id-1. Persisted across reset() like prefetched_samples_ref;
        # 0 before the first step / when the previous step met its target. fully_async only.
        if not hasattr(self, "last_step_current_deficit"):
            self.last_step_current_deficit = 0

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        max_aborted_count = getattr(self.args, "partial_rollout_max_aborted_count", None)
        for group in samples:
            task = asyncio.create_task(
                generate_and_rm_group(
                    self.args,
                    group,
                    sampling_params=self.sampling_params.copy(),
                    evaluation=False,
                )
            )
            # If any sample in the group has been aborted >= partial_rollout_max_aborted_count,
            # mark this task as protected so it won't be aborted again.
            if max_aborted_count is not None and any(sample.abort_count >= max_aborted_count for sample in group):
                self.protected_pendings.add(task)
            else:
                self.pendings.add(task)
        self.remaining_batch_size += len(samples)


async def _run_image_processor(
    state: GenerateState, args: Namespace, prompt: str | list[dict[str, str]], multimodal_inputs: dict
) -> tuple[list[int], dict | None, float]:
    """Run HF processor and return (prompt_ids, mm_train_inputs,
    elapsed_seconds)."""
    t_start = monotonic()
    loop = asyncio.get_running_loop()

    if state.processor_pool is not None:
        mm_inputs_ipc = prepare_mm_inputs_for_ipc(multimodal_inputs)
        processor_kwargs = {
            "use_audio_in_video": args.use_audio_in_video,
            "return_mm_token_type_ids": False,
        }
        processor_prompt_ids, mm_train_inputs = await loop.run_in_executor(
            state.processor_pool.executor,
            process_sample_in_worker,
            prompt,
            mm_inputs_ipc,
            processor_kwargs,
        )
    else:
        from relax.utils.data.processing_utils import (
            adapt_processor_kwargs,
            expand_kimi_k25_placeholders,
            remap_mm_train_inputs,
        )

        def _run_processor():
            adapted = adapt_processor_kwargs(
                state.processor,
                multimodal_inputs,
                {
                    "use_audio_in_video": args.use_audio_in_video,
                    "return_mm_token_type_ids": False,
                },
            )
            processor_output = state.processor(text=prompt, **adapted)
            prompt_ids = processor_output["input_ids"][0]
            # K2.x adapt_processor_kwargs forces return_tensors="pt", so
            # input_ids is a 1D Tensor; downstream sample.tokens contract is list[int].
            if isinstance(prompt_ids, torch.Tensor):
                prompt_ids = prompt_ids.tolist()
            train_inputs = {
                k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
                for k, v in processor_output.items()
                if k not in ["input_ids", "attention_mask"]
            } or None
            train_inputs = remap_mm_train_inputs(state.processor, train_inputs)
            prompt_ids = expand_kimi_k25_placeholders(state.processor, prompt_ids, train_inputs)
            return prompt_ids, train_inputs

        processor_prompt_ids, mm_train_inputs = await loop.run_in_executor(state.encode_executor, _run_processor)

    return processor_prompt_ids, mm_train_inputs, monotonic() - t_start


async def _encode_multimodal_inputs(multimodal_inputs: dict) -> tuple[dict[str, list], float]:
    """Base64-encode multimodal data and return (encoded_data,
    elapsed_seconds)."""
    t_start = monotonic()
    encode_coros = []

    if image_data := multimodal_inputs["images"]:
        encode_coros.extend(async_encode_image_for_rollout_engine(image) for image in image_data)
    image_count = len(image_data) if multimodal_inputs.get("images") else 0

    if video_data := multimodal_inputs["videos"]:
        encode_coros.extend(async_encode_video_tensor_for_rollout_engine(video) for video in video_data)
    video_count = len(video_data) if multimodal_inputs.get("videos") else 0

    if audio_data := multimodal_inputs["audio"]:
        encode_coros.extend(async_encode_audio_for_rollout_engine(audio) for audio in audio_data)

    encoded: dict[str, list] = {}
    if encode_coros:
        results = await asyncio.gather(*encode_coros)
        offset = 0
        if image_count:
            encoded["image_data"] = list(results[offset : offset + image_count])
            offset += image_count
        if video_count:
            encoded["video_data"] = list(results[offset : offset + video_count])
            offset += video_count
        if offset < len(results):
            encoded["audio_data"] = list(results[offset:])

    return encoded, monotonic() - t_start


async def generate(
    args: Namespace, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False
) -> Sample:
    """Generate using traditional SGLang router with token-based workflow."""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED, (
        f"Sample status is {sample.status}"
    )

    tokenizer_prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)

    _t_image_processor: float | None = None
    # K2.x ships a multimodal AutoProcessor even for text-only fine-tunes; the
    # data loader always populates multimodal_inputs with empty-list placeholders
    # in that case, so check for actual media content before routing through the
    # image processor (which would otherwise raise on text-only K2.x) — and use
    # the same gate downstream so SGLang's payload doesn't get empty media fields.
    _has_media = sample.multimodal_inputs is not None and any(
        sample.multimodal_inputs.get(k) for k in ("images", "videos", "audio")
    )
    if state.processor and _has_media:
        processor_prompt_ids, sample.multimodal_train_inputs, _t_image_processor = await _run_image_processor(
            state, args, sample.prompt, sample.multimodal_inputs
        )
    else:
        processor_prompt_ids = tokenizer_prompt_ids

    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(processor_prompt_ids)

    assert sampling_params["max_new_tokens"] >= 0, (
        f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    )
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": not evaluation,
    }

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    if state.opd_manager and not evaluation:
        state.opd_manager.before_rollout(payload)
    # LoRA adapter mode: route generation through the trained adapter registered on the engine
    # (see UpdateWeightFromTensor._push_lora_adapter). Without lora_path the engine serves the
    # base model only and rollouts would not reflect the trained policy.
    if getattr(args, "lora_rank", 0) > 0 and getattr(args, "lora_adapter_mode", False):
        from relax.utils.megatron_peft_utils import LORA_ADAPTER_NAME

        payload["lora_path"] = LORA_ADAPTER_NAME

    _t_mm_encode: float | None = None
    if _has_media:
        # Use pre-encoded data from group-level de-dup if available; otherwise encode inline.
        pre_encoded = getattr(sample, "_pre_encoded_mm", None)
        if pre_encoded is not None:
            encoded_mm = pre_encoded
            _t_mm_encode = getattr(sample, "_pre_encoded_mm_elapsed", 0.0)
            del sample._pre_encoded_mm
            if hasattr(sample, "_pre_encoded_mm_elapsed"):
                del sample._pre_encoded_mm_elapsed
        else:
            encoded_mm, _t_mm_encode = await _encode_multimodal_inputs(sample.multimodal_inputs)
        payload.update(encoded_mm)

    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.rollout_tokens
    else:
        payload["input_ids"] = tokenizer_prompt_ids
        # Initialize sample.tokens for the first turn
        if not sample.tokens:
            sample.tokens = processor_prompt_ids
        if not sample.rollout_tokens:
            sample.rollout_tokens = tokenizer_prompt_ids

    # Provide a routing key so cache-affinity routers pin related requests to the same
    # engine and reuse its prefix/KV cache.
    headers = None
    if args.sglang_router_policy == "consistent_hashing" and sample.session_id:
        headers = {"X-SMG-Routing-Key": sample.session_id}
    elif getattr(args, "slime_router_sticky", False) and sample.group_index is not None:
        # Pin all samples of one prompt group to the same engine so the shared prompt
        # prefix is prefilled once and reused across the group.
        headers = {"X-SMG-Routing-Key": str(sample.group_index)}

    _t_generate_start = monotonic()
    output = await post(url, payload, headers=headers)
    _t_generate = monotonic() - _t_generate_start

    _t_post_generate_start = monotonic()
    if args.use_slime_router and "RadixTreeMiddleware" in args.slime_router_middleware_paths:
        from relax.engine.router.middleware.radix_tree_middleware import postprocess_sample_with_radix_tree

        sample = await postprocess_sample_with_radix_tree(args, sample, output)
    else:
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            new_response_tokens = output["output_ids"]
            new_response_log_probs = []

        if state.opd_manager and not evaluation:
            new_response_tokens, new_response_log_probs = state.opd_manager.parse_rollout_logprobs(
                output["meta_info"], new_response_tokens, new_response_log_probs
            )

        while hasattr(state.tokenizer, "image_token_id") and state.tokenizer.image_token_id in new_response_tokens:
            index = new_response_tokens.index(state.tokenizer.image_token_id)
            new_response_tokens[index] = state.tokenizer.pad_token_id
            logger.warning(
                "Image token found in output tokens, replaced with pad_token_id. Consider updating the model's stop condition to stop at image_token_id if you want to avoid this."
            )

        while hasattr(state.tokenizer, "audio_token_id") and state.tokenizer.audio_token_id in new_response_tokens:
            index = new_response_tokens.index(state.tokenizer.audio_token_id)
            new_response_tokens[index] = state.tokenizer.pad_token_id
            logger.warning(
                "Audio token found in output tokens, replaced with pad_token_id. Consider updating the model's stop condition to stop at audio_token_id if you want to avoid this."
            )

        while hasattr(state.tokenizer, "video_token_id") and state.tokenizer.video_token_id in new_response_tokens:
            index = new_response_tokens.index(state.tokenizer.video_token_id)
            new_response_tokens[index] = state.tokenizer.pad_token_id
            logger.warning(
                "Video token found in output tokens, replaced with pad_token_id. Consider updating the model's stop condition to stop at video_token_id if you want to avoid this."
            )

        # K2.x tokenizers don't expose image_token_id but reserve <|media_pad|>
        # for vision input slots. A hallucinated <|media_pad|> in the response
        # inflates num_placeholders past sum(feature_lengths) in the bridge,
        # forcing dynamic expansion → broadcast → 233 GiB OOM. Replace in-place
        # so positional accounting matches sglang's per-token logprobs.
        if state.processor is not None:
            from relax.utils.data.processing_utils import sanitize_kimi_k25_response_tokens

            sanitized = sanitize_kimi_k25_response_tokens(state.processor, new_response_tokens)
            if sanitized is not new_response_tokens:
                replaced = sum(1 for a, b in zip(new_response_tokens, sanitized, strict=True) if a != b)
                if replaced:
                    logger.warning(
                        f"K2.x: replaced {replaced} stray <|media_pad|> token(s) in rollout response with pad_token_id."
                    )
                new_response_tokens = sanitized

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.rollout_tokens = sample.rollout_tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response += output["text"]

        # When partial rollout and masking off policy is enabled, update the loss mask
        if sample.loss_mask is not None:
            assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
            sample.loss_mask += [1] * len(new_response_tokens)

        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += new_response_log_probs

        if state.opd_manager and not evaluation:
            state.opd_manager.after_rollout(sample, output)

    if "routed_experts" in output["meta_info"]:
        sample.rollout_routed_experts = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )

    sample.update_from_meta_info(args, output["meta_info"])
    _t_post_generate = monotonic() - _t_post_generate_start

    _timing: dict[str, float] = {"generate": _t_generate, "post_generate": _t_post_generate}
    if _t_image_processor is not None:
        _timing["image_processor"] = _t_image_processor
    if _t_mm_encode is not None:
        _timing["mm_encode"] = _t_mm_encode
    sample.metadata["_timing"] = _timing

    return sample


async def _dispatch_generate(
    state: "GenerateState",
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    """Resolve the generate function and run it under the correct permit scope.

    Touches only ``state.aborted`` / ``state.dp_rank_context`` / ``state.semaphore``
    so it can be unit-tested with a lightweight stub state (no GenerateState).

    - Default (built-in ``generate`` or a custom function without the opt-in flag):
      hold the session-level lock for the whole call, exactly as before.
    - Opt-in (custom function with ``manages_inference_permit = True``): do NOT
      hold the session lock; the function acquires a per-request permit per turn
      via ``state.post_generate``/``state.inference_permit``.

    ``GenerationAborted`` must never escape this function: ``generate_and_rm_group``
    runs the per-sample tasks through ``asyncio.gather`` *without*
    ``return_exceptions=True``, and no existing abort path raises -- so a leaked
    exception would crash the whole rollout step. It is contained here and mapped
    to an ABORTED sample (mirroring the legacy in-lock abort check).
    """
    custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path
    custom_generate_func = load_function(custom_func_path) if custom_func_path is not None else None
    manages_permit = bool(getattr(custom_generate_func, "manages_inference_permit", False))

    async def _run() -> Sample | list[Sample]:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample
        with state.dp_rank_context() as _:
            if custom_generate_func is not None:
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    return await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
                return await custom_generate_func(args, sample, sampling_params)
            return await generate(args, sample, sampling_params, evaluation=evaluation)

    try:
        if manages_permit:
            return await _run()
        async with state.semaphore:
            token = _holding_session_lock.set(True)
            try:
                return await _run()
            finally:
                _holding_session_lock.reset(token)
    except GenerationAborted:
        sample.status = Sample.Status.ABORTED
        return sample


async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # Resolve the generate function and run it under the right permit scope
    # (session-level lock by default; per-request permits for opt-in functions).
    sample = await _dispatch_generate(state, args, sample, sampling_params, evaluation=evaluation)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any(sample.status == Sample.Status.ABORTED for sample in samples):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward

        if state.opd_manager and not evaluation:
            await state.opd_manager.prefill(samples, _encode_multimodal_inputs)

        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            sample.reward = await async_rm(args, sample)

        if state.opd_manager and not evaluation:
            await state.opd_manager.prefill(sample, _encode_multimodal_inputs)

    return sample


def _collect_timing_from_samples(samples: list[Sample]) -> dict[str, list[float]]:
    """Extract per-phase timing lists from sample metadata written by
    generate()."""
    collected: dict[str, list[float]] = {}
    for sample in samples:
        timing = sample.metadata.get("_timing")
        if not timing:
            continue
        for key, value in timing.items():
            collected.setdefault(key, []).append(value)
    return collected


def _aggregate_rollout_timing(all_samples: list[Sample], get_samples_times: list[float]) -> dict[str, float]:
    timing_data = _collect_timing_from_samples(all_samples)
    metrics: dict[str, float] = {}

    for phase in ("image_processor", "mm_encode", "generate", "post_generate"):
        values = timing_data.get(phase, [])
        if not values:
            continue
        metrics[f"perf_detail/rollout/{phase}_time/mean"] = sum(values) / len(values)
        metrics[f"perf_detail/rollout/{phase}_time/max"] = max(values)

    if get_samples_times:
        metrics["perf_detail/rollout/get_samples_time/total"] = sum(get_samples_times)
        metrics["perf_detail/rollout/get_samples_time/mean"] = sum(get_samples_times) / len(get_samples_times)

    return metrics


async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)

    # eval requests should not be affected by abort state; only skip for training rollout
    if state.aborted and not evaluation:
        return group

    # Generate a unique session_id for each sample in the group
    for sample in group:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())

    # Group-level multimodal encoding de-duplication: when samples in the same
    # group share the same multimodal_inputs object (e.g. after shallow-copy in
    # data_source), encode once and attach the result to every sample so that
    # generate() picks up the pre-encoded data instead of re-encoding per sample.
    first_mm = getattr(group[0], "multimodal_inputs", None)
    if first_mm is not None and all(getattr(s, "multimodal_inputs", None) is first_mm for s in group[1:]):
        encoded_mm, t_enc = await _encode_multimodal_inputs(first_mm)
        for sample in group:
            sample._pre_encoded_mm = encoded_mm
            sample._pre_encoded_mm_elapsed = t_enc

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    group = await asyncio.gather(*tasks)

    # eval should still compute group reward even if abort was triggered by a concurrent rollout
    if (not state.aborted or evaluation) and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

        if state.opd_manager and not evaluation:
            await state.opd_manager.prefill(group, _encode_multimodal_inputs)

    return group


async def abort(args: Namespace, rollout_id: int) -> tuple[list[list[Sample]], list[list[Sample]]]:
    aborted_samples = []
    completed_protected_samples = []

    state = GenerateState(args)
    assert not state.aborted

    # Wait for any in-progress eval to finish before aborting.
    # Aborting during eval would send abort_all to SGLang workers and kill eval requests.
    if state.evaluating > 0:
        logger.info(
            f"Abort deferred: {state.evaluating} eval task(s) in progress. "
            f"Waiting for eval to complete before aborting rollout {rollout_id}."
        )
        while state.evaluating > 0:
            await asyncio.sleep(0.5)
        logger.info(f"Eval completed. Proceeding with abort for rollout {rollout_id}.")

    # Step 1: Wait for protected tasks (abort_count >= partial_rollout_max_aborted_count) to finish naturally.
    if state.protected_pendings:
        logger.info(
            f"Waiting for {len(state.protected_pendings)} protected tasks "
            f"(abort_count >= partial_rollout_max_aborted_count) to complete before aborting others."
        )
        while state.protected_pendings:
            done, state.protected_pendings = await asyncio.wait(
                state.protected_pendings, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                group = task.result()
                completed_protected_samples.append(group)

        logger.info(f"All {len(completed_protected_samples)} protected tasks completed.")

    # Step 2: Now abort the remaining (non-protected) pending tasks.
    state.aborted = True

    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_slime_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    urls = router_worker_base_urls(urls)
    logger.info(f"Abort request for {urls}")
    abort_tasks = [post(f"{url}/abort_request", {"abort_all": True}) for url in urls]
    abort_results = await asyncio.gather(*abort_tasks, return_exceptions=True)
    for url, result in zip(urls, abort_results, strict=False):
        if isinstance(result, BaseException):
            logger.warning(f"Failed to abort worker at {url}: {result}")

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.status == Sample.Status.ABORTED:
                    sample.abort_count += 1
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples, completed_protected_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]], data_system_client: Any
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based
    rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch
        data_system_client: the data system client to use for transferring batches

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    timer = Timer()
    timer.start("rollout")
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # Start SGLang profiling if enabled
    await start_sglang_profile(args, rollout_id)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()

    # Backfill debt for the previous step = the deficit it recorded at close
    # (rollout_batch_size - committed_current), NOT the buffer carryover size. Known up
    # front (no longer derived from how much get_samples returned), so target_data_size
    # is fixed here instead of being mutated mid-loop.
    num_old_samples = state.last_step_current_deficit if args.fully_async else 0

    is_final_backfill = args.fully_async and rollout_id >= args.num_rollout

    # target_data_size = how many groups this step COMMITS to the transfer queue:
    # current-partition target (rollout_batch_size) + previous-partition backfill
    # (num_old_samples). The final backfill step is special: there is no
    # train_{rollout_id} partition, so it only closes train_{rollout_id-1}.
    target_data_size = num_old_samples if is_final_backfill else args.rollout_batch_size + num_old_samples
    if target_data_size <= 0:
        raise RuntimeError(f"Final rollout backfill requested for rollout_id={rollout_id} without pending deficit")

    # Inner-loop top-up threshold = the commit target. Each submit_generate_tasks call
    # admits a full over_sampling_batch_size of groups, so one round already puts more
    # tasks in flight than target_data_size (the over-sampling envelope). Gating top-up on
    # target_data_size (not over_sampling_batch_size) keeps the over-sample surplus as slack
    # that absorbs aborted/filtered groups; gating on over_sampling_batch_size removes that
    # slack when over_sampling_batch_size == rollout_batch_size and deadlocks the outer loop
    # on an empty pending set once any group aborts.
    submit_target = target_data_size

    data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc=f"Rollout {rollout_id} generation")
    transfer_tasks = []
    batch_to_transfer = []
    aborted_samples = []
    # Completed groups beyond target_data_size (over-sampling surplus). Carried back to
    # the buffer for the next step instead of being dropped, so no samples are lost.
    oversample_surplus = []
    total_transfer_samples = 0
    get_samples_times: list[float] = []

    # is_last bookkeeping: a partition train_X is filled across two steps (step X
    # commits committed_current, step X+1 backfills the deficit). Mark is_last on
    # the transfer that brings a partition's cumulative commits to its final target:
    #   - prev partition (train_{rollout_id-1}): closed by this step's num_old_samples backfill
    #   - cur partition (train_{rollout_id}): only if this step fully meets rollout_batch_size
    #     (no deficit carried); otherwise its tail is backfilled next step.
    committed_prev = 0  # groups committed to train_{rollout_id-1} this step
    committed_curr = 0  # groups committed to train_{rollout_id} this step
    prev_target = num_old_samples
    curr_target = 0 if is_final_backfill else args.rollout_batch_size

    if is_final_backfill:
        logger.info(
            f"Starting final rollout backfill step {rollout_id}: target(prev)={target_data_size}, "
            f"submit_target={submit_target}"
        )
    else:
        logger.info(
            f"Starting rollout step {rollout_id}: target(commit)={target_data_size} "
            f"(rollout_batch={args.rollout_batch_size} + old={num_old_samples}), submit_target={submit_target}"
        )

    loop = asyncio.get_running_loop()

    def target_reached() -> bool:
        if is_final_backfill:
            return total_transfer_samples >= target_data_size
        return len(data) >= target_data_size

    # Outer loop stops once we've COMMITTED target_data_size groups; inner loop tops up
    # submissions whenever in-flight admitted groups drop below the commit target, each
    # round admitting a full over_sampling_batch_size so the surplus absorbs aborts.
    while not target_reached():
        while state.remaining_batch_size < submit_target:
            _t_get_samples = monotonic()

            if state.prefetched_samples_ref is not None:
                ref = state.prefetched_samples_ref
                state.prefetched_samples_ref = None
                logger.info(f"Rollout step {rollout_id}: using pre-fetched data from previous step")
            else:
                ref = data_source.get_samples.remote(args.over_sampling_batch_size + num_old_samples)

            samples = await loop.run_in_executor(None, ray.get, ref)

            get_samples_times.append(monotonic() - _t_get_samples)
            state.submit_generate_tasks(samples)
        # wait for the generation to finish (from both normal and protected pending sets)
        all_pendings = state.pendings | state.protected_pendings
        done, remaining = await asyncio.wait(all_pendings, return_when=asyncio.FIRST_COMPLETED)
        state.pendings = state.pendings & remaining
        state.protected_pendings = state.protected_pendings & remaining
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # Classify each finished group. Nothing is dropped: aborted groups carry to
            # the buffer for partial resume; completed groups beyond target_data_size are
            # over-sampling surplus carried to the buffer (reused next step); completed
            # groups within target are committed to the transfer queue.
            group_aborted = any(sample.status == Sample.Status.ABORTED for sample in group)
            should_commit = (
                total_transfer_samples < target_data_size if is_final_backfill else len(data) < target_data_size
            )
            if group_aborted:
                for sample in group:
                    if sample.response and "start_rollout_id" not in sample.metadata:
                        sample.metadata["start_rollout_id"] = rollout_id
                aborted_samples.append(group)
            elif should_commit:
                batch_to_transfer.append(group)
                total_transfer_samples += 1
            else:
                # Over-sampling surplus: target already met. Keep it for the next step
                # (added back to the buffer after this step) instead of dropping it.
                oversample_surplus.append(group)

            if (is_final_backfill and should_commit and not group_aborted) or (
                not is_final_backfill and len(data) < target_data_size
            ):
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

        # Only spawn a transfer task when there are samples to transfer.
        transfer_batch_size = (
            args.global_batch_size // args.num_iters_per_train_update // args.n_samples_per_prompt
            if args.fully_async
            else args.rollout_batch_size
        )  # Samples per batch to transfer
        # in fully async mode, we transfer all remaining samples when we reach the target size
        if len(batch_to_transfer) >= transfer_batch_size:
            if total_transfer_samples <= num_old_samples:
                n = len(batch_to_transfer)
                # is_last: this backfill closes the previous partition's debt.
                prev_is_last = args.fully_async and (committed_prev + n >= prev_target)
                transfer_task = asyncio.create_task(
                    transfer_batch_to_data_system(
                        args,
                        batch_to_transfer,
                        n,
                        rollout_id - 1,
                        data_system_client,
                        is_last=prev_is_last,
                    )
                )
                committed_prev += n
                transfer_tasks.append(transfer_task)
                batch_to_transfer = []
                logger.info(f"Total yielded: {total_transfer_samples}/{num_old_samples} for step: {rollout_id - 1}")
            else:
                if len(batch_to_transfer) > total_transfer_samples - num_old_samples:
                    cutoff_batch = len(batch_to_transfer) - total_transfer_samples + num_old_samples
                    n_prev = len(batch_to_transfer[:cutoff_batch])
                    # This split sends the remaining backfill to the previous partition;
                    # it always closes the debt, so it is the previous partition's last.
                    prev_is_last = args.fully_async and (committed_prev + n_prev >= prev_target)
                    transfer_task = asyncio.create_task(
                        transfer_batch_to_data_system(
                            args,
                            batch_to_transfer[:cutoff_batch],
                            n_prev,
                            rollout_id - 1,
                            data_system_client,
                            is_last=prev_is_last,
                        )
                    )
                    committed_prev += n_prev
                    transfer_tasks.append(transfer_task)
                    batch_to_transfer = batch_to_transfer[cutoff_batch:]
                    logger.info(
                        f"{num_old_samples} old samples completed! Total yielded: {num_old_samples}/{num_old_samples} for step: {rollout_id - 1}"
                    )
                else:
                    n = len(batch_to_transfer)
                    # is_last for the current partition ONLY if this step fully meets its
                    # target (no deficit carried) — otherwise the tail is backfilled next step.
                    curr_is_last = args.fully_async and (committed_curr + n >= curr_target)
                    transfer_task = asyncio.create_task(
                        transfer_batch_to_data_system(
                            args,
                            batch_to_transfer,
                            n,
                            rollout_id,
                            data_system_client,
                            is_last=curr_is_last,
                        )
                    )
                    committed_curr += n
                    transfer_tasks.append(transfer_task)
                    batch_to_transfer = []
                    logger.info(
                        f"Total yielded: {total_transfer_samples - num_old_samples}/{args.rollout_batch_size} for step: {rollout_id}"
                    )

    if len(batch_to_transfer) > 0:
        n = len(batch_to_transfer)
        if is_final_backfill:
            prev_is_last = args.fully_async and (committed_prev + n >= prev_target)
            transfer_task = asyncio.create_task(
                transfer_batch_to_data_system(
                    args,
                    batch_to_transfer,
                    n,
                    rollout_id - 1,
                    data_system_client,
                    is_last=prev_is_last,
                )
            )
            committed_prev += n
            transfer_tasks.append(transfer_task)
            batch_to_transfer = []
            logger.info(f"Total yielded: {committed_prev}/{num_old_samples} for step: {rollout_id - 1}")
        else:
            # Tail flush to the current partition: last only if it completes this step's target.
            curr_is_last = args.fully_async and (committed_curr + n >= curr_target)
            transfer_task = asyncio.create_task(
                transfer_batch_to_data_system(
                    args,
                    batch_to_transfer,
                    n,
                    rollout_id,
                    data_system_client,
                    is_last=curr_is_last,
                )
            )
            committed_curr += n
            transfer_tasks.append(transfer_task)
            batch_to_transfer = []
            logger.info(
                f"Total yielded: {total_transfer_samples - num_old_samples}/{args.rollout_batch_size} for step: {rollout_id}"
            )

    logger.info(f"Generator exhausted. Waiting for {len(transfer_tasks)} transfer tasks to complete...")
    # Wait for all transfer tasks to complete
    if transfer_tasks:
        await asyncio.gather(*transfer_tasks)
    pbar.close()

    # Stop SGLang profiling if enabled (no-op if num_steps was set — SGLang auto-stops)
    await stop_sglang_profile(args, rollout_id)

    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
    )

    rollout_time = timer.end("rollout")

    all_samples = [sample for group in data for sample in (group if isinstance(group, list) else [group])]
    timing_metrics = _aggregate_rollout_timing(all_samples, get_samples_times)

    # there are still some unfinished requests, abort them
    # abort() returns (aborted_samples, completed_protected_samples)
    new_aborted, completed_protected = await abort(args, rollout_id)
    aborted_samples.extend(new_aborted)
    aborted_samples.extend(completed_protected)
    if aborted_samples:
        logger.info(
            f"Rollout not completed for rollout_id: {rollout_id}, have {len(aborted_samples)} samples aborted."
        )
    else:
        logger.info(f"Rollout fully completed for rollout_id: {rollout_id}.")

    # Record this step's current-partition deficit for the next step's backfill debt.
    # committed_current = groups committed to rollout_id (current partition); the first
    # num_old_samples committed went to rollout_id-1 (previous-partition backfill).
    committed_current = 0 if is_final_backfill else max(total_transfer_samples - num_old_samples, 0)
    if is_final_backfill:
        state.last_step_current_deficit = 0
    elif args.fully_async:
        state.last_step_current_deficit = max(args.rollout_batch_size - committed_current, 0)
    else:
        state.last_step_current_deficit = 0

    # Carry over-sampling surplus back to the buffer (reused next step, committed first
    # via the completed-group fast path in generate_and_rm_group). Surplus groups are
    # complete (length == n_samples_per_prompt), satisfying add_samples' assertion.
    aborted_samples.extend(oversample_surplus)
    logger.info(
        f"Rollout step {rollout_id} carry-over: committed_current={committed_current} "
        f"next_step_deficit={state.last_step_current_deficit} "
        f"oversample_surplus={len(oversample_surplus)} aborted={len(aborted_samples) - len(oversample_surplus)}"
    )

    # When dynamic global batch size is enabled, extract fully-completed groups
    # from aborted_samples and send them to training instead of carrying over.
    if args.partial_rollout and args.use_dynamic_global_batch_size:
        extra_completed = [
            group
            for group in aborted_samples
            if all(s.status in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) for s in group)
        ]
        if extra_completed:
            aborted_samples = [
                group
                for group in aborted_samples
                if not all(s.status in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) for s in group)
            ]
            # Trim so total groups in TQ is divisible by dp_size (required by SeqlenBalancedSampler)
            dp_size = compute_dp_size(args)
            max_total = len(data) + len(extra_completed)
            max_extra = max_total - (max_total % dp_size) - len(data)
            accepted = extra_completed[:max_extra]
            surplus = extra_completed[max_extra:]
            aborted_samples.extend(surplus)
            for group in accepted:
                data.append(group)
            if accepted:
                await transfer_batch_to_data_system(args, accepted, len(accepted), rollout_id, data_system_client)
            logger.info(f"Transferred {len(accepted)} extra completed groups to training ")

    global CURRENT_ROLLOUT_BATCH
    if CURRENT_ROLLOUT_BATCH:
        save_debug_rollout_data(
            args, CURRENT_ROLLOUT_BATCH, rollout_id=rollout_id, evaluation=False, tokenizer=state.tokenizer
        )
        rollout_metrics = dict(timing_metrics)
        if args.partial_rollout and not args.fully_async:
            assert len(CURRENT_ROLLOUT_BATCH) == len(data) * args.n_samples_per_prompt, (
                f"len(CURRENT_ROLLOUT_BATCH)={len(CURRENT_ROLLOUT_BATCH)}, len(data) * args.n_samples_per_prompt={len(data) * args.n_samples_per_prompt}"
            )
        _log_rollout_data(rollout_id, args, CURRENT_ROLLOUT_BATCH, rollout_metrics, rollout_time)
        if args.debug_rollout_only:
            logger.info("Debug rollout only mode - data system cleanup")
            await data_system_client.async_clear_partition(partition_id=f"train_{rollout_id}")
        # Cleanup
        CURRENT_ROLLOUT_BATCH.clear()

    state.reset()

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:

    state = GenerateState(args)
    # Increment evaluating counter so that abort() knows to wait for eval to finish.
    # This prevents abort_all from killing in-flight eval requests on SGLang workers.
    state.evaluating += 1
    try:
        coros = []
        for dataset_cfg in getattr(args, "eval_datasets", []) or []:
            coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
        results_list = await asyncio.gather(*coros)
        results = {}
        for r in results_list:
            results.update(r)
        return RolloutFnEvalOutput(data=results), []
    finally:
        state.evaluating -= 1


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm
    rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    global EVAL_PROMPT_DATASET

    prepare_model_maybe_update_args(args, completeness="metadata")
    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            use_audio_in_video=args.use_audio_in_video,
            system_prompt=args.system_prompt,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    sample_index = 0

    if args.group_rm:
        # group_rm mode: group samples by prompt and use generate_and_rm_group
        # so that the RM can see all responses for the same prompt together.
        tasks = []
        for _i, prompt_sample in enumerate(dataset.samples):
            group = []
            for j in range(dataset_cfg.n_samples_per_eval_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.index = sample_index
                sample_index += 1
                sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
                sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
                group.append(sample)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed
            tasks.append(
                asyncio.create_task(
                    generate_and_rm_group(args, group, sampling_params=sampling_params, evaluation=True)
                )
            )

        data = []
        do_print = True
        pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
        for coro in asyncio.as_completed(tasks):
            group = await coro
            if do_print:
                sample = group[0]
                logger.info(
                    "eval_rollout_single_dataset example data: "
                    f"{[str(sample.prompt) + sample.response]} "
                    f"reward={sample.reward}"
                )
                do_print = False
            data.extend(group)
            pbar.update(1)
        pbar.close()
    else:
        tasks = []
        for _i, prompt_sample in enumerate(dataset.samples):
            for j in range(dataset_cfg.n_samples_per_eval_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.index = sample_index
                sample_index += 1
                sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
                sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
                sampling_params = base_sampling_params
                if getattr(args, "sglang_enable_deterministic_inference", False):
                    sampling_params = base_sampling_params.copy()
                    sampling_params["sampling_seed"] = args.rollout_seed + j
                tasks.append(
                    asyncio.create_task(
                        generate_and_rm(args, sample, sampling_params=sampling_params, evaluation=True)
                    )
                )

        data = []
        do_print = True
        pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
        for coro in asyncio.as_completed(tasks):
            sample = await coro
            if do_print:
                logger.info(
                    "eval_rollout_single_dataset example data: "
                    f"{[str(sample.prompt) + sample.response]} "
                    f"reward={sample.reward}"
                )
                do_print = False
            if isinstance(sample, list):
                data.extend(sample)
            else:
                data.append(sample)
            pbar.update(1)
        pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_buffer: Any, data_system_client: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based
    rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        data_system_client: the data system client to use for transferring batches
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        RolloutFnTrainOutput | RolloutFnEvalOutput: the output of the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_buffer, data_system_client))
    if aborted_samples:
        ray.get(data_buffer.add_samples.remote(aborted_samples))
    # LoRA adapter mode disables next-step prefetch: the per-step adapter update is ~1s, so a
    # prefetched generation is almost always aborted by the next step's pause_generation and its
    # truncated samples would pollute the next rollout. Other modes keep the optimization.
    _lora_adapter_mode = getattr(args, "lora_rank", 0) > 0 and getattr(args, "lora_adapter_mode", False)
    if not args.fully_async and not _lora_adapter_mode:
        state = GenerateState(args)
        state.prefetched_samples_ref = data_buffer.get_samples.remote(args.over_sampling_batch_size)
        logger.info(f"Rollout step {rollout_id}: pre-submitted data fetch for next step")
    return output
