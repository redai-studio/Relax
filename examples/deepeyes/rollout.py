from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pybase64
import torch

from examples.deepeyes.base_env import BaseInteractionEnv
from relax.engine.rollout.request_permit import GenerationAborted
from relax.engine.rollout.sglang_rollout import GenerateState
from relax.utils.data.processing_utils import encode_image_for_rollout_engine, get_encode_executor
from relax.utils.types import Sample


DEFAULT_ENV_MODULE = "examples.deepeyes.env_deepeyes"

# Dummy messages used for calculating trim length in chat template encoding
DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _load_env_module(env_path: str | None):
    """Load the interaction environment module from a module path or a file
    path."""
    target = env_path or DEFAULT_ENV_MODULE
    module_path = Path(target)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location(f"rollout_env_{module_path.stem}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import environment module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


def _build_env(env_module, sample: Sample, args: Any):
    """Instantiate the interaction environment using the provided module."""
    build_fn = env_module.build_env
    if not callable(build_fn):
        raise ValueError("Environment module must expose a callable `build_env(sample, args)`.")
    try:
        return build_fn(sample=sample, args=args)
    except TypeError:
        # Fallback to positional signature
        return build_fn(sample, args)


def _encode_observation_for_generation(
    tokenizer,
    processor,
    message: dict,
    metadata: dict | None,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
):
    """Encode a single observation turn that may include images/videos in the
    content list.

    Trim out the system/tool preamble added by the chat template so only the
    observation tokens remain.
    """
    tools = metadata.get("tools") if metadata else None
    apply_kwargs = apply_chat_template_kwargs or {}

    trim_length = 0

    if apply_chat_template:
        dummy_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            **apply_kwargs,
        )
        formatted_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES + [message],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **apply_kwargs,
        )
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
    else:
        formatted_prompt = [message]

    multimodal_inputs = None
    multimodal_train_inputs = None
    if processor:
        # Convert content-embedded images/videos into multimodal inputs for the processor.
        from relax.utils.data.processing_utils import process_vision_info

        multimodal_inputs = process_vision_info([message], processor, use_audio_in_video=False)
        processor_output = processor(text=formatted_prompt, **multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    if trim_length:
        prompt_ids = prompt_ids[trim_length:]

    image_data = []
    if multimodal_inputs and multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in multimodal_inputs["images"]]
    return prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs


def _merge_multimodal_train_inputs(chunks: list[dict | None]) -> dict | None:
    """Merge per-turn multimodal_train_inputs with a single concat per key.

    Note: Only torch.Tensor values are merged; non-tensor fields are ignored by design.
    """
    if not chunks:
        return None

    values_by_key = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, val in chunk.items():
            if val is None:
                continue
            values_by_key.setdefault(key, []).append(val)

    merged = {}
    for key, values in values_by_key.items():
        if all(isinstance(v, torch.Tensor) for v in values):
            merged[key] = torch.cat(values, dim=0)

    return merged


def _initialize_resources(args: Any, sample: Sample):
    env_module = _load_env_module(args.rollout_interaction_env_path)
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sample.metadata = sample.metadata or {}
    env = _build_env(env_module, sample, args)
    config = {"max_turns": max_turns}
    return env, env_module, config, state, url


def _prepare_initial_inputs(sample: Sample, processor, tokenizer):
    if processor:
        processor_output = processor(text=sample.prompt, **(sample.multimodal_inputs or {}))
        prompt_ids = processor_output["input_ids"][0]
        init_mm_train = {k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]} or None
    else:
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
        init_mm_train = None

    image_data = []
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in sample.multimodal_inputs["images"]]
    return prompt_ids, image_data, init_mm_train


async def _prepare_start_state(sample: Sample, state, args: Any, sampling_params: dict, is_resuming: bool = False):
    loop = asyncio.get_running_loop()
    prompt_ids, image_data, init_mm_train = await loop.run_in_executor(
        get_encode_executor(),
        _prepare_initial_inputs,
        sample,
        state.processor,
        state.tokenizer,
    )

    if is_resuming:
        saved_image_data = sample.metadata.get("_current_image_data")
        current_image_data = saved_image_data if saved_image_data is not None else image_data
    else:
        current_image_data = image_data

    saved_mm_buffer = sample.metadata.get("_multimodal_train_inputs_buffer") if sample.metadata else None
    if saved_mm_buffer is not None:
        multimodal_train_inputs_buffer: list[dict | None] = saved_mm_buffer
    else:
        multimodal_train_inputs_buffer: list[dict | None] = []
        if init_mm_train:
            multimodal_train_inputs_buffer.append(init_mm_train)

    if not sample.tokens:
        sample.tokens = list(prompt_ids)
    response_tokens: list[int] = sample.tokens[len(prompt_ids) :] if len(sample.tokens) >= len(prompt_ids) else []
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []
    sample.response_length = len(response_tokens)

    context_budget = (
        args.rollout_max_context_len - len(sample.tokens) if args.rollout_max_context_len is not None else None
    )
    generation_budget = None
    if is_resuming and sampling_params.get("max_new_tokens") is not None:
        current_turn_used = _current_turn_generated_token_count(sample, sample.response_length)
        generation_budget = max(0, int(sampling_params["max_new_tokens"]) - current_turn_used)
    return current_image_data, response_tokens, context_budget, generation_budget, multimodal_train_inputs_buffer


async def _run_inference_step(
    state: GenerateState, url: str, tokens: list[int], sampling_params: dict, image_data, tokenizer, args=None
):
    payload = {
        "input_ids": tokens,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if args and getattr(args, "use_rollout_routing_replay", False):
        payload["return_routed_experts"] = True
    if image_data:
        payload["image_data"] = image_data

    # Per-request permit: one permit == this one in-flight model request. The
    # permit is released as soon as post() returns, before env/tool execution.
    output = await state.post_generate(url, payload)
    response_text = output["text"]
    if "output_token_logprobs" in output["meta_info"]:
        new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_tokens, new_log_probs = [], []
    finish_type = output["meta_info"]["finish_reason"]["type"]
    meta_info = output["meta_info"]
    return response_text, new_tokens, new_log_probs, finish_type, meta_info


async def _process_env_step(env: BaseInteractionEnv, response_text: str, tokenizer, processor, args, sample_metadata):
    result = env.step(response_text)
    # 兼容 async env.step（如 VideoSearchEnv）：若返回 coroutine 则 await
    if inspect.isawaitable(result):
        result = await result
    observation, done, info = result
    if done:
        return None, None, None, None, True, info

    next_user_message = env.format_observation(observation)
    loop = asyncio.get_running_loop()
    obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs = await loop.run_in_executor(
        get_encode_executor(),
        _encode_observation_for_generation,
        tokenizer,
        processor,
        next_user_message,
        sample_metadata,
        args.apply_chat_template,
        args.apply_chat_template_kwargs,
    )

    bos_id = tokenizer.bos_token_id
    if bos_id is not None and obs_prompt_ids and obs_prompt_ids[0] == bos_id:
        obs_prompt_ids = obs_prompt_ids[1:]

    return obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs, False, info


def _append_to_sample(
    sample: Sample,
    response_tokens: list[int],
    tokens_to_add: list[int],
    logprobs: list[float],
    loss_mask_val: int,
) -> None:
    sample.tokens.extend(tokens_to_add)
    response_tokens.extend(tokens_to_add)
    sample.loss_mask.extend([loss_mask_val] * len(tokens_to_add))
    sample.rollout_log_probs.extend(logprobs)
    sample.response_length = len(response_tokens)


def _update_multimodal_state(
    current_image_data,
    obs_image_data,
    obs_multimodal_train_inputs,
    multimodal_train_inputs_buffer: list[dict | None],
):
    if obs_image_data:
        current_image_data = (current_image_data or []) + obs_image_data

    if obs_multimodal_train_inputs:
        multimodal_train_inputs_buffer.append(obs_multimodal_train_inputs)

    return current_image_data


def _should_stop_on_finish(sample: Sample, finish_type: str) -> str | None:
    match finish_type:
        case "length":
            sample.status = Sample.Status.TRUNCATED
            return "finish_length"
        case "abort":
            sample.status = Sample.Status.ABORTED
            return "finish_abort"
    return None


def _update_budget(budget, consumed: int):
    return None if budget is None else budget - consumed


def _current_turn_generated_token_count(sample: Sample, response_length: int) -> int:
    turn_start = sample.metadata.get("_current_turn_response_start") if sample.metadata else None
    return response_length - turn_start if isinstance(turn_start, int) and 0 <= turn_start <= response_length else 0


def _update_routed_experts(sample: Sample, meta_info: dict, args: Any) -> None:
    """Decode and store routed experts from meta_info for routing replay (MoE
    models).

    In multi-turn inference, each turn's routed experts covers the entire token
    sequence accumulated so far (including observation tokens from previous
    turns), so the result from the last turn is the final complete routing
    information.
    """
    if "routed_experts" not in meta_info:
        return
    sample.rollout_routed_experts = np.frombuffer(
        pybase64.b64decode(meta_info["routed_experts"].encode("ascii")),
        dtype=np.int32,
    ).reshape(
        len(sample.tokens) - 1,
        args.num_layers,
        args.moe_router_topk,
    )


def _finalize_sample(sample: Sample, tokenizer, response_tokens, multimodal_train_inputs_buffer):
    sample.multimodal_train_inputs = _merge_multimodal_train_inputs(multimodal_train_inputs_buffer)
    sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    if sample.status is None:
        sample.status = Sample.Status.COMPLETED
    return sample


class _RolloutTraceRecorder:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._trace = None

    def _decode_tokens(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=False) if tokens else ""

    @staticmethod
    def _summarize_image_data(image_data):
        if not image_data:
            return None
        if isinstance(image_data, list):
            return {"count": len(image_data), "item_types": [type(item).__name__ for item in image_data]}
        return {"type": type(image_data).__name__}

    def start(self, turn_idx: int, sample_tokens, sampling_params, current_image_data, budget):
        self._trace = {
            "turn_index": turn_idx,
            "inference": {
                "input": {
                    "tokens": self._decode_tokens(list(sample_tokens)),
                    "sampling_params": sampling_params,
                    "image_data": self._summarize_image_data(current_image_data),
                    "budget": budget,
                }
            },
        }
        return self._trace

    def record_inference_output(self, response_text: str, finish_type: str, elapsed: float) -> None:
        self._trace["inference"]["output"] = {
            "response_text": response_text,
            "finish_type": finish_type,
            "elapsed": elapsed,
        }

    def record_env_step(self, obs_prompt_ids, obs_image_data, done, info, elapsed: float) -> None:
        self._trace["env_step"] = {
            "obs_prompt_ids": self._decode_tokens(obs_prompt_ids or []),
            "obs_image_data": self._summarize_image_data(obs_image_data),
            "done": done,
            "info": info,
            "elapsed": elapsed,
        }


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    """Custom multi-turn rollout that interacts with a pluggable
    environment."""

    env, env_module, config, state, url = _initialize_resources(args, sample)
    sampling_params = sampling_params.copy()

    is_resuming = sample.status == Sample.Status.ABORTED and sample.response_length > 0

    if (
        getattr(args, "partial_rollout", False)
        and is_resuming
        and getattr(args, "mask_offpolicy_in_partial_rollout", False)
        and sample.response_length > 0
    ):
        sample.loss_mask = [0] * sample.response_length

    (
        current_image_data,
        response_tokens,
        context_budget,
        generation_budget,
        multimodal_train_inputs_buffer,
    ) = await _prepare_start_state(sample, state, args, sampling_params, is_resuming=is_resuming)
    rollout_traces = sample.metadata.setdefault("rollout_traces", [])

    resume_turn = 0
    if is_resuming:
        resume_turn = sample.metadata.get("rollout_turns", len(rollout_traces))
        sample.status = Sample.Status.PENDING

    def _is_context_budget_exhausted() -> bool:
        return context_budget is not None and context_budget <= 0

    def _record_rollout_stats(stop_reason: str) -> None:
        sample.metadata["rollout_turns"] = turns_executed
        sample.metadata["rollout_stop_reason"] = stop_reason

    turns_executed = resume_turn
    stop_reason = None
    try:
        env.reset()

        if is_resuming:
            env.turn = resume_turn
            saved_image = sample.metadata.get("_env_current_image")
            if saved_image is not None:
                env.current_image = saved_image

        if _is_context_budget_exhausted() or (generation_budget is not None and generation_budget <= 0):
            sample.status = Sample.Status.TRUNCATED
            stop_reason = "budget_exhausted"
            sample.metadata.pop("_current_turn_response_start", None)
            _record_rollout_stats(stop_reason)
            return _finalize_sample(sample, state.tokenizer, response_tokens, multimodal_train_inputs_buffer)

        trace_recorder = _RolloutTraceRecorder(state.tokenizer)
        for turn_idx in range(resume_turn, config["max_turns"]):
            turns_executed = turn_idx + 1
            cur_sampling_params = sampling_params.copy()
            active_budget = context_budget
            if generation_budget is not None:
                active_budget = generation_budget if active_budget is None else min(active_budget, generation_budget)
            if active_budget is not None:
                active_budget = max(0, int(active_budget))
                if cur_sampling_params.get("max_new_tokens") is not None:
                    active_budget = min(active_budget, int(cur_sampling_params["max_new_tokens"]))
                cur_sampling_params["max_new_tokens"] = active_budget

            turn_start = sample.metadata.get("_current_turn_response_start")
            if not isinstance(turn_start, int) or not 0 <= turn_start <= len(response_tokens):
                sample.metadata["_current_turn_response_start"] = len(response_tokens)

            turn_record = trace_recorder.start(
                turn_idx, sample.tokens, cur_sampling_params, current_image_data, active_budget
            )

            inference_start_ts = time.time()
            try:
                (
                    response_text,
                    new_response_tokens,
                    new_response_log_probs,
                    finish_type,
                    meta_info,
                ) = await _run_inference_step(
                    state, url, sample.tokens, cur_sampling_params, current_image_data, state.tokenizer, args=args
                )
            except GenerationAborted:
                # Abort was signalled before this turn's request went out. Mirror the
                # in-flight finish_abort path: mark ABORTED, do not count this turn, and
                # keep _current_turn_response_start so resume continues this same turn.
                sample.status = Sample.Status.ABORTED
                stop_reason = "finish_abort"
                turns_executed = turn_idx
                rollout_traces.append(turn_record)
                break
            inference_end_ts = time.time()
            trace_recorder.record_inference_output(
                response_text, finish_type, max(0.0, inference_end_ts - inference_start_ts)
            )
            _append_to_sample(sample, response_tokens, new_response_tokens, new_response_log_probs, loss_mask_val=1)
            context_budget = _update_budget(context_budget, len(new_response_tokens))
            generation_budget = _update_budget(generation_budget, len(new_response_tokens))

            _update_routed_experts(sample, meta_info, args)

            finish_reason = _should_stop_on_finish(sample, finish_type)
            if finish_reason:
                stop_reason = finish_reason
                if finish_reason == "finish_abort":
                    turns_executed = turn_idx
                else:
                    sample.metadata.pop("_current_turn_response_start", None)
                rollout_traces.append(turn_record)
                break
            if _is_context_budget_exhausted():
                sample.status = Sample.Status.TRUNCATED
                stop_reason = stop_reason or "budget_exhausted"
                sample.metadata.pop("_current_turn_response_start", None)
                rollout_traces.append(turn_record)
                break
            generation_budget = None

            env_start_ts = time.time()
            (
                obs_prompt_ids,
                obs_image_data,
                obs_multimodal_inputs,
                obs_multimodal_train_inputs,
                done,
                info,
            ) = await _process_env_step(env, response_text, state.tokenizer, state.processor, args, sample.metadata)
            env_end_ts = time.time()

            trace_recorder.record_env_step(
                obs_prompt_ids,
                obs_image_data,
                done,
                info,
                max(0.0, env_end_ts - env_start_ts),
            )
            if done:
                sample.status = Sample.Status.COMPLETED
                stop_reason = stop_reason or "env_done"
                sample.metadata.pop("_current_turn_response_start", None)
                rollout_traces.append(turn_record)
                break

            obs_log_probs = [0.0] * len(obs_prompt_ids)
            _append_to_sample(sample, response_tokens, obs_prompt_ids, obs_log_probs, loss_mask_val=0)
            context_budget = _update_budget(context_budget, len(obs_prompt_ids))
            sample.metadata.pop("_current_turn_response_start", None)

            current_image_data = _update_multimodal_state(
                current_image_data,
                obs_image_data,
                obs_multimodal_train_inputs,
                multimodal_train_inputs_buffer,
            )

            # Snapshot state for aborted rollout resumption.
            # Must be AFTER _update_multimodal_state so current_image_data
            # includes images from this turn's observation.
            if hasattr(env, "current_image"):
                sample.metadata["_env_current_image"] = env.current_image
            sample.metadata["_multimodal_train_inputs_buffer"] = multimodal_train_inputs_buffer
            sample.metadata["_current_image_data"] = current_image_data

            if _is_context_budget_exhausted():
                sample.status = Sample.Status.TRUNCATED
                stop_reason = stop_reason or "budget_exhausted"
                rollout_traces.append(turn_record)
                break
            if turn_idx + 1 >= config["max_turns"]:
                sample.status = Sample.Status.COMPLETED
                stop_reason = stop_reason or "max_turns"
                rollout_traces.append(turn_record)
                break
            rollout_traces.append(turn_record)

        if stop_reason is None:
            stop_reason = "completed"
        _record_rollout_stats(stop_reason)
        return _finalize_sample(sample, state.tokenizer, response_tokens, multimodal_train_inputs_buffer)
    finally:
        try:
            env.close()
        except Exception:
            pass


# Opt-in: this custom rollout manages its own per-request inference permits
# (via state.post_generate per turn), so generate_and_rm must NOT wrap it in the
# session-level lock. See docs/{zh,en}/guide/customize-training.md.
generate.manages_inference_permit = True
