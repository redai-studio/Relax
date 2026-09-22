# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Runtime control plane over group-affine Ray SessionShards."""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
import zlib
from argparse import Namespace
from concurrent.futures import Executor
from dataclasses import dataclass, field
from functools import partial
from io import BytesIO
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypeVar, cast

import httpx
import numpy as np
import ray
import torch

from relax.agentic import AGENTIC_CHAT_API_SERVICE_NAME, format_agentic_status
from relax.agentic.pipeline import (
    GroupExport,
    GroupInput,
    RuntimeGroupError,
    SampleExport,
    SessionExport,
    SessionExportTransport,
    SessionShardProgress,
    SessionSpec,
    TrainingFieldArtifact,
)
from relax.agentic.profile import mark_sample_agentic_event
from relax.agentic.session.state import check_messages
from relax.utils.http_utils import get, init_http_client, post, router_worker_base_urls
from relax.utils.logging_utils import get_logger
from relax.utils.multimodal.config import MultimodalConfig
from relax.utils.s3_model_loader import prepare_model_maybe_update_args
from relax.utils.types import Sample


logger = get_logger(__name__)


# Compiler calibration messages used to measure the chat-template prefix. Their
# exact roles and text affect prefix subtraction, so changes require tokenizer
# and template coverage rather than cosmetic editing.
_DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


# OpenAI stores historical function.arguments as JSON text, while current
# Hugging Face tool-use templates such as Qwen3.5 expect a mapping. Match
# SGLang serving_chat by decoding arguments only in this rendering copy.
def _messages_for_chat_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered_messages = list(messages)
    for message_index, message in enumerate(messages):
        tool_calls = message.get("tool_calls") or ()
        rendered_tool_calls = list(tool_calls)
        changed = False
        for call_index, tool_call in enumerate(tool_calls):
            function = tool_call["function"]
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError("Assistant tool call function.arguments must be valid JSON.") from exc
                if not isinstance(arguments, dict):
                    raise ValueError("Assistant tool call function.arguments must be a JSON object.")
                rendered_function = dict(function)
                rendered_function["arguments"] = arguments
                rendered_tool_call = dict(tool_call)
                rendered_tool_call["function"] = rendered_function
                rendered_tool_calls[call_index] = rendered_tool_call
                changed = True
        if changed:
            rendered_message = dict(message)
            rendered_message["tool_calls"] = rendered_tool_calls
            rendered_messages[message_index] = rendered_message
    return rendered_messages


@dataclass(frozen=True)
class AgenticCompilerResources:
    """CPU resources shared by one message compiler."""

    tokenizer: Any
    processor: Any
    cpu_executor: Executor
    processor_pool: Any | None = None

    def shutdown(self) -> None:
        if self.processor_pool is not None:
            self.processor_pool.shutdown(wait=False)


def load_agentic_compiler_resources(args: Namespace) -> AgenticCompilerResources:
    """Load compiler resources used by one SessionShard."""

    from relax.utils.data.processing_utils import configure_encode_executor, load_processor, load_tokenizer

    prepare_model_maybe_update_args(args, completeness="metadata")
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
    cpu_executor = configure_encode_executor(args.encode_max_workers)
    processor_pool = None
    if args.mm_processor_pool_size > 0:
        from relax.utils.data.processor_pool import ProcessorPool

        processor_pool = ProcessorPool(
            model_path=args.hf_checkpoint,
            pool_size=args.mm_processor_pool_size,
            trust_remote_code=True,
        )
    return AgenticCompilerResources(
        tokenizer=tokenizer,
        processor=processor,
        cpu_executor=cpu_executor,
        processor_pool=processor_pool,
    )


def _normalize_multimodal_inputs_sync(
    multimodal_inputs: dict[str, Any] | None,
    processor: Any,
    use_audio_in_video: bool,
    multimodal_config: Any | None = None,
) -> tuple[dict[str, Any] | None, dict[str, float]]:
    if not multimodal_inputs:
        return None, {}

    images = list(multimodal_inputs.get("images") or [])
    videos = list(multimodal_inputs.get("videos") or [])
    audio_items = list(multimodal_inputs.get("audio") or [])
    content = [
        {"type": modality, modality: value}
        for modality, values in (("image", images), ("video", videos), ("audio", audio_items))
        for value in values
        if value is not None
    ]
    if not content:
        return None, {}
    if processor is None:
        raise RuntimeError("Agentic multimodal inputs require a processor loaded from the model checkpoint.")
    from relax.utils.data.processing_utils import process_vision_info

    started_at = time.monotonic()
    return process_vision_info(
        [{"role": "user", "content": content}], processor, use_audio_in_video, multimodal_config
    ), {"process_vision_info_elapsed_s": time.monotonic() - started_at}


@dataclass
class EncodedMessages:
    """Compiler output consumed by SessionForest and the generation backend."""

    train_prompt_ids: list[int]
    backend_prompt_ids: list[int]
    backend_image_data: list[str] = field(default_factory=list)
    backend_audio_data: list[str] = field(default_factory=list)
    backend_video_data: list[str] = field(default_factory=list)
    multimodal_train_inputs: dict[str, Any] | None = None
    timing: dict[str, float] = field(default_factory=dict)


@dataclass
class BackendGenerateResult:
    """One SGLang-shaped generation attempt for a stable IR."""

    new_tokens: list[int]
    new_log_probs: list[float]
    finish_type: str
    meta_info: dict[str, Any]
    elapsed: float


class BackendContextLengthExceededError(RuntimeError):
    """The generation backend rejected a request whose context is too long."""


class SGLangMessageCompiler:
    """Message compiler used by the Shard-owned SGLang adapter."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        processor: Any,
        processor_pool: Any | None = None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
        use_audio_in_video: bool = False,
        multimodal_config: Any | None = None,
        cpu_executor: Executor,
    ) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.processor_pool = processor_pool
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.use_audio_in_video = use_audio_in_video
        self.multimodal_config = multimodal_config
        self.cpu_executor = cpu_executor

    def _build_prompt_sync(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> tuple[str, list[int], int, dict[str, float]]:
        template_started_at = time.monotonic()
        prompt_text = self.tokenizer.apply_chat_template(
            _messages_for_chat_template(messages),
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
        template_elapsed_s = time.monotonic() - template_started_at

        tokenize_started_at = time.monotonic()
        backend_prompt_ids = list(self.tokenizer.encode(prompt_text, add_special_tokens=False))
        tokenize_elapsed_s = time.monotonic() - tokenize_started_at
        return (
            prompt_text,
            backend_prompt_ids,
            0,
            {
                "apply_chat_template_elapsed_s": template_elapsed_s,
                "tokenizer_encode_elapsed_s": tokenize_elapsed_s,
            },
        )

    def _build_observation_prompt_sync(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> tuple[str, list[int], int, dict[str, float]]:
        template_kwargs = chat_template_kwargs or {}
        rendered_messages = _messages_for_chat_template(messages)
        template_started_at = time.monotonic()
        dummy_prompt = self.tokenizer.apply_chat_template(
            _DUMMY_MESSAGES,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            **template_kwargs,
        ).rstrip("\n")
        formatted_prompt = self.tokenizer.apply_chat_template(
            _DUMMY_MESSAGES + rendered_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        template_elapsed_s = time.monotonic() - template_started_at

        tokenize_started_at = time.monotonic()
        trim_length = len(self.tokenizer.encode(dummy_prompt, add_special_tokens=False))
        backend_prompt_ids = list(self.tokenizer.encode(formatted_prompt, add_special_tokens=False))
        tokenize_elapsed_s = time.monotonic() - tokenize_started_at
        if trim_length:
            backend_prompt_ids = backend_prompt_ids[trim_length:]
        return (
            formatted_prompt,
            backend_prompt_ids,
            trim_length,
            {
                "apply_chat_template_elapsed_s": template_elapsed_s,
                "tokenizer_encode_elapsed_s": tokenize_elapsed_s,
            },
        )

    async def _run_processor_async(
        self,
        prompt_text: str,
        multimodal_inputs: dict[str, Any],
        *,
        trim_length: int = 0,
    ) -> tuple[list[int], dict[str, Any] | None, dict[str, float]]:
        loop = asyncio.get_running_loop()
        started_at = time.monotonic()
        if self.processor_pool is not None:
            from relax.utils.data.processor_pool import prepare_mm_inputs_for_ipc, process_sample_in_worker

            mm_inputs_ipc = prepare_mm_inputs_for_ipc(multimodal_inputs)
            processor_kwargs = {
                "use_audio_in_video": self.use_audio_in_video,
                "return_mm_token_type_ids": False,
            }
            train_prompt_ids, multimodal_train_inputs = await loop.run_in_executor(
                self.processor_pool.executor,
                process_sample_in_worker,
                prompt_text,
                mm_inputs_ipc,
                processor_kwargs,
            )
            train_prompt_ids = list(train_prompt_ids)
        else:

            def _run_processor_sync():
                from relax.utils.data.processing_utils import (
                    adapt_processor_kwargs,
                    expand_kimi_k25_placeholders,
                    remap_mm_train_inputs,
                )

                processor_kwargs = adapt_processor_kwargs(
                    self.processor,
                    multimodal_inputs,
                    {
                        "use_audio_in_video": self.use_audio_in_video,
                        "return_mm_token_type_ids": False,
                    },
                )
                processor_output = self.processor(text=prompt_text, **processor_kwargs)
                prompt_ids = list(processor_output["input_ids"][0])
                train_inputs = {
                    key: (torch.from_numpy(value) if isinstance(value, np.ndarray) else value)
                    for key, value in processor_output.items()
                    if key not in {"input_ids", "attention_mask"}
                } or None
                train_inputs = remap_mm_train_inputs(self.processor, train_inputs)
                prompt_ids = expand_kimi_k25_placeholders(self.processor, prompt_ids, train_inputs)
                return prompt_ids, train_inputs

            train_prompt_ids, multimodal_train_inputs = await loop.run_in_executor(
                self.cpu_executor,
                _run_processor_sync,
            )

        if trim_length:
            train_prompt_ids = train_prompt_ids[trim_length:]
        return (
            train_prompt_ids,
            multimodal_train_inputs,
            {"processor_elapsed_s": time.monotonic() - started_at},
        )

    async def _encode_media_async(
        self,
        multimodal_inputs: dict[str, Any] | None,
    ) -> tuple[list[str], list[str], list[str], dict[str, float]]:
        if not multimodal_inputs:
            return [], [], [], {}
        started_at = time.monotonic()
        images = multimodal_inputs.get("images") or []
        videos = multimodal_inputs.get("videos") or []
        audio_items = list(multimodal_inputs.get("audio") or [])

        from relax.utils.data.processing_utils import (
            encode_audio_for_rollout_engine,
            encode_image_for_rollout_engine,
            encode_video_tensor_for_rollout_engine,
        )

        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(self.cpu_executor, encode_image_for_rollout_engine, image) for image in images]
        tasks.extend(
            loop.run_in_executor(
                self.cpu_executor,
                encode_video_tensor_for_rollout_engine,
                video,
            )
            for video in videos
        )
        for audio in audio_items:
            if isinstance(audio, tuple) and len(audio) == 2:
                waveform, sample_rate = audio
                tasks.append(
                    loop.run_in_executor(
                        self.cpu_executor,
                        encode_audio_for_rollout_engine,
                        waveform,
                        sample_rate,
                    )
                )

        if not tasks:
            return [], [], [], {}

        results = await asyncio.gather(*tasks)
        offset = 0
        image_data = results[offset : offset + len(images)]
        offset += len(images)
        video_data = results[offset : offset + len(videos)]
        offset += len(videos)
        audio_data = list(results[offset:])
        return image_data, audio_data, video_data, {"media_encode_elapsed_s": time.monotonic() - started_at}

    async def _encode_with_prompt_builder(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None,
        multimodal_inputs: dict[str, Any] | None,
        prompt_builder: Callable[
            [list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, Any] | None],
            tuple[str, list[int], int, dict[str, float]],
        ],
    ) -> EncodedMessages:
        total_started_at = time.monotonic()
        loop = asyncio.get_running_loop()
        prompt_future = loop.run_in_executor(
            self.cpu_executor,
            prompt_builder,
            messages,
            tools,
            chat_template_kwargs,
        )
        if multimodal_inputs is not None:
            multimodal_future = loop.run_in_executor(
                self.cpu_executor,
                _normalize_multimodal_inputs_sync,
                multimodal_inputs,
                self.processor,
                self.use_audio_in_video,
                self.multimodal_config,
            )
        else:
            multimodal_future = None

        normalized_multimodal_inputs = None
        multimodal_timing: dict[str, float] = {}
        if multimodal_future is None:
            prompt_text, backend_prompt_ids, trim_length, prompt_timing = await prompt_future
        else:
            prompt_result, multimodal_result = await asyncio.gather(prompt_future, multimodal_future)
            prompt_text, backend_prompt_ids, trim_length, prompt_timing = prompt_result
            normalized_multimodal_inputs, multimodal_timing = multimodal_result
        has_media = normalized_multimodal_inputs is not None and any(
            normalized_multimodal_inputs.get(key) for key in ("images", "videos", "audio")
        )

        train_prompt_ids = backend_prompt_ids
        multimodal_train_inputs = None
        processor_timing: dict[str, float] = {}
        media_timing: dict[str, float] = {}
        image_data: list[str] = []
        audio_data: list[str] = []
        video_data: list[str] = []

        if has_media:
            processor_result, media_result = await asyncio.gather(
                self._run_processor_async(prompt_text, normalized_multimodal_inputs, trim_length=trim_length),
                self._encode_media_async(normalized_multimodal_inputs),
            )
            train_prompt_ids, multimodal_train_inputs, processor_timing = processor_result
            image_data, audio_data, video_data, media_timing = media_result

        timing = {
            key: value
            for timings in (prompt_timing, multimodal_timing, processor_timing, media_timing)
            for key, value in timings.items()
        }
        timing["total_elapsed_s"] = time.monotonic() - total_started_at
        return EncodedMessages(
            train_prompt_ids=train_prompt_ids,
            backend_prompt_ids=backend_prompt_ids,
            backend_image_data=image_data,
            backend_audio_data=audio_data,
            backend_video_data=video_data,
            multimodal_train_inputs=multimodal_train_inputs,
            timing=timing,
        )

    async def encode_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None = None,
        multimodal_inputs: dict[str, Any] | None = None,
    ) -> EncodedMessages:
        return await self._encode_with_prompt_builder(
            messages,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
            multimodal_inputs=multimodal_inputs,
            prompt_builder=self._build_prompt_sync,
        )

    async def encode_observation_delta(
        self,
        messages_delta: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None = None,
        multimodal_inputs: dict[str, Any] | None = None,
    ) -> EncodedMessages:
        return await self._encode_with_prompt_builder(
            messages_delta,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
            multimodal_inputs=multimodal_inputs,
            prompt_builder=self._build_observation_prompt_sync,
        )


_Result = TypeVar("_Result")


async def finish_before_cancellation(work: Awaitable[_Result], task_name: str) -> _Result:
    """Finish owned cleanup before forwarding caller cancellation."""

    task = asyncio.ensure_future(work)
    if isinstance(task, asyncio.Task):
        task.set_name(task_name)
    cancellation_received = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.done():
                task.result()
                raise
            cancellation_received = True
    if cancellation_received:
        raise asyncio.CancelledError
    return result


def _sanitize_output_tokens(tokens: list[int], tokenizer: Any, processor: Any) -> list[int]:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        return tokens
    sanitized = tokens
    for token_name in ("image_token_id", "audio_token_id", "video_token_id"):
        special_token_id = getattr(tokenizer, token_name, None)
        if special_token_id is not None:
            sanitized = [pad_token_id if token == special_token_id else token for token in sanitized]
    if processor is not None:
        from relax.utils.data.processing_utils import sanitize_kimi_k25_response_tokens

        sanitized = sanitize_kimi_k25_response_tokens(processor, sanitized)
    return sanitized


def _is_context_length_error(error: httpx.HTTPStatusError) -> bool:
    if error.response.status_code != 400:
        return False
    response_text = error.response.text or ""
    return "maximum context length" in response_text or "Requested token count exceeds" in response_text


class SGLangBackendAdapter:
    """Shard-owned SGLang compiler and HTTP capability."""

    def __init__(self, args: Namespace) -> None:
        init_http_client(args)
        self._args = args
        self._resources = load_agentic_compiler_resources(args)
        self._session_lifecycle = args.agentic_session_lifecycle
        self.tokenizer = self._resources.tokenizer
        self.compiler = SGLangMessageCompiler(
            tokenizer=self._resources.tokenizer,
            processor=self._resources.processor,
            processor_pool=self._resources.processor_pool,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            use_audio_in_video=args.use_audio_in_video,
            multimodal_config=MultimodalConfig.from_args(args),
            cpu_executor=self._resources.cpu_executor,
        )

    @property
    def _router_url(self) -> str:
        return f"http://{self._args.sglang_router_ip}:{self._args.sglang_router_port}"

    async def generate(
        self,
        *,
        input_ids: list[int],
        sampling_params: dict[str, Any],
        session_id: str | None,
        request_id: str,
        image_data: list[str] | None = None,
        audio_data: list[str] | None = None,
        video_data: list[str] | None = None,
        return_logprob: bool = True,
    ) -> BackendGenerateResult:
        payload = {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "rid": request_id,
            "return_logprob": return_logprob,
        }
        if self._args.use_rollout_routing_replay:
            payload["return_routed_experts"] = True
        if image_data:
            payload["image_data"] = image_data
        if audio_data:
            payload["audio_data"] = list(audio_data)
        if video_data:
            payload["video_data"] = video_data
        if session_id and self._session_lifecycle:
            # The full input_ids remain authoritative; session_id only tags
            # the resulting radix leaves for terminal cleanup.
            payload["session_id"] = session_id
        headers = None
        if session_id and (self._args.sglang_router_policy == "consistent_hashing" or self._args.slime_router_sticky):
            headers = {"X-SMG-Routing-Key": session_id}
        started = time.monotonic()
        try:
            output = await post(f"{self._router_url}/generate", payload, headers=headers)
        except httpx.HTTPStatusError as error:
            if _is_context_length_error(error):
                raise BackendContextLengthExceededError(error.response.text) from error
            raise
        elapsed = time.monotonic() - started
        meta_info = dict(output["meta_info"])
        token_logprobs = list(meta_info.get("output_token_logprobs") or ())
        tokens = [int(item[1]) for item in token_logprobs] if token_logprobs else list(output["output_ids"])
        return BackendGenerateResult(
            new_tokens=_sanitize_output_tokens(tokens, self.tokenizer, self.compiler.processor),
            new_log_probs=[float(item[0]) for item in token_logprobs],
            finish_type=str(meta_info["finish_reason"]["type"]),
            meta_info=meta_info,
            elapsed=elapsed,
        )

    async def abort_request(self, request_id: str) -> None:
        urls = await self._worker_urls()
        await asyncio.gather(*(post(f"{url}/abort_request", {"rid": request_id}) for url in urls))

    async def close_session(self, session_id: str, *, timeout_s: float) -> bool:
        """Release one terminal Session from every engine radix cache."""

        if not self._session_lifecycle:
            return True

        async def close_all_workers() -> list[Any]:
            urls = await self._worker_urls()
            return await asyncio.gather(
                *(post(f"{url}/close_session", {"session_id": session_id}) for url in urls),
                return_exceptions=True,
            )

        try:
            outcomes = await asyncio.wait_for(close_all_workers(), timeout=timeout_s)
        except Exception:
            return False
        return all(not isinstance(outcome, BaseException) for outcome in outcomes)

    async def _worker_urls(self) -> list[str]:
        try:
            response = await get(f"{self._router_url}/workers")
            urls = [worker["url"] for worker in response["workers"] if worker["url"] and worker["is_healthy"]]
            if urls:
                return router_worker_base_urls(urls)
        except Exception as workers_error:
            try:
                response = await get(f"{self._router_url}/list_workers")
            except Exception as list_workers_error:
                raise RuntimeError("Failed to query SGLang workers") from list_workers_error
            urls = response["urls"]
            if not urls:
                raise RuntimeError("SGLang returned no workers") from workers_error
            return router_worker_base_urls(urls)
        response = await get(f"{self._router_url}/list_workers")
        return router_worker_base_urls(response["urls"])

    async def shutdown(self) -> None:
        self._resources.shutdown()


def _normalize_prompt(prompt: Any) -> list[dict[str, Any]]:
    if prompt is None or (isinstance(prompt, (dict, list)) and not prompt):
        return []
    if isinstance(prompt, str):
        if not prompt.strip():
            return []
        return check_messages([{"role": "user", "content": prompt}])
    if isinstance(prompt, dict):
        messages = [prompt]
    elif isinstance(prompt, list):
        messages = prompt
    else:
        raise TypeError(f"prompt must be a string, dict, list, or None, got {type(prompt)}")
    if len(messages) == 1 and isinstance(messages[0], dict):
        message = messages[0]
        content = message.get("content")
        if message.get("role") == "user" and (
            content in (None, []) or (isinstance(content, str) and not content.strip())
        ):
            return []
    return check_messages(messages)


def _transport_image_payload(payload: Any) -> str:
    if isinstance(payload, str) and payload.startswith(("data:image/", "http://", "https://")):
        return payload

    from PIL import Image

    from relax.utils.multimodal.image_utils import load_image

    if isinstance(payload, dict):
        if isinstance(payload.get("bytes"), bytes):
            payload = payload["bytes"]
        elif isinstance(payload.get("path"), str) and payload["path"]:
            payload = payload["path"]
    image = payload if isinstance(payload, Image.Image) else load_image(payload)
    buffer = BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _transport_dataset_message_media(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered_messages: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            rendered_messages.append(message)
            continue
        rendered_content: list[dict[str, Any]] = []
        for item in content:
            if item.get("type") == "image":
                rendered_content.append(
                    {"type": "image_url", "image_url": {"url": _transport_image_payload(item.get("image"))}}
                )
                continue
            rendered_content.append(item)
        rendered_message = dict(message)
        rendered_message["content"] = rendered_content
        rendered_messages.append(rendered_message)
    return rendered_messages


def _sample_messages(sample: Sample) -> list[dict[str, Any]]:
    return _transport_dataset_message_media(_normalize_prompt(sample.prompt))


def _build_session_specs(
    samples: Sequence[Sample],
    *,
    session_ids: Sequence[str],
    include_input_payload: bool,
) -> Tuple[SessionSpec, ...]:
    shared_messages = _sample_messages(samples[0]) if include_input_payload else []
    session_specs = []
    for sample, session_id in zip(samples, session_ids, strict=True):
        metadata = sample.metadata
        input_payload: dict[str, Any] = {}
        if include_input_payload:
            input_payload["messages"] = shared_messages
            if metadata:
                input_payload["metadata"] = metadata
        sampling_params = getattr(sample, "sampling_params", None)
        session_specs.append(
            SessionSpec(
                metadata=metadata,
                input_payload=input_payload,
                sampling_params=sampling_params,
                session_id=session_id,
                group_index=sample.group_index,
                index=sample.index,
                label=sample.label,
                train_metadata=sample.train_metadata,
            )
        )
    return tuple(session_specs)


@dataclass(eq=False)
class _SessionShardBinding:
    """One borrowed SessionShard handle and its Runtime-local observer
    state."""

    actor_name: str
    handle: Any
    group_streams: Dict[str, "RuntimeGroupStream"] = field(default_factory=dict)
    observed_revision: int = 0
    revision_changed: asyncio.Event = field(default_factory=asyncio.Event)
    observer_task: Optional["asyncio.Task[None]"] = None
    state_change_ref: Optional[Any] = None


@dataclass(eq=False)
class _SessionResultRef:
    """Driver-local completion ref addressed by the Session token."""

    session_id: str
    completion: "asyncio.Future[SessionExport | None]"
    take_task: Optional["asyncio.Task[None]"] = None


@dataclass(eq=False)
class RuntimeGroupStream:
    """Driver capability for one group owned wholly by one SessionShard.

    It binds group identity, Shard ownership, the first-request barrier, and
    terminal Session export refs.
    """

    group_id: str
    shard: _SessionShardBinding
    first_request_barrier: "asyncio.Future[bool]"
    session_results: Tuple[_SessionResultRef, ...]
    interrupted: bool = False
    protected: bool = False


def agentic_eval_concurrency_from_args(args: Namespace, eval_group_size: int) -> int:
    """Resolve the resident Eval Group capacity for one dataset."""

    if args.agentic_eval_concurrency is not None:
        return args.agentic_eval_concurrency
    train_session_capacity = args.agentic_concurrency * args.n_samples_per_prompt
    return (train_session_capacity + eval_group_size - 1) // eval_group_size


class RuntimeDomain:
    """Sole adapter between pipeline group refs and their Session owner."""

    def __init__(
        self,
        args: Namespace,
        rollout_mode: Literal["train", "eval"],
        shards: List[_SessionShardBinding],
        concurrency: int,
        admission_coordinator: Any | None = None,
    ) -> None:
        self.args = args
        self.rollout_mode = rollout_mode
        self._shards = shards
        self.concurrency = concurrency
        self._admission_coordinator = admission_coordinator
        self._resident_group_permits = asyncio.Semaphore(concurrency)
        self._progress_callback: Callable[[], None] = lambda: None
        self._state_changed = asyncio.Event()

    def set_progress_callback(self, callback: Callable[[], None]) -> None:
        """Connect actor-local changes to the resident control loop."""

        self._progress_callback = callback

    @classmethod
    async def connect(
        cls,
        args: Namespace,
        rollout_mode: Literal["train", "eval"],
        concurrency: int,
    ) -> "RuntimeDomain":
        """Borrow Shard handles from the named Ray Serve deployment.

        Frequent control and terminal-data RPCs go directly to their Shard
        owners.
        """

        from ray import serve

        service = serve.get_app_handle(AGENTIC_CHAT_API_SERVICE_NAME)
        shard_entries, admission_coordinator = await service.runtime_resources.remote()
        return cls(
            args,
            rollout_mode,
            [_SessionShardBinding(actor_name=actor_name, handle=handle) for actor_name, handle in shard_entries],
            concurrency,
            admission_coordinator,
        )

    @property
    def resident_group_count(self) -> int:
        """Derive Runtime residency from Shard group capabilities."""

        return sum(len(shard.group_streams) for shard in self._shards)

    @property
    def interrupted_group_ids(self) -> Tuple[str, ...]:
        """Derive close credit from current handles and Shard tokens."""

        return tuple(
            stream.group_id for shard in self._shards for stream in shard.group_streams.values() if stream.interrupted
        )

    async def prepare_group(
        self,
        group_input: GroupInput,
        launch_decision: Optional[asyncio.Future[bool]] = None,
    ) -> Optional[RuntimeGroupStream]:
        """Start one group on a Shard and wait for its first-IR barrier.

        Session launch and progress observation share one Shard event stream. A
        controlled drop before the barrier resolves this operation without a
        Runtime capability.
        """

        await self._resident_group_permits.acquire()
        try:
            self._start_shard_observers()
            group_id = group_input.group_id
            shard = self._shards[zlib.crc32(group_id.encode("utf-8")) % len(self._shards)]
            session_ids = tuple(f"{shard.actor_name}.session-{uuid.uuid4().hex}" for _ in group_input.samples)
            loop = asyncio.get_running_loop()
            session_specs = await loop.run_in_executor(
                None,
                partial(
                    _build_session_specs,
                    group_input.samples,
                    session_ids=session_ids,
                    include_input_payload=self.args.rollout_global_dataset,
                ),
            )
            if launch_decision is not None:
                if not launch_decision.done():
                    launch_decision.set_result(True)
                if not launch_decision.result():
                    self._resident_group_permits.release()
                    return None
            for sample_position, session_id in enumerate(session_ids):
                group_input.samples[sample_position].session_id = session_id
            stream = RuntimeGroupStream(
                group_id=group_id,
                shard=shard,
                first_request_barrier=loop.create_future(),
                session_results=tuple(
                    _SessionResultRef(session_spec.session_id, loop.create_future()) for session_spec in session_specs
                ),
            )
            # Register before the first await so concurrent Prepare calls observe
            # one Shard binding for this stable Group token.
            if group_id in shard.group_streams:
                raise RuntimeGroupError(f"group token is already resident: {group_id}")
        except BaseException:
            self._resident_group_permits.release()
            raise
        shard.group_streams[group_id] = stream
        try:
            registration = asyncio.ensure_future(
                shard.handle.start_group.remote(self.rollout_mode, group_id, session_specs)
            )
        except BaseException:
            await finish_before_cancellation(
                self._release_shard_group(stream),
                f"start-group-submit-cleanup:{group_id}",
            )
            raise
        try:
            await finish_before_cancellation(
                registration,
                f"start-group:{group_id}",
            )
        except BaseException:
            if registration.cancelled() or registration.exception() is not None:
                await self._release_shard_group(stream)
            else:
                await self._drop_prepared_group(stream, f"prepare-group-cleanup:{group_id}")
            raise

        # Registration copied every launch field into the Shard. Drop the
        # driver-side source and transport payloads before the first-IR wait.
        del group_input, session_specs, registration, session_ids
        try:
            ready_for_lease = await asyncio.shield(stream.first_request_barrier)
            if not ready_for_lease:
                logger.warning("Dropping group %s because an agent exited before every first IR existed", group_id)
                logger.info(format_agentic_status("Dropped 1 group"))
                await self._drop_prepared_group(stream, f"prepare-group-drop:{group_id}")
                return None
        except BaseException:
            await self._drop_prepared_group(stream, f"prepare-group-cleanup:{group_id}")
            raise
        return stream

    async def lease_group(self, stream: RuntimeGroupStream, rollout_id: int) -> None:
        """Move one prepared group into Runtime and release its IR gate."""

        try:
            await stream.shard.handle.lease_group.remote(stream.group_id, rollout_id)
        except BaseException:
            await finish_before_cancellation(
                self.drop_group(stream),
                f"lease-group-cleanup:{stream.group_id}",
            )
            raise

    async def resume_generation(self, rollout_id: int) -> None:
        """Release the partial-resume gate for the same requests."""

        self._start_shard_observers()
        revisions = await asyncio.gather(
            *(shard.handle.resume_generation.remote(rollout_id) for shard in self._shards)
        )
        await self._wait_for_shard_revisions(revisions)

    async def pause_generation(self) -> None:
        """Seal train IR admission, abort active attempts, and join
        protection."""

        await self._wait_for_no_protected_groups()
        revisions = await asyncio.gather(*(shard.handle.pause_generation.remote() for shard in self._shards))
        await self._wait_for_shard_revisions(revisions)
        await self._wait_for_no_protected_groups()

    def _start_shard_observers(self) -> None:
        """Start the sole progress observer for each Ray Shard."""

        for shard in self._shards:
            if shard.observer_task is None:
                shard.observer_task = asyncio.create_task(
                    self._watch_shard_progress(shard),
                    name="session-shard-observer",
                )
            elif shard.observer_task.done():
                # A failed observer is a failed Runtime transport. Re-raising
                # preserves one owner instead of silently creating a new stream.
                shard.observer_task.result()

    async def _watch_shard_progress(self, shard: _SessionShardBinding) -> None:
        """Bridge one Shard event stream into local group refs."""

        def fail_group_stream(stream: RuntimeGroupStream, error: BaseException) -> None:
            for local_ref in (
                stream.first_request_barrier,
                *(result.completion for result in stream.session_results),
            ):
                if not local_ref.done():
                    local_ref.set_exception(error)

        try:
            while True:
                state_change_ref = shard.handle.wait_for_state_change.remote(shard.observed_revision)
                shard.state_change_ref = state_change_ref
                try:
                    progress = await state_change_ref
                finally:
                    if shard.state_change_ref is state_change_ref:
                        shard.state_change_ref = None
                self._apply_shard_progress(shard, progress)
                self._progress_callback()
        except asyncio.CancelledError:
            pass
        except BaseException as error:
            for stream in tuple(shard.group_streams.values()):
                fail_group_stream(stream, error)
            shard.revision_changed.set()
            self._state_changed.set()
            self._progress_callback()
            raise

    def _apply_shard_progress(self, shard: _SessionShardBinding, progress: SessionShardProgress) -> None:
        """Apply control progress and dispatch terminal-data takes."""

        shard.observed_revision = progress.revision
        for group_progress in progress.groups:
            stream = shard.group_streams.get(group_progress.group_id)
            if stream is None:
                continue
            # RuntimeDomain.prepare_group() owns the first-IR readiness waiter.
            if group_progress.ready_for_lease and not stream.first_request_barrier.done():
                stream.first_request_barrier.set_result(True)
            # The step close predicate reads interrupted Group IDs as Fully-async credit.
            stream.interrupted = group_progress.interrupted
            stream.protected = group_progress.protected
            # Group execution and ordinary RM await these terminal Session refs.
            results = []
            takeable_session_ids = set(group_progress.takeable_session_ids)
            for result in stream.session_results:
                if (
                    result.session_id in takeable_session_ids
                    and not result.completion.done()
                    and result.take_task is None
                ):
                    results.append(result)
            if results:
                take_task = asyncio.create_task(
                    _take_session_results(stream, tuple(results)),
                    name=f"shard-take:{stream.group_id}",
                )
                for result in results:
                    result.take_task = take_task
        shard.revision_changed.set()
        self._state_changed.set()

    async def _wait_for_no_protected_groups(self) -> None:
        """Join every protected Session around a train gate transition."""

        self._start_shard_observers()
        while any(stream.protected for shard in self._shards for stream in shard.group_streams.values()):
            self._state_changed.clear()
            if not any(stream.protected for shard in self._shards for stream in shard.group_streams.values()):
                break
            for shard in self._shards:
                observer_task = cast("asyncio.Task[None]", shard.observer_task)
                if observer_task.done():
                    observer_task.result()
            await self._state_changed.wait()

    async def _wait_for_shard_revisions(self, target_revisions: List[int]) -> None:
        """Join control RPCs with their event-stream observations."""

        async def wait_for_revision(shard: _SessionShardBinding, target_revision: int) -> None:
            while shard.observed_revision < target_revision:
                shard.revision_changed.clear()
                if shard.observed_revision >= target_revision:
                    break
                observer_task = cast("asyncio.Task[None]", shard.observer_task)
                if observer_task.done():
                    observer_task.result()
                await shard.revision_changed.wait()

        await asyncio.gather(
            *(
                wait_for_revision(shard, target_revision)
                for shard, target_revision in zip(self._shards, target_revisions)
            )
        )

    async def _stop_shard_observers(self) -> None:
        """Stop Shard observers during Runtime shutdown."""

        tasks = tuple(shard.observer_task for shard in self._shards if shard.observer_task is not None)
        remote_waiters = tuple(
            (shard, shard.state_change_ref)
            for shard in self._shards
            if isinstance(shard.state_change_ref, ray.ObjectRef)
        )
        for task in tasks:
            task.cancel()
        wake_outcomes = await asyncio.gather(
            *(shard.handle.wake_state_change_waiters.remote() for shard, _ref in remote_waiters),
            return_exceptions=True,
        )
        await asyncio.gather(*(ref for _shard, ref in remote_waiters), return_exceptions=True)
        task_outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        for shard in self._shards:
            shard.observer_task = None
            shard.state_change_ref = None
        for outcome in (*wake_outcomes, *task_outcomes):
            if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
                raise outcome

    async def collect_group(self, stream: RuntimeGroupStream) -> Optional[GroupExport]:
        """Gather original Session exports into one complete Group export.

        A controlled drop can complete the group early.
        """

        try:
            session_export_refs = tuple(result.completion for result in stream.session_results)
            pending = set(session_export_refs)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                # Read the whole completion batch so a controlled drop cannot hide a Runtime failure.
                outcomes = [session_export_ref.result() for session_export_ref in done]
                if any(outcome is None for outcome in outcomes):
                    logger.info(format_agentic_status("Dropped 1 group"))
                    return None

            await stream.shard.handle.release_group.remote(stream.group_id)
            group = GroupExport(
                group_id=stream.group_id,
                sessions=tuple(
                    cast(SessionExport, session_export_ref.result()) for session_export_ref in session_export_refs
                ),
            )
            await self._release_shard_group(stream)
            return group
        except BaseException:
            await finish_before_cancellation(
                self.drop_group(stream),
                f"wait-group-cleanup:{stream.group_id}",
            )
            raise

    async def drop_group(self, stream: RuntimeGroupStream) -> None:
        """Release Prepare, Runtime, and runner state with one group RPC."""

        if stream.shard.group_streams.get(stream.group_id) is not stream:
            return
        await stream.shard.handle.drop_group.remote(stream.group_id)
        if not stream.first_request_barrier.done():
            stream.first_request_barrier.set_result(False)
        for result in stream.session_results:
            if not result.completion.done():
                result.completion.set_result(None)
        await self._release_shard_group(stream)

    async def trim_memory(self) -> None:
        """Run allocator maintenance on every borrowed SessionShard."""

        await asyncio.gather(
            *(shard.handle.trim_memory.remote() for shard in self._shards),
            return_exceptions=True,
        )

    async def debug_state(self) -> dict[str, Any]:
        """Project Shard refs into one debug dictionary."""

        shard_snapshots = tuple(
            await asyncio.gather(*(shard.handle.debug_state.remote(sample_limit=8) for shard in self._shards))
        )
        return {
            "shards": shard_snapshots,
            "group_count": sum(snapshot["group_count"] for snapshot in shard_snapshots),
            "session_count": sum(snapshot["session_count"] for snapshot in shard_snapshots),
            "retained_request_count": sum(snapshot["retained_request_count"] for snapshot in shard_snapshots),
            "backend_request_count": sum(snapshot["backend_request_count"] for snapshot in shard_snapshots),
            "live_process_count": sum(snapshot["live_process_count"] for snapshot in shard_snapshots),
        }

    async def collect_agentic_kv_metrics(self) -> dict[str, float]:
        if not getattr(self.args, "agentic_program_admission", False) and not getattr(
            self.args, "agentic_session_lifecycle", False
        ):
            return {}
        shard_metrics = await asyncio.gather(
            *(shard.handle.agentic_kv_metrics.remote(reset=True) for shard in self._shards),
            return_exceptions=True,
        )
        metrics: dict[str, float] = {}
        for snapshot in shard_metrics:
            if isinstance(snapshot, BaseException):
                continue
            for key, value in snapshot.items():
                if key == "session/lifecycle_enabled":
                    metrics[key] = max(metrics.get(key, 0.0), float(value))
                else:
                    metrics[key] = metrics.get(key, 0.0) + float(value)
        if self._admission_coordinator is not None:
            try:
                metrics.update(await self._admission_coordinator.metrics.remote(True))
            except Exception:
                pass
        return {f"agentic_kv/{key}": value for key, value in metrics.items()}

    async def shutdown(self) -> None:
        """Join every Ray owner and raise the first cleanup failure."""

        first_error: Optional[BaseException] = None
        try:
            await self._stop_shard_observers()
        except BaseException as error:
            first_error = error

        streams = tuple(stream for shard in self._shards for stream in shard.group_streams.values())
        drop_outcomes = await asyncio.gather(
            *(stream.shard.handle.drop_group.remote(stream.group_id) for stream in streams),
            return_exceptions=True,
        )
        for outcome in drop_outcomes:
            if first_error is None and isinstance(outcome, BaseException):
                first_error = outcome
        for stream in streams:
            await self._release_shard_group(stream)

        await self.trim_memory()

        # Shards belong to the Serve deployment lifecycle. Runtime releases its
        # borrowed handles after every owned Group has completed cleanup.
        self._shards.clear()
        if first_error is not None:
            raise first_error

    async def _release_shard_group(self, stream: RuntimeGroupStream) -> None:
        """Release driver refs after Shard export/drop commits."""

        take_tasks = {result.take_task for result in stream.session_results if result.take_task is not None}
        for task in take_tasks:
            if not task.done():
                task.cancel()
        if take_tasks:
            await asyncio.gather(*take_tasks, return_exceptions=True)
        local_refs = (
            stream.first_request_barrier,
            *(result.completion for result in stream.session_results),
        )
        for local_ref in local_refs:
            if not local_ref.done():
                local_ref.cancel()
        await asyncio.gather(*local_refs, return_exceptions=True)
        # Remote cleanup can commit before cancellation reaches this driver ref.
        group_id = stream.group_id
        if stream.shard.group_streams.get(group_id) is stream:
            stream.shard.group_streams.pop(group_id)
            self._resident_group_permits.release()

    async def _drop_prepared_group(self, stream: RuntimeGroupStream, task_name: str) -> None:
        """Drop a registered Prepare group and release its local stream."""

        if stream.shard.group_streams.get(stream.group_id) is not stream:
            return
        try:
            await finish_before_cancellation(
                stream.shard.handle.drop_group.remote(stream.group_id),
                task_name,
            )
        finally:
            await self._release_shard_group(stream)


async def _take_session_results(
    stream: RuntimeGroupStream,
    results: Tuple[_SessionResultRef, ...],
) -> None:
    """Take one Group's announced results into stable local refs."""

    session_ids = tuple(result.session_id for result in results)
    try:
        session_results = await stream.shard.handle.take_session_results.remote(stream.group_id, session_ids)
        for result, session_result in zip(results, session_results, strict=True):
            _resolve_session_export(stream, result, session_result)
    except Exception as error:
        for result in results:
            completion = result.completion
            if not completion.done():
                completion.set_exception(error)
        if not stream.first_request_barrier.done():
            stream.first_request_barrier.set_exception(error)


def _resolve_session_export(
    stream: RuntimeGroupStream,
    result: _SessionResultRef,
    session_result: SessionExportTransport | RuntimeGroupError | None,
) -> None:
    """Resolve one terminal result into its driver-owned ref."""

    completion = result.completion
    if completion.done():
        return
    if isinstance(session_result, RuntimeGroupError):
        completion.set_exception(session_result)
        if not stream.first_request_barrier.done():
            stream.first_request_barrier.set_exception(session_result)
        return
    if session_result is None:
        completion.set_result(None)
        if not stream.first_request_barrier.done():
            stream.first_request_barrier.set_result(False)
        return
    exports = []
    for export_payload in session_result.exports:
        sample = TrainingFieldArtifact(sample_payload=export_payload["sample_payload"]).to_sample()
        mark_sample_agentic_event(sample, "sample_export_collected_at")
        exports.append(
            SampleExport(
                name=export_payload["name"],
                sample=sample,
            )
        )
    completion.set_result(SessionExport(exports=tuple(exports)))


__all__ = ["RuntimeDomain", "RuntimeGroupError", "RuntimeGroupStream"]
