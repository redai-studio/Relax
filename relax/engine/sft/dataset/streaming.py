# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Streaming SFT dataset over the shared Relax prompt-data format."""

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Iterable, Optional

import torch

from relax.engine.sft.dataset.chat_template import render_to_text, render_with_loss_mask
from relax.engine.sft.dataset.multimodal import (
    has_multimodal_content,
    preprocess_multimodal,
    preprocess_multimodal_async,
)
from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample
from relax.utils.data.data_utils import (
    build_messages,
    collect_message_multimodal_data,
    count_message_multimodal_items,
    resolve_path_plan,
)
from relax.utils.data.streaming_dataset import CompositeStreamingReader, IndexManager, PrefetchBuffer, StreamingReader
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_LEARN_ROLES = {"assistant", "function_call"}


class SFTMultimodalContractError(ValueError):
    """A prompt's multimodal placeholders do not match its media fields."""


@dataclass
class ProcessedSample:
    """Output of ``SFTStreamingDataset._process_one``."""

    tokens: torch.Tensor
    loss_mask: torch.Tensor
    total_length: int
    multimodal_train_inputs: dict[str, Any] | None
    source_idx: int
    classification_label: torch.Tensor | None = None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _validate_multimodal_contract(
    row: dict[str, Any],
    prompt: str | list[Any],
    *,
    row_index: int,
    multimodal_keys: dict[str, str] | None,
    source_name: str,
) -> None:
    message_counts = count_message_multimodal_items(prompt)
    inline_media = collect_message_multimodal_data(prompt)
    mismatches = {}
    for media_name, message_count in message_counts.items():
        inline_count = len(inline_media[media_name])
        top_level_required_count = message_count - inline_count
        data_key = (multimodal_keys or {}).get(media_name)
        top_level_count = len(_as_list(row.get(data_key))) if data_key else 0
        if message_count > 0 and data_key is None:
            mismatches[media_name] = {
                "message_count": message_count,
                "inline_count": inline_count,
                "data_key": None,
                "reason": f"configure {media_name!r} in --multimodal-keys",
            }
            continue
        if top_level_required_count == top_level_count:
            continue
        if message_count == 0 and top_level_count == 0:
            continue
        mismatches[media_name] = {
            "message_count": message_count,
            "inline_count": inline_count,
            "top_level_required_count": top_level_required_count,
            "data_key": data_key,
            "top_level_count": top_level_count,
        }
    if mismatches:
        sample_id = row.get("_sample_id")
        raise SFTMultimodalContractError(
            f"SFT multimodal contract mismatch in {source_name!r} at sample idx={row_index}, "
            f"sample_id={sample_id!r}: {mismatches}. Structured image/image_url/video/audio items with a valid "
            "inline payload are self-contained; literal or payload-less placeholders require exactly one media "
            "entry in the field configured by --multimodal-keys."
        )


def _validate_resolved_multimodal_contract(
    messages: list[Any],
    *,
    row: dict[str, Any],
    row_index: int,
    source_name: str,
) -> dict[str, list[Any]]:
    message_counts = count_message_multimodal_items(messages)
    media = collect_message_multimodal_data(messages)
    mismatches = {
        media_name: {"message_count": message_count, "resolved_source_count": len(media[media_name])}
        for media_name, message_count in message_counts.items()
        if message_count != len(media[media_name])
    }
    if mismatches:
        sample_id = row.get("_sample_id")
        raise SFTMultimodalContractError(
            f"SFT multimodal source resolution failed in {source_name!r} at sample idx={row_index}, "
            f"sample_id={sample_id!r}: {mismatches}. Each rendered media marker must resolve to exactly one "
            "inline or top-level media source."
        )
    return media


def _normalize_tools(value: Any) -> list[dict] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TypeError(f"tools must be a list, got {type(value)}")
    return value


def _apply_conversation_key_map(messages: list[Any], key_map: dict[str, str]) -> list[dict]:
    """Rewrite non-OpenAI message dicts into the ``{role, content}`` shape.

    A single flat ``key_map`` covers two distinct renames:
      * **Field-name rename** — applied to every key of every message dict
        (e.g. ``"from" -> "role"``, ``"value" -> "content"``).
      * **Role-value rename** — applied **only** to the value of the field
        that ends up named ``"role"`` after the field rename (e.g.
        ``"human" -> "user"``, ``"gpt" -> "assistant"``). Limiting the value
        rename to the ``role`` field keeps message bodies that happen to
        contain the same words (``"... from a human ..."``) untouched.

    Non-dict elements pass through unchanged so a caller that already mixed in
    canonical messages stays valid.
    """
    out: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        new: dict = {}
        for k, v in m.items():
            new_k = key_map.get(k, k) if isinstance(k, str) else k
            if new_k == "role" and isinstance(v, str):
                new_v = key_map.get(v, v)
            else:
                new_v = v
            new[new_k] = new_v
        out.append(new)
    return out


def _canonicalize_messages(raw_messages: list[dict], *, require_response: bool) -> list[CanonicalMessage]:
    messages: list[CanonicalMessage] = []
    has_learn = False
    for raw in raw_messages:
        role = raw.get("role")
        if role is None:
            raise ValueError(f"SFT message missing role: {raw}")
        if "content" not in raw:
            raise ValueError(f"SFT message missing content: {raw}")
        learn = bool(raw.get("learn", role in _LEARN_ROLES))
        has_learn = has_learn or learn
        messages.append(
            CanonicalMessage(
                role=role,
                content=raw["content"],
                learn=learn,
                tool_calls=raw.get("tool_calls"),
            )
        )
    if require_response and not has_learn:
        raise ValueError("SFT training row has no assistant/function_call response tokens.")
    return messages


def _build_canonical_sample_from_row(
    row: dict[str, Any],
    *,
    row_index: int,
    prompt_key: str,
    label_key: str | None,
    multimodal_keys: dict[str, str] | None,
    metadata_key: str,
    tool_key: str | None,
    system_prompt: str | None,
    require_response: bool,
    source_name: str,
    task_type: str = "causal_lm",
    conversation_key_map: dict[str, str] | None = None,
) -> CanonicalSample:
    prompt = row.get(prompt_key)
    if prompt is None:
        raise ValueError(f"SFT row missing prompt key {prompt_key!r}: available keys={list(row.keys())}")

    if label_key is None:
        if not isinstance(prompt, list):
            raise TypeError(
                f"--label-key is not set, so SFT expects --input-key {prompt_key!r} "
                f"to contain an OpenAI messages list; got {type(prompt)}"
            )
        if conversation_key_map:
            # Rewrite e.g. sharegpt {"from","value","human","gpt"} -> OpenAI
            # {"role","content","user","assistant"} before downstream code
            # (build_messages, _canonicalize_messages) hits the hard-coded
            # OpenAI-format requirements. Shallow-copy the row so we don't
            # mutate the reader's cache.
            prompt = _apply_conversation_key_map(prompt, conversation_key_map)
            row = {**row, prompt_key: prompt}
        _validate_multimodal_contract(
            row,
            prompt,
            row_index=row_index,
            multimodal_keys=multimodal_keys,
            source_name=source_name,
        )
        raw_messages = build_messages(row, prompt_key, system_prompt, True, multimodal_keys)
    elif task_type == "seq_cls":
        if not isinstance(prompt, (str, list)):
            raise TypeError(
                f"--task-type seq_cls expects --input-key {prompt_key!r} to contain a prompt string "
                f"or OpenAI messages list; got {type(prompt)}"
            )
        if row.get(label_key) is None:
            raise ValueError(
                f"Sequence classification row idx={row_index} in {source_name!r} is missing label key "
                f"{label_key!r}: available keys={list(row.keys())}"
            )
        if isinstance(prompt, list) and conversation_key_map:
            prompt = _apply_conversation_key_map(prompt, conversation_key_map)
            row = {**row, prompt_key: prompt}
        _validate_multimodal_contract(
            row,
            prompt,
            row_index=row_index,
            multimodal_keys=multimodal_keys,
            source_name=source_name,
        )
        raw_messages = build_messages(row, prompt_key, system_prompt, True, multimodal_keys)
    else:
        if not isinstance(prompt, str):
            raise TypeError(
                f"--label-key is set, so SFT expects --input-key {prompt_key!r} "
                f"to contain a prompt string; got {type(prompt)}"
            )
        if row.get(label_key) is None:
            raise ValueError(f"SFT row missing label key {label_key!r}: available keys={list(row.keys())}")
        _validate_multimodal_contract(
            row,
            prompt,
            row_index=row_index,
            multimodal_keys=multimodal_keys,
            source_name=source_name,
        )
        raw_messages = build_messages(row, prompt_key, system_prompt, True, multimodal_keys)
        raw_messages = list(raw_messages)
        raw_messages.append({"role": "assistant", "content": row[label_key], "learn": True})

    if not isinstance(raw_messages, list):
        raise TypeError(f"SFT prompt key {prompt_key!r} did not normalize to messages.")

    metadata = row.get(metadata_key) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    metadata.setdefault("source_dataset", source_name)
    metadata["row_index"] = row_index

    media = _validate_resolved_multimodal_contract(
        raw_messages,
        row=row,
        row_index=row_index,
        source_name=source_name,
    )

    return CanonicalSample(
        messages=_canonicalize_messages(raw_messages, require_response=require_response),
        metadata=metadata,
        tools=_normalize_tools(row.get(tool_key)) if tool_key else None,
        images=media.get("image") or None,
        videos=media.get("video") or None,
        audios=media.get("audio") or None,
    )


def _build_reader(path: str | list[str] | tuple[str, ...]):
    paths, row_slice = resolve_path_plan(path)
    if len(paths) == 1 and row_slice is None:
        return StreamingReader(paths[0])
    return CompositeStreamingReader(paths, row_slice)


class SFTStreamingDataset:
    """Lazy SFT dataset with epoch-aware shuffle and optional prefetch."""

    def __init__(
        self,
        path: str | list[str] | tuple[str, ...],
        *,
        tokenizer=None,
        processor_pool=None,
        capacity: int | None = None,
        prompt_key: str = "input",
        label_key: str | None = None,
        multimodal_keys: dict[str, str] | None = None,
        conversation_key_map: dict[str, str] | None = None,
        metadata_key: str = "metadata",
        tool_key: str | None = None,
        system_prompt: str | None = None,
        source_name: str = "prompt_data",
        require_response: bool = True,
        seed: int = 42,
        prefetch_max_cached: int = 256,
        prefetch_chunk_size: int = 32,
        prefetch_num_workers: int = 4,
        pad_token_ids: Iterable[int] | None = None,
        oversize_strategy: str = "keep",
        oversize_custom_fn: Callable[..., Optional[tuple[torch.Tensor, torch.Tensor]]] | None = None,
        invalid_multimodal_strategy: str = "error",
        apply_chat_template_kwargs: dict | None = None,
        task_type: str = "causal_lm",
        num_labels: int | None = None,
        problem_type: str = "single_label_classification",
        classification_sentinel_token_id: int | None = None,
    ) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.processor_pool = processor_pool
        self.capacity = capacity
        self.prompt_key = prompt_key
        self.label_key = label_key
        self.multimodal_keys = multimodal_keys
        self.conversation_key_map = conversation_key_map
        self.metadata_key = metadata_key
        self.tool_key = tool_key
        self.system_prompt = system_prompt
        self.source_name = source_name
        self.require_response = require_response
        self._pad_token_ids: frozenset[int] = frozenset(pad_token_ids or ())
        # Forwarded into tokenizer.apply_chat_template() via render_with_loss_mask
        # / render_to_text. Lets callers (and per-sample metadata) override the
        # tokenizer's bound chat_template — needed for models whose native
        # template is inference-only and drops training-critical content
        # (e.g. DeepSeek-R1 distill templates strip <think>...</think>).
        self.apply_chat_template_kwargs = apply_chat_template_kwargs
        self.task_type = task_type
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.classification_sentinel_token_id = classification_sentinel_token_id
        if self.task_type == "seq_cls":
            valid_problem_types = {"single_label_classification", "multi_label_classification"}
            if self.problem_type not in valid_problem_types:
                raise ValueError(
                    f"SFTStreamingDataset seq_cls problem_type must be one of {sorted(valid_problem_types)}, "
                    f"got {self.problem_type!r}"
                )
            if self.label_key is None:
                raise ValueError("SFTStreamingDataset seq_cls mode requires label_key")
            if self.num_labels is None or self.num_labels < 2:
                raise ValueError("SFTStreamingDataset seq_cls mode requires num_labels >= 2")
            if self.classification_sentinel_token_id is None:
                raise ValueError("SFTStreamingDataset seq_cls mode requires a sentinel token id")
            if self.capacity is not None and self.capacity < 2:
                raise ValueError("SFTStreamingDataset seq_cls capacity must be >= 2")
        valid_strategies = {"skip", "keep", "truncate_left", "truncate_right", "custom"}
        if oversize_strategy not in valid_strategies:
            raise ValueError(f"oversize_strategy must be one of {sorted(valid_strategies)}, got {oversize_strategy!r}")
        if oversize_strategy == "custom" and oversize_custom_fn is None:
            raise ValueError("oversize_strategy='custom' requires oversize_custom_fn to be provided")
        self._oversize_strategy = oversize_strategy
        self._oversize_custom_fn = oversize_custom_fn

        valid_invalid_multimodal_strategies = {"error", "skip"}
        if invalid_multimodal_strategy not in valid_invalid_multimodal_strategies:
            raise ValueError(
                "invalid_multimodal_strategy must be one of "
                f"{sorted(valid_invalid_multimodal_strategies)}, got {invalid_multimodal_strategy!r}"
            )
        self._invalid_multimodal_strategy = invalid_multimodal_strategy

        # Fail-fast latch. Background prefetch threads can't propagate
        # exceptions to the asyncio producer, so the first error is recorded
        # here and re-raised by the next ``get_batch*`` call.
        self._first_error: BaseException | None = None
        self._error_lock = threading.Lock()

        self.reader = _build_reader(path)
        self.index_manager = IndexManager(len(self.reader), seed=seed)

        self._prefetch: PrefetchBuffer | None = None
        if prefetch_max_cached > 0:
            self._prefetch = PrefetchBuffer(
                process_fn=self._process_one_safe,
                chunk_size=prefetch_chunk_size,
                max_cached=prefetch_max_cached,
                num_workers=prefetch_num_workers,
            )
            logger.info(
                f"SFTStreamingDataset: prefetch enabled "
                f"(max_cached={prefetch_max_cached}, chunk_size={prefetch_chunk_size}, "
                f"num_workers={prefetch_num_workers})"
            )

        logger.info(
            f"SFTStreamingDataset path={path} total_size={len(self.reader)} "
            f"capacity={self.capacity} prompt_key={self.prompt_key!r} label_key={self.label_key!r} "
            f"task_type={self.task_type!r}"
        )

    def __len__(self) -> int:
        return len(self.reader)

    @property
    def prefetch_enabled(self) -> bool:
        return self._prefetch is not None

    def restrict_training_size(self, size: int) -> None:
        """Restrict shuffled training indices to the leading ``size`` rows."""
        if not 0 < size <= len(self.reader):
            raise ValueError(f"training size must be in [1, {len(self.reader)}], got {size}")
        self.restrict_training_indices(range(size))

    def restrict_training_indices(self, indices: Iterable[int]) -> None:
        """Restrict epoch shuffling to the provided physical row IDs."""
        if self.index_manager.current_epoch >= 0 or self.index_manager.position != 0:
            raise RuntimeError("training indices must be restricted before the dataset is shuffled or consumed")
        index_pool = tuple(indices)
        if not index_pool:
            raise ValueError("training indices must not be empty")
        if len(set(index_pool)) != len(index_pool):
            raise ValueError("training indices must be unique")
        if any(not isinstance(index, int) or index < 0 or index >= len(self.reader) for index in index_pool):
            raise ValueError(f"training indices must be integers in [0, {len(self.reader)})")
        self.index_manager = IndexManager(len(index_pool), seed=self.index_manager.seed, index_pool=index_pool)

    def shuffle(self, epoch_id: int, position: int = 0) -> None:
        self.index_manager.shuffle(epoch_id)
        if position:
            self.index_manager.position = min(position, self.index_manager.total_size)
        if self._prefetch is not None and self.index_manager.indices is not None:
            remaining = self.index_manager.indices[self.index_manager.position :]
            self._prefetch.set_index_order(list(remaining))
            logger.info(
                f"SFTStreamingDataset: prefetch primed for epoch={epoch_id} "
                f"position={self.index_manager.position} remaining={len(remaining)}"
            )

    def get_batch(self, n: int) -> tuple[list[ProcessedSample], bool]:
        if self._prefetch is not None:
            return self._get_batch_prefetch(n)
        return self._get_batch_inline(n)

    async def get_batch_async(self, n: int) -> tuple[list[ProcessedSample], bool]:
        if self._prefetch is not None:
            return await self._get_batch_prefetch_async(n)
        return await self._get_batch_async_gather(n)

    def get_batch_in_order(self, start: int, n: int) -> list[ProcessedSample]:
        self._raise_if_failed()
        out: list[ProcessedSample] = []
        for offset in range(n):
            idx = start + offset
            if idx >= len(self.reader):
                break
            sample = self._process_one(idx)
            if sample is not None:
                out.append(sample)
        return out

    def get_batch_by_indices(self, indices: Iterable[int]) -> list[ProcessedSample]:
        """Render physical row IDs in the provided deterministic order."""
        self._raise_if_failed()
        out: list[ProcessedSample] = []
        for idx in indices:
            if idx < 0 or idx >= len(self.reader):
                raise IndexError(f"index {idx} out of range [0, {len(self.reader)})")
            sample = self._process_one(idx)
            if sample is not None:
                out.append(sample)
        return out

    def _record_first_error(self, exc: BaseException) -> None:
        with self._error_lock:
            if self._first_error is None:
                self._first_error = exc

    def _raise_if_failed(self) -> None:
        with self._error_lock:
            err = self._first_error
        if err is not None:
            raise err

    def stop(self) -> None:
        if self._prefetch is not None:
            self._prefetch.stop()

    def get_canonical_sample(self, idx: int, *, row: dict[str, Any] | None = None) -> CanonicalSample:
        if row is None:
            row = self.reader[idx]
        return _build_canonical_sample_from_row(
            row,
            row_index=idx,
            prompt_key=self.prompt_key,
            label_key=self.label_key,
            multimodal_keys=self.multimodal_keys,
            conversation_key_map=self.conversation_key_map,
            metadata_key=self.metadata_key,
            tool_key=self.tool_key,
            system_prompt=self.system_prompt,
            require_response=self.require_response,
            source_name=self.source_name,
            task_type=self.task_type,
        )

    def get_classification_label(self, idx: int, *, row: dict[str, Any] | None = None) -> torch.Tensor | None:
        if self.task_type != "seq_cls":
            return None
        assert self.label_key is not None and self.num_labels is not None
        if row is None:
            row = self.reader[idx]
        if self.label_key not in row or row[self.label_key] is None:
            raise ValueError(
                f"Sequence classification row idx={idx} in {self.source_name!r} is missing label key "
                f"{self.label_key!r}"
            )
        value = row[self.label_key]
        if self.problem_type == "single_label_classification":
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(
                    f"Sequence classification row idx={idx} in {self.source_name!r} requires an integer label; "
                    f"got {type(value).__name__}"
                )
            label = int(value)
            if not 0 <= label < self.num_labels:
                raise ValueError(
                    f"Sequence classification row idx={idx} in {self.source_name!r} has label {label}; "
                    f"expected 0 <= label < {self.num_labels}"
                )
            return torch.tensor(label, dtype=torch.long)

        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"Multi-label classification row idx={idx} in {self.source_name!r} requires a class-index list; "
                f"got {type(value).__name__}"
            )
        labels: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, Integral):
                raise TypeError(
                    f"Multi-label classification row idx={idx} in {self.source_name!r} contains a non-integer "
                    f"class index {item!r}"
                )
            label = int(item)
            if not 0 <= label < self.num_labels:
                raise ValueError(
                    f"Multi-label classification row idx={idx} in {self.source_name!r} has class index {label}; "
                    f"expected 0 <= index < {self.num_labels}"
                )
            labels.append(label)
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"Multi-label classification row idx={idx} in {self.source_name!r} contains duplicate indices: "
                f"{labels}"
            )
        multi_hot = torch.zeros(self.num_labels, dtype=torch.float32)
        if labels:
            multi_hot[labels] = 1.0
        return multi_hot

    def _get_batch_prefetch(self, n: int) -> tuple[list[ProcessedSample], bool]:
        self._raise_if_failed()
        samples: list[ProcessedSample] = []
        crossed_epoch = False
        max_attempts = max(n * 10, 32)
        attempts = 0
        while len(samples) < n and attempts < max_attempts:
            indices, epoch_crossed = self.index_manager.get_next_indices(1)
            attempts += 1
            idx = indices[0]
            if epoch_crossed:
                crossed_epoch = True
                remaining = [idx, *self.index_manager.indices[self.index_manager.position :]]
                assert self._prefetch is not None
                self._prefetch.set_index_order(remaining)
                logger.info(
                    f"SFTStreamingDataset: epoch boundary crossed, prefetch re-primed "
                    f"(epoch={self.index_manager.current_epoch}, remaining={len(remaining)})"
                )
            assert self._prefetch is not None
            sample = self._prefetch.get(idx)
            if sample is None:
                # PrefetchBuffer swallows worker-thread exceptions; surface
                # any latched failure now instead of silently returning a
                # short batch (which deadlocks the TQ consumer).
                self._raise_if_failed()
            else:
                samples.append(sample)
        self._raise_if_failed()
        if len(samples) < n:
            logger.warning(
                f"SFTStreamingDataset.get_batch: returned {len(samples)}/{n} samples after {attempts} attempts."
            )
        return samples, crossed_epoch

    async def _get_batch_prefetch_async(self, n: int) -> tuple[list[ProcessedSample], bool]:
        self._raise_if_failed()
        samples: list[ProcessedSample] = []
        crossed_epoch = False
        max_attempts = max(n * 10, 32)
        attempts = 0
        prefetch_wait_timeout_s = 300.0
        prefetch_wait_poll_s = 0.02
        assert self._prefetch is not None
        while len(samples) < n and attempts < max_attempts:
            indices, epoch_crossed = self.index_manager.get_next_indices(1)
            attempts += 1
            idx = indices[0]
            if epoch_crossed:
                crossed_epoch = True
                remaining = [idx, *self.index_manager.indices[self.index_manager.position :]]
                self._prefetch.set_index_order(remaining)
                logger.info(
                    f"SFTStreamingDataset: epoch boundary crossed, prefetch re-primed "
                    f"(epoch={self.index_manager.current_epoch}, remaining={len(remaining)})"
                )
            found, sample = self._prefetch.get_cached(idx)
            wait_started = time.monotonic()
            while not found:
                self._raise_if_failed()
                if not self._prefetch.is_alive:
                    raise RuntimeError(
                        f"SFTStreamingDataset: prefetch worker exited before sample idx={idx} was cached "
                        f"(cache_size={self._prefetch.cache_size})"
                    )
                if time.monotonic() - wait_started >= prefetch_wait_timeout_s:
                    raise TimeoutError(
                        f"SFTStreamingDataset: timed out waiting for prefetched sample idx={idx} "
                        f"after {prefetch_wait_timeout_s:.1f}s "
                        f"(cache_size={self._prefetch.cache_size}, prefetch_alive={self._prefetch.is_alive})"
                    )
                await asyncio.sleep(prefetch_wait_poll_s)
                found, sample = self._prefetch.get_cached(idx, record_miss=False)
            if sample is None:
                # Cached None means the background worker processed this index
                # and decided it should be skipped. Surface any latched error;
                # otherwise keep refilling the batch.
                self._raise_if_failed()
            else:
                samples.append(sample)
        self._raise_if_failed()
        if len(samples) < n:
            logger.warning(
                f"SFTStreamingDataset.get_batch_async (prefetch): returned {len(samples)}/{n} samples "
                f"after {attempts} attempts."
            )
        return samples, crossed_epoch

    def _get_batch_inline(self, n: int) -> tuple[list[ProcessedSample], bool]:
        self._raise_if_failed()
        samples: list[ProcessedSample] = []
        crossed_epoch = False
        max_attempts = max(n * 10, 32)
        attempts = 0
        while len(samples) < n and attempts < max_attempts:
            need = n - len(samples)
            indices, epoch_crossed = self.index_manager.get_next_indices(need)
            crossed_epoch = crossed_epoch or epoch_crossed
            for idx in indices:
                attempts += 1
                sample = self._process_one(idx)
                if sample is not None:
                    samples.append(sample)
        if len(samples) < n:
            logger.warning(
                f"SFTStreamingDataset.get_batch (inline): returned {len(samples)}/{n} samples "
                f"after {attempts} attempts."
            )
        return samples, crossed_epoch

    async def _get_batch_async_gather(self, n: int) -> tuple[list[ProcessedSample], bool]:
        self._raise_if_failed()
        rendered: list[_RenderedSample] = []
        crossed_epoch = False
        max_attempts = max(n * 10, 32)
        attempts = 0
        while len(rendered) < n and attempts < max_attempts:
            need = n - len(rendered)
            indices, ec = self.index_manager.get_next_indices(need)
            crossed_epoch = crossed_epoch or ec
            for idx in indices:
                attempts += 1
                pre = self._render_one(idx)
                if pre is not None:
                    rendered.append(pre)
        if len(rendered) < n:
            logger.warning(
                f"SFTStreamingDataset.get_batch_async: rendered {len(rendered)}/{n} after {attempts} attempts."
            )
        if not rendered:
            return [], crossed_epoch
        coros = [self._finalize_async(r) for r in rendered]
        finalized = await asyncio.gather(*coros)
        out = [r for r in finalized if r is not None]
        return out, crossed_epoch

    def _process_one_safe(self, idx: int) -> ProcessedSample | None:
        """Wrapper passed to ``PrefetchBuffer`` worker threads.

        The buffer's internal ``try/except`` would otherwise drop the exception
        on the floor and store ``None`` in its cache, leaving the producer to
        push short batches that deadlock the TQ consumer. We capture the first
        exception into a latch that the next ``get_batch*`` call re-raises from
        the asyncio context.
        """
        try:
            return self._process_one(idx)
        except Exception as exc:
            logger.exception(f"SFTStreamingDataset: failed to process sample idx={idx}")
            self._record_first_error(exc)
            return None

    def _process_one(self, idx: int) -> ProcessedSample | None:
        rendered = self._render_one(idx)
        if rendered is None:
            return None
        prompt_ids, mm_inputs = preprocess_multimodal(
            rendered.sample,
            processor_pool=self.processor_pool,
            rendered_text=rendered.rendered_text or "",
        )
        return self._build_processed(rendered, prompt_ids, mm_inputs)

    def _render_one(self, idx: int) -> "_RenderedSample | None":
        if self.tokenizer is None:
            raise RuntimeError("SFTStreamingDataset: tokenizer is required for _render_one")
        row = self.reader[idx]
        try:
            sample = self.get_canonical_sample(idx, row=row)
        except SFTMultimodalContractError as exc:
            if self._invalid_multimodal_strategy == "error":
                raise
            logger.warning(f"SFTStreamingDataset[invalid-multimodal=skip]: {exc} Skipping.")
            return None
        short_ids, short_mask = render_with_loss_mask(
            sample, tokenizer=self.tokenizer, apply_chat_template_kwargs=self.apply_chat_template_kwargs
        )
        n = int(short_ids.shape[0])
        effective_n = n + 1 if self.task_type == "seq_cls" else n
        if self.capacity is not None and effective_n > self.capacity and self._oversize_strategy == "skip":
            logger.warning(
                f"SFTStreamingDataset[oversize=skip]: sample idx={idx} length {effective_n} "
                f"exceeds per-GPU capacity {self.capacity}; skipping."
            )
            return None
        rendered_text = None
        if has_multimodal_content(sample):
            rendered_text = render_to_text(
                sample,
                tokenizer=self.tokenizer,
                apply_chat_template_kwargs=self.apply_chat_template_kwargs,
            )
        return _RenderedSample(
            idx=idx,
            sample=sample,
            short_ids=short_ids,
            short_mask=short_mask,
            rendered_text=rendered_text,
            total_length=effective_n,
            classification_label=self.get_classification_label(idx, row=row),
        )

    async def _finalize_async(self, rendered: "_RenderedSample") -> ProcessedSample | None:
        prompt_ids, mm_inputs = await preprocess_multimodal_async(
            rendered.sample,
            processor_pool=self.processor_pool,
            rendered_text=rendered.rendered_text or "",
        )
        return self._build_processed(rendered, prompt_ids, mm_inputs)

    def _build_processed(
        self,
        rendered: "_RenderedSample",
        prompt_ids: Any | None,
        mm_inputs: dict[str, Any] | None,
    ) -> ProcessedSample | None:
        if prompt_ids is None:
            tokens = rendered.short_ids
            loss_mask = rendered.short_mask
        else:
            tokens = _to_long_tensor(prompt_ids)
            if self.task_type == "seq_cls":
                loss_mask = torch.zeros_like(tokens)
            else:
                loss_mask = _expand_loss_mask_via_alignment(
                    short_ids=rendered.short_ids,
                    short_mask=rendered.short_mask,
                    expanded_ids=tokens,
                    pad_token_ids=self._pad_token_ids,
                )
        if self.task_type == "seq_cls":
            if tokens.numel() == 0:
                raise ValueError(
                    f"Sequence classification row idx={rendered.idx} in {self.source_name!r} produced no prompt tokens"
                )
            prompt_capacity = None if self.capacity is None else self.capacity - 1
            if prompt_capacity is not None and prompt_capacity < 1:
                raise ValueError(
                    f"Sequence classification capacity must leave room for one prompt token and one sentinel; "
                    f"got capacity={self.capacity}"
                )
            result = self._apply_oversize_strategy(
                tokens=tokens,
                loss_mask=torch.zeros_like(tokens),
                idx=rendered.idx,
                has_multimodal=mm_inputs is not None,
                capacity_override=prompt_capacity,
            )
            if result is None:
                return None
            tokens, _ = result
            sentinel = torch.tensor(
                [self.classification_sentinel_token_id],
                dtype=tokens.dtype,
                device=tokens.device,
            )
            tokens = torch.cat([tokens, sentinel])
            loss_mask = torch.ones(1, dtype=torch.long, device=tokens.device)
        else:
            result = self._apply_oversize_strategy(
                tokens=tokens,
                loss_mask=loss_mask,
                idx=rendered.idx,
                has_multimodal=mm_inputs is not None,
            )
            if result is None:
                return None
            tokens, loss_mask = result
        n = int(tokens.shape[0])
        return ProcessedSample(
            tokens=tokens,
            loss_mask=loss_mask,
            total_length=n,
            classification_label=rendered.classification_label,
            multimodal_train_inputs=mm_inputs,
            source_idx=rendered.idx,
        )

    def _apply_oversize_strategy(
        self,
        *,
        tokens: torch.Tensor,
        loss_mask: torch.Tensor,
        idx: int,
        has_multimodal: bool,
        capacity_override: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        n = int(tokens.shape[0])
        cap = self.capacity if capacity_override is None else capacity_override
        if cap is None or n <= cap:
            return tokens, loss_mask
        strategy = self._oversize_strategy
        if strategy == "skip":
            logger.warning(
                f"SFTStreamingDataset[oversize=skip]: sample idx={idx} expanded length {n} "
                f"exceeds per-GPU capacity {cap}; skipping."
            )
            return None
        if strategy == "keep":
            logger.warning(
                f"SFTStreamingDataset[oversize=keep]: sample idx={idx} expanded length {n} "
                f"exceeds per-GPU capacity {cap}; keeping unchanged (may OOM or be dropped "
                f"by the dynamic batcher downstream)."
            )
            return tokens, loss_mask
        if strategy in ("truncate_left", "truncate_right"):
            if has_multimodal:
                logger.warning(
                    f"SFTStreamingDataset[oversize={strategy}]: sample idx={idx} expanded length {n} "
                    f"exceeds per-GPU capacity {cap} and is multimodal; truncating tokens in-place "
                    f"WILL misalign multimodal_train_inputs."
                )
            if strategy == "truncate_left":
                tokens_t = tokens[-cap:].contiguous()
                mask_t = loss_mask[-cap:].contiguous()
            else:
                tokens_t = tokens[:cap].contiguous()
                mask_t = loss_mask[:cap].contiguous()
            logger.warning(
                f"SFTStreamingDataset[oversize={strategy}]: sample idx={idx} expanded length {n} "
                f"exceeds per-GPU capacity {cap}; truncated to {int(tokens_t.shape[0])} tokens."
            )
            return tokens_t, mask_t
        if strategy == "custom":
            assert self._oversize_custom_fn is not None
            result = self._oversize_custom_fn(tokens=tokens, loss_mask=loss_mask, capacity=cap, idx=idx)
            if result is None:
                logger.warning(
                    f"SFTStreamingDataset[oversize=custom]: sample idx={idx} expanded length {n} "
                    f"exceeds per-GPU capacity {cap}; custom function returned None; skipping."
                )
                return None
            tokens_t, mask_t = result
            logger.warning(
                f"SFTStreamingDataset[oversize=custom]: sample idx={idx} expanded length {n} "
                f"exceeds per-GPU capacity {cap}; custom function returned {int(tokens_t.shape[0])} tokens."
            )
            return tokens_t, mask_t
        raise ValueError(f"unknown oversize strategy: {strategy!r}")


@dataclass
class _RenderedSample:
    idx: int
    sample: CanonicalSample
    short_ids: torch.Tensor
    short_mask: torch.Tensor
    rendered_text: str | None
    total_length: int
    classification_label: torch.Tensor | None


def _to_long_tensor(ids: Any) -> torch.Tensor:
    if isinstance(ids, torch.Tensor):
        t = ids
    else:
        t = torch.as_tensor(ids)
    if t.dim() == 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.dim() != 1:
        raise ValueError(f"Expected 1D ids tensor, got shape {tuple(t.shape)}")
    return t.long()


def _expand_loss_mask_via_alignment(
    *,
    short_ids: torch.Tensor,
    short_mask: torch.Tensor,
    expanded_ids: torch.Tensor,
    pad_token_ids: frozenset[int],
) -> torch.Tensor:
    if short_ids.dim() != 1 or short_mask.dim() != 1 or expanded_ids.dim() != 1:
        raise ValueError("Inputs must be 1D tensors")
    if short_ids.shape[0] != short_mask.shape[0]:
        raise ValueError(f"short_ids/short_mask length mismatch: {short_ids.shape[0]} vs {short_mask.shape[0]}")

    s = short_ids.tolist()
    m = short_mask.tolist()
    e = expanded_ids.tolist()
    n_short, n_long = len(s), len(e)
    out = [0] * n_long

    i = j = 0
    while i < n_short and j < n_long:
        s_tok = s[i]
        e_tok = e[j]
        if s_tok == e_tok and s_tok not in pad_token_ids:
            out[j] = m[i]
            i += 1
            j += 1
            continue
        if s_tok in pad_token_ids and e_tok == s_tok:
            mask_val = m[i]
            while j < n_long and e[j] == s_tok:
                out[j] = mask_val
                j += 1
            i += 1
            continue
        raise ValueError(
            f"SFT alignment failed: short[{i}]={s_tok} vs expanded[{j}]={e_tok} "
            f"(pad_token_ids={sorted(pad_token_ids)}). "
            f"Chat-template tokenizer and processor disagree on a non-multimodal token."
        )

    if i != n_short or j != n_long:
        raise ValueError(
            f"SFT alignment ended early: short {i}/{n_short}, expanded {j}/{n_long}. "
            f"Trailing tokens on one side cannot be matched."
        )
    return torch.tensor(out, dtype=torch.long)


def pack_samples_for_tq(
    samples: list[ProcessedSample],
    force_multimodal_field: bool = False,
    sample_weights: list[float] | None = None,
) -> Optional[dict]:
    if not samples:
        return None
    tokens = [s.tokens for s in samples]
    loss_masks = [s.loss_mask for s in samples]
    total_lengths = [s.total_length for s in samples]
    has_mm = force_multimodal_field or any(s.multimodal_train_inputs is not None for s in samples)
    is_classification = any(s.classification_label is not None for s in samples)
    if is_classification and not all(s.classification_label is not None for s in samples):
        raise ValueError("classification labels must be present for every sample in a batch")
    batch = {
        "tokens": tokens,
        "loss_masks": loss_masks,
        "total_lengths": total_lengths,
        # Megatron data.py uses response_length == total_length to select
        # the SFT loss-mask alignment path; loss_masks carry the actual
        # assistant/function_call token positions.
        "response_lengths": [1] * len(samples) if is_classification else total_lengths,
    }
    if is_classification:
        batch["classification_labels"] = [s.classification_label.tolist() for s in samples]
    if sample_weights is not None:
        if len(sample_weights) != len(samples):
            raise ValueError(f"sample_weights length must match samples: {len(sample_weights)} != {len(samples)}")
        batch["sample_weights"] = sample_weights
    if has_mm:
        batch["multimodal_train_inputs"] = [s.multimodal_train_inputs for s in samples]
    batch_size = len(samples)
    for key, value in batch.items():
        if len(value) != batch_size:
            raise ValueError(f"SFT batch field {key!r} has {len(value)} rows; expected {batch_size}")
    return batch


__all__ = [
    "ProcessedSample",
    "SFTStreamingDataset",
    "pack_samples_for_tq",
]
