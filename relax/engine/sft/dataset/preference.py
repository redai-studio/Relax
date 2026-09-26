# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Streaming chosen/rejected dataset for offline preference objectives."""

import hashlib
import threading
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from relax.engine.sft.dataset.chat_template import (
    HAS_GENERATION_MARKER,
    _resolve_sft_template_kwargs,
    render_with_loss_mask,
)
from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample
from relax.engine.sft.dataset.streaming import _build_reader
from relax.utils.data.streaming_dataset import IndexManager, PrefetchBuffer
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class PreferenceDataError(ValueError):
    """Stable, classified preference-row rejection."""

    def __init__(self, reason_code: str, message: str, *, source_idx: int, pair_id: str | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.source_idx = source_idx
        self.pair_id = pair_id


class _PreferenceRowError(ValueError):
    """Internal row rejection that carries its stable reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _classify_preference_error(error: BaseException) -> str:
    # Prefer the reason code attached at the raise site; message matching is
    # only a fallback for errors raised outside this module (e.g. tokenizer).
    reason_code = getattr(error, "reason_code", None)
    if isinstance(reason_code, str) and reason_code:
        return reason_code
    message = str(error).lower()
    if "post-truncation identical" in message:
        return "post_truncation"
    if "must not be identical" in message:
        return "identical"
    if "prompt tokens differ" in message or "strict common prefix" in message:
        return "prompt_mismatch"
    if "empty completion" in message or "no supervised tokens" in message:
        return "empty_completion"
    if "capacity" in message or "exceeds max_length" in message:
        return "oversize"
    return "schema"


@dataclass(frozen=True)
class PreferencePair:
    """Canonical text-only preference pair before tokenization."""

    pair_id: str
    prompt: list[CanonicalMessage]
    chosen: CanonicalMessage
    rejected: CanonicalMessage
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ProcessedPreferencePair:
    """Tokenized pair kept atomic until after DP assignment."""

    pair_id: str
    chosen_tokens: torch.Tensor
    rejected_tokens: torch.Tensor
    chosen_loss_mask: torch.Tensor
    rejected_loss_mask: torch.Tensor
    chosen_total_length: int
    rejected_total_length: int
    chosen_prompt_length: int
    rejected_prompt_length: int
    chosen_completion_length: int
    rejected_completion_length: int
    source_idx: int

    @property
    def pair_total_length(self) -> int:
        return self.chosen_total_length + self.rejected_total_length


def _message_from_raw(raw: Any, *, learn: bool, field: str) -> CanonicalMessage:
    if not isinstance(raw, dict):
        raise _PreferenceRowError("schema", f"preference {field} must be a message object")
    role = raw.get("role")
    content = raw.get("content")
    if role is None or content is None:
        raise _PreferenceRowError("schema", f"preference {field} message requires role and content")
    if not isinstance(content, str):
        raise _PreferenceRowError("schema", f"preference {field} must be pure text")
    return CanonicalMessage(role=role, content=content, learn=learn, tool_calls=raw.get("tool_calls"))


def _normalize_pair_row(
    row: dict[str, Any],
    *,
    row_index: int,
    prompt_key: str,
    chosen_key: str,
    rejected_key: str,
    pair_id_key: str,
    metadata_key: str,
    source_name: str,
) -> PreferencePair:
    pair_id = row.get(pair_id_key)
    if not isinstance(pair_id, str) or not pair_id:
        raise _PreferenceRowError("schema", f"preference row requires a non-empty {pair_id_key}")
    chosen_raw = row.get(chosen_key)
    rejected_raw = row.get(rejected_key)
    prompt_raw = row.get(prompt_key)

    if prompt_raw is None:
        if not isinstance(chosen_raw, list) or not isinstance(rejected_raw, list):
            raise _PreferenceRowError("schema", "implicit preference rows require chosen/rejected message lists")
        prefix_length = 0
        for chosen_message, rejected_message in zip(chosen_raw, rejected_raw, strict=False):
            if chosen_message != rejected_message:
                break
            prefix_length += 1
        chosen_suffix = chosen_raw[prefix_length:]
        rejected_suffix = rejected_raw[prefix_length:]
        if len(chosen_suffix) != 1 or len(rejected_suffix) != 1:
            raise _PreferenceRowError(
                "prompt_mismatch",
                "implicit preference rows require one assistant message after the strict common prefix",
            )
        prompt_raw = chosen_raw[:prefix_length]
        chosen_raw = chosen_suffix[0]
        rejected_raw = rejected_suffix[0]
    elif not isinstance(prompt_raw, list):
        raise _PreferenceRowError("schema", f"preference {prompt_key} must be a message list")

    prompt = [_message_from_raw(message, learn=False, field=prompt_key) for message in prompt_raw]
    chosen = _message_from_raw(chosen_raw, learn=True, field=chosen_key)
    rejected = _message_from_raw(rejected_raw, learn=True, field=rejected_key)
    if chosen.role != "assistant" or rejected.role != "assistant":
        raise _PreferenceRowError("schema", "preference chosen/rejected messages must have role assistant")
    if chosen.content == rejected.content:
        raise _PreferenceRowError("identical", "preference chosen/rejected responses must not be identical")

    metadata = row.get(metadata_key) or {}
    if not isinstance(metadata, dict):
        raise _PreferenceRowError("schema", f"preference {metadata_key} must be an object")
    metadata = dict(metadata)
    metadata.update({"source_dataset": source_name, "row_index": row_index, "pair_id": pair_id})
    return PreferencePair(pair_id=pair_id, prompt=prompt, chosen=chosen, rejected=rejected, metadata=metadata)


def _split_branch(
    tokens: torch.Tensor, mask: torch.Tensor, *, pair_id: str, branch: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 1 or mask.ndim != 1 or tokens.shape != mask.shape:
        raise _PreferenceRowError(
            "schema", f"preference pair {pair_id!r} {branch} tokens/mask must be aligned one-dimensional tensors"
        )
    supervised = torch.nonzero(mask, as_tuple=False).flatten()
    if supervised.numel() == 0:
        raise _PreferenceRowError(
            "empty_completion", f"preference pair {pair_id!r} {branch} completion has no supervised tokens"
        )
    first = int(supervised[0])
    if not bool(mask[first:].to(dtype=torch.bool).all()):
        raise _PreferenceRowError(
            "schema", f"preference pair {pair_id!r} {branch} completion mask must be one contiguous suffix"
        )
    return tokens[:first], tokens[first:]


def _truncate_pair(
    prompt: torch.Tensor,
    chosen_completion: torch.Tensor,
    rejected_completion: torch.Tensor,
    *,
    pair_id: str,
    max_length: int,
    max_completion_length: int,
    pair_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen_completion = chosen_completion[:max_completion_length]
    rejected_completion = rejected_completion[:max_completion_length]
    if chosen_completion.numel() == 0 or rejected_completion.numel() == 0:
        raise _PreferenceRowError(
            "empty_completion", f"preference pair {pair_id!r} has an empty completion after truncation"
        )
    prompt_budget = max_length - max(chosen_completion.numel(), rejected_completion.numel())
    if prompt_budget < 0:
        raise _PreferenceRowError(
            "oversize", f"preference pair {pair_id!r} completion exceeds max_length={max_length}"
        )
    prompt = prompt[-prompt_budget:] if prompt_budget else prompt[:0]

    def total() -> int:
        return 2 * prompt.numel() + chosen_completion.numel() + rejected_completion.numel()

    while total() > pair_capacity:
        if chosen_completion.numel() >= rejected_completion.numel() and chosen_completion.numel() > 1:
            chosen_completion = chosen_completion[:-1]
        elif rejected_completion.numel() > 1:
            rejected_completion = rejected_completion[:-1]
        elif prompt.numel() > 0:
            prompt = prompt[1:]
        else:
            raise _PreferenceRowError(
                "oversize",
                f"preference pair {pair_id!r} cannot fit pair capacity {pair_capacity} while retaining both completions",
            )
    return prompt.contiguous(), chosen_completion.contiguous(), rejected_completion.contiguous()


class PreferenceStreamingDataset:
    """Lazy text-only preference dataset with deterministic epoch shuffling."""

    def __init__(
        self,
        path: str | list[str] | tuple[str, ...],
        *,
        tokenizer=None,
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        pair_id_key: str = "prompt_id",
        metadata_key: str = "metadata",
        source_name: str = "preference_data",
        max_length: int = 1024,
        max_completion_length: int = 512,
        pair_capacity: int | None = None,
        seed: int = 42,
        prefetch_max_cached: int = 256,
        prefetch_chunk_size: int = 32,
        prefetch_num_workers: int = 4,
        apply_chat_template_kwargs: dict | None = None,
        expected_chat_template_sha256: str | None = None,
        require_no_generation_marker: bool = False,
    ) -> None:
        if max_length <= 0 or max_completion_length <= 0:
            raise ValueError("preference length limits must be positive")
        self.reader = _build_reader(path)
        self.index_manager = IndexManager(len(self.reader), seed=seed)
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.pair_id_key = pair_id_key
        self.metadata_key = metadata_key
        self.source_name = source_name
        self.max_length = max_length
        self.max_completion_length = max_completion_length
        self.pair_capacity = pair_capacity or (2 * max_length)
        self.apply_chat_template_kwargs = apply_chat_template_kwargs
        self.expected_chat_template_sha256 = expected_chat_template_sha256
        self.require_no_generation_marker = require_no_generation_marker
        self._rejection_counts: Counter[str] = Counter()
        self._rejection_records: list[dict[str, Any]] = []
        self._rejection_lock = threading.Lock()
        self._template_contract_validated = False
        self._template_contract_lock = threading.Lock()
        self._validate_unique_pair_ids()
        self._first_error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._prefetch: PrefetchBuffer | None = None
        if prefetch_max_cached > 0:
            self._prefetch = PrefetchBuffer(
                process_fn=self._process_one_safe,
                chunk_size=prefetch_chunk_size,
                max_cached=prefetch_max_cached,
                num_workers=prefetch_num_workers,
            )

    def __len__(self) -> int:
        return len(self.reader)

    def _validate_unique_pair_ids(self) -> None:
        seen: set[str] = set()
        for index in range(len(self.reader)):
            pair_id = self.reader[index].get(self.pair_id_key)
            if not isinstance(pair_id, str) or not pair_id:
                message = f"preference row {index} requires a non-empty {self.pair_id_key}"
                with self._rejection_lock:
                    self._rejection_counts["schema"] += 1
                    self._rejection_records.append(
                        {"source_idx": index, "pair_id": None, "reason_code": "schema", "message": message}
                    )
                raise PreferenceDataError("schema", message, source_idx=index)
            if pair_id in seen:
                message = f"duplicate preference pair ID {pair_id!r} at row {index}"
                with self._rejection_lock:
                    self._rejection_counts["schema"] += 1
                    self._rejection_records.append(
                        {"source_idx": index, "pair_id": pair_id, "reason_code": "schema", "message": message}
                    )
                raise PreferenceDataError("schema", message, source_idx=index, pair_id=pair_id)
            seen.add(pair_id)

    def restrict_training_indices(self, indices: Iterable[int], *, dataset_seed_offset: int = 0) -> None:
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
        self.index_manager = IndexManager(
            len(index_pool),
            seed=self.index_manager.seed,
            index_pool=index_pool,
            dataset_seed_offset=dataset_seed_offset,
        )

    def get_batch_by_indices(self, indices: Iterable[int]) -> list[ProcessedPreferencePair]:
        """Read physical rows in order without advancing the training
        cursor."""
        self._raise_if_failed()
        indices = tuple(indices)
        if any(not isinstance(index, int) or index < 0 or index >= len(self.reader) for index in indices):
            raise ValueError(f"row indices must be integers in [0, {len(self.reader)})")
        return [self.get_processed_pair(index) for index in indices]

    def shuffle(self, epoch_id: int, position: int = 0) -> None:
        self.index_manager.shuffle(epoch_id)
        if position:
            self.index_manager.position = min(position, self.index_manager.total_size)
        if self._prefetch is not None and self.index_manager.indices is not None:
            remaining = self.index_manager.indices[self.index_manager.position :]
            self._prefetch.set_index_order(list(remaining))

    def stop(self) -> None:
        if self._prefetch is not None:
            self._prefetch.stop()

    def get_canonical_pair(self, idx: int) -> PreferencePair:
        return _normalize_pair_row(
            self.reader[idx],
            row_index=idx,
            prompt_key=self.prompt_key,
            chosen_key=self.chosen_key,
            rejected_key=self.rejected_key,
            pair_id_key=self.pair_id_key,
            metadata_key=self.metadata_key,
            source_name=self.source_name,
        )

    def get_processed_pair(self, idx: int) -> ProcessedPreferencePair:
        try:
            return self._get_processed_pair(idx)
        except PreferenceDataError:
            raise
        except Exception as exc:
            pair_id = None
            try:
                raw_pair_id = self.reader[idx].get(self.pair_id_key)
                pair_id = raw_pair_id if isinstance(raw_pair_id, str) else None
            except Exception:
                pass
            reason_code = _classify_preference_error(exc)
            error = PreferenceDataError(reason_code, str(exc), source_idx=idx, pair_id=pair_id)
            with self._rejection_lock:
                self._rejection_counts[reason_code] += 1
                self._rejection_records.append(
                    {"source_idx": idx, "pair_id": pair_id, "reason_code": reason_code, "message": str(exc)}
                )
            logger.error(
                "Rejected preference pair source_idx=%s pair_id=%r reason_code=%s counts=%s",
                idx,
                pair_id,
                reason_code,
                dict(self.rejection_counts),
            )
            raise error from exc

    def _get_processed_pair(self, idx: int) -> ProcessedPreferencePair:
        if self.tokenizer is None:
            raise RuntimeError("PreferenceStreamingDataset requires a tokenizer for processing")
        pair = self.get_canonical_pair(idx)
        chosen_sample = CanonicalSample(messages=[*pair.prompt, pair.chosen], metadata=dict(pair.metadata), tools=None)
        rejected_sample = CanonicalSample(
            messages=[*pair.prompt, pair.rejected], metadata=dict(pair.metadata), tools=None
        )
        self._validate_template_contract(chosen_sample)
        chosen_tokens, chosen_mask = render_with_loss_mask(
            chosen_sample,
            tokenizer=self.tokenizer,
            apply_chat_template_kwargs=self.apply_chat_template_kwargs,
        )
        rejected_tokens, rejected_mask = render_with_loss_mask(
            rejected_sample,
            tokenizer=self.tokenizer,
            apply_chat_template_kwargs=self.apply_chat_template_kwargs,
        )
        chosen_prompt, chosen_completion = _split_branch(
            chosen_tokens, chosen_mask, pair_id=pair.pair_id, branch="chosen"
        )
        rejected_prompt, rejected_completion = _split_branch(
            rejected_tokens, rejected_mask, pair_id=pair.pair_id, branch="rejected"
        )
        if not torch.equal(chosen_prompt, rejected_prompt):
            raise _PreferenceRowError(
                "prompt_mismatch", f"preference pair {pair.pair_id!r} chosen/rejected prompt tokens differ"
            )
        prompt, chosen_completion, rejected_completion = _truncate_pair(
            chosen_prompt,
            chosen_completion,
            rejected_completion,
            pair_id=pair.pair_id,
            max_length=self.max_length,
            max_completion_length=self.max_completion_length,
            pair_capacity=self.pair_capacity,
        )
        chosen_tokens = torch.cat((prompt, chosen_completion))
        rejected_tokens = torch.cat((prompt, rejected_completion))
        if torch.equal(chosen_tokens, rejected_tokens) or torch.equal(chosen_completion, rejected_completion):
            raise _PreferenceRowError(
                "post_truncation", f"preference pair {pair.pair_id!r} is post-truncation identical"
            )
        chosen_mask = torch.cat((torch.zeros_like(prompt), torch.ones_like(chosen_completion)))
        rejected_mask = torch.cat((torch.zeros_like(prompt), torch.ones_like(rejected_completion)))
        return ProcessedPreferencePair(
            pair_id=pair.pair_id,
            chosen_tokens=chosen_tokens,
            rejected_tokens=rejected_tokens,
            chosen_loss_mask=chosen_mask,
            rejected_loss_mask=rejected_mask,
            chosen_total_length=chosen_tokens.numel(),
            rejected_total_length=rejected_tokens.numel(),
            chosen_prompt_length=prompt.numel(),
            rejected_prompt_length=prompt.numel(),
            chosen_completion_length=chosen_completion.numel(),
            rejected_completion_length=rejected_completion.numel(),
            source_idx=idx,
        )

    @property
    def rejection_counts(self) -> dict[str, int]:
        """Return a thread-safe snapshot of classified rejection counts."""
        with self._rejection_lock:
            return dict(self._rejection_counts)

    @property
    def rejection_records(self) -> list[dict[str, Any]]:
        """Return row IDs and stable reason codes for evidence manifests."""
        with self._rejection_lock:
            return [dict(record) for record in self._rejection_records]

    def _validate_template_contract(self, sample: CanonicalSample) -> None:
        if self._template_contract_validated:
            return
        with self._template_contract_lock:
            if self._template_contract_validated:
                return
            resolved = _resolve_sft_template_kwargs(
                sample,
                tokenizer=self.tokenizer,
                apply_chat_template_kwargs=self.apply_chat_template_kwargs,
            )
            template = resolved.template or ""
            digest = hashlib.sha256(template.encode()).hexdigest()
            if self.expected_chat_template_sha256 is not None and digest != self.expected_chat_template_sha256:
                raise ValueError(
                    "preference chat template SHA-256 mismatch: "
                    f"expected={self.expected_chat_template_sha256}, actual={digest}"
                )
            if self.require_no_generation_marker and HAS_GENERATION_MARKER(template):
                raise ValueError("preference recipe requires a chat template without {% generation %} markers")
            self._template_contract_validated = True

    def _process_one_safe(self, idx: int) -> ProcessedPreferencePair | None:
        try:
            return self.get_processed_pair(idx)
        except Exception as exc:
            with self._error_lock:
                if self._first_error is None:
                    self._first_error = exc
            logger.exception(f"PreferenceStreamingDataset: failed to process pair idx={idx}")
            return None

    def _raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._first_error
        if error is not None:
            raise error

    def get_batch(self, n: int) -> tuple[list[ProcessedPreferencePair], bool]:
        if n <= 0:
            raise ValueError(f"batch size must be positive, got {n}")
        self._raise_if_failed()
        pairs: list[ProcessedPreferencePair] = []
        crossed_epoch = False
        attempts = 0
        max_attempts = max(n * 10, 32)
        while len(pairs) < n and attempts < max_attempts:
            indices, crossed = self.index_manager.get_next_indices(1)
            crossed_epoch = crossed_epoch or crossed
            attempts += 1
            index = indices[0]
            pair = self._prefetch.get(index) if self._prefetch is not None else self.get_processed_pair(index)
            if pair is None:
                self._raise_if_failed()
            else:
                pairs.append(pair)
        if len(pairs) != n:
            raise RuntimeError(f"preference dataset returned a partial batch: expected {n}, got {len(pairs)}")
        return pairs, crossed_epoch

    async def get_batch_async(self, n: int) -> tuple[list[ProcessedPreferencePair], bool]:
        return self.get_batch(n)

    def get_batch_in_order(self, start: int, n: int) -> list[ProcessedPreferencePair]:
        return [self.get_processed_pair(index) for index in range(start, min(start + n, len(self.reader)))]


def pack_preference_pairs_for_tq(
    pairs: list[ProcessedPreferencePair],
) -> tuple[dict[str, list[Any]], list[dict[str, int]]]:
    """Pack atomic pair rows and matching TransferQueue length metadata."""
    if not pairs:
        raise ValueError("preference pair batch must not be empty")
    encoded_pair_ids = [
        int.from_bytes(hashlib.sha256(pair.pair_id.encode()).digest()[:8], "big") >> 1 for pair in pairs
    ]
    if len(set(encoded_pair_ids)) != len(encoded_pair_ids):
        raise ValueError("preference pair ID hash collision within batch")
    batch: dict[str, list[Any]] = {
        "pair_ids": encoded_pair_ids,
        "chosen_tokens": [pair.chosen_tokens.tolist() for pair in pairs],
        "rejected_tokens": [pair.rejected_tokens.tolist() for pair in pairs],
        "chosen_loss_masks": [pair.chosen_loss_mask.tolist() for pair in pairs],
        "rejected_loss_masks": [pair.rejected_loss_mask.tolist() for pair in pairs],
        "chosen_total_lengths": [pair.chosen_total_length for pair in pairs],
        "rejected_total_lengths": [pair.rejected_total_length for pair in pairs],
    }
    custom_meta = [{"total_lengths": pair.pair_total_length} for pair in pairs]
    if len(custom_meta) != len(pairs):
        raise RuntimeError("preference custom metadata is not row-aligned")
    return batch, custom_meta


__all__ = [
    "PreferenceDataError",
    "PreferencePair",
    "PreferenceStreamingDataset",
    "ProcessedPreferencePair",
    "pack_preference_pairs_for_tq",
]
