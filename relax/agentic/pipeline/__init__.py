# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Typed values crossing Agentic pipeline boundaries."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np
import torch

from relax.utils.types import Sample


_TRAINING_ARTIFACT_ARRAY_FIELDS: list[tuple[str, Any]] = [
    ("tokens", np.int32),
    ("rollout_tokens", np.int32),
    ("loss_mask", np.uint8),
    ("rollout_log_probs", np.float64),
    ("teacher_log_probs", np.float64),
    ("teacher_topk_token_ids", np.int32),
]


class RuntimeGroupError(RuntimeError):
    """Failure of one actor-local resident Group."""


@dataclass(frozen=True)
class SessionGroupProgress:
    """Latest barrier state for one changed resident Group."""

    group_id: str
    ready_for_lease: bool
    takeable_session_ids: Tuple[str, ...]
    interrupted: bool
    protected: bool


@dataclass(frozen=True)
class SessionShardProgress:
    """Incremental progress returned by one Shard event stream."""

    revision: int
    groups: Tuple[SessionGroupProgress, ...]


@dataclass(frozen=True)
class GroupInput:
    """Immutable group input passed from Prepare to Runtime.

    Source Samples leave the lifecycle after Runtime projects them into Session
    specifications.
    """

    group_id: str
    samples: list[Sample]


@dataclass
class SessionSpec:
    """Session creation specification sent from Runtime to a Shard."""

    session_id: str
    group_index: int | None = None
    index: int | None = None
    label: str | None = None
    train_metadata: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    input_payload: dict[str, Any] = field(default_factory=dict)
    sampling_params: dict[str, Any] | None = None


@dataclass
class SampleExport:
    """One Sample exported from a terminal SessionForest."""

    name: str | None
    sample: Sample


@dataclass(frozen=True)
class SessionExport:
    """All Samples exported by one terminal Session.

    Runtime releases the Session's volatile resources before publishing this
    value.
    """

    exports: Tuple[SampleExport, ...]


@dataclass
class SessionExportTransport:
    """Sample export payloads published by one finalized Session."""

    exports: tuple[dict[str, Any], ...]


@dataclass
class TrainingFieldArtifact:
    sample_payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sample(cls, sample: Sample) -> "TrainingFieldArtifact":
        return cls(sample_payload=_compact_sample_payload(sample.to_dict()))

    def to_sample(self) -> Sample:
        return Sample.from_dict(_expand_sample_payload(self.sample_payload))


def _plain_training_graph(value: Any, *, safe_ids: set[int]) -> bool:
    """Exclude user copy hooks before changing the order of numeric copies."""
    pending = [value]
    seen = set(safe_ids)
    scalar_types = (type(None), bool, int, float, complex, str, bytes)
    while pending:
        item = pending.pop()
        if type(item) in scalar_types:
            continue
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        item_type = type(item)
        if item_type is dict:
            pending.extend(item.keys())
            pending.extend(item.values())
        elif item_type in (list, tuple, set, frozenset):
            pending.extend(item)
        elif item_type is np.ndarray and not item.dtype.hasobject:
            continue
        elif item_type is torch.Tensor and not item.__dict__ and not item.requires_grad and item.grad is None:
            # Storage is also a Python object whose copy hook may be replaced.
            if item.device.type != "cpu" or item.layout is not torch.strided or item.is_quantized:
                return False
            storage = item.untyped_storage()
            if type(storage) is not torch.UntypedStorage or storage.__dict__:
                return False
        else:
            return False
    return True


def _compact_sample_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # Converting a plain numeric list already allocates independent storage.
    array_lists: dict[str, list[Any]] = {}
    scalar_types = (int, float, bool)
    if type(payload) is dict:
        for field_name, _dtype in _TRAINING_ARTIFACT_ARRAY_FIELDS:
            value = payload.get(field_name)
            if type(value) is list and value and all(type(item) in scalar_types for item in value):
                array_lists[field_name] = value
    if array_lists and _plain_training_graph(payload, safe_ids={id(value) for value in array_lists.values()}):
        # Unknown copy hooks can inspect the root memo or mutate a source list.
        # Only plain data may skip the original ordered deepcopy traversal.
        compacted: dict[str, Any] = {}
        remainder = {key: value for key, value in payload.items() if key not in array_lists}
        copied = copy.deepcopy(remainder, {id(payload): compacted})
        compacted.update((key, array_lists[key] if key in array_lists else copied[key]) for key in payload)
    else:
        array_lists.clear()
        compacted = copy.deepcopy(payload)

    tokens = compacted.get("tokens")
    rollout_tokens = compacted.get("rollout_tokens")
    if isinstance(tokens, list) and isinstance(rollout_tokens, list) and rollout_tokens == tokens:
        compacted["rollout_tokens"] = None
        compacted["_rollout_tokens_shared"] = True

    for field_name, dtype in _TRAINING_ARTIFACT_ARRAY_FIELDS:
        value = compacted.get(field_name)
        if not isinstance(value, list) or not value:
            continue
        if field_name in array_lists:
            compacted[field_name] = np.fromiter(value, dtype=dtype, count=len(value))
        else:
            compacted[field_name] = np.asarray(value, dtype=dtype)
    return compacted


def _expand_sample_payload(payload: dict[str, Any]) -> dict[str, Any]:
    expanded = copy.deepcopy(payload)

    def _tolist(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    tokens = _tolist(expanded.get("tokens"))
    if isinstance(tokens, list):
        expanded["tokens"] = tokens

    rollout_tokens = _tolist(expanded.get("rollout_tokens"))
    if expanded.pop("_rollout_tokens_shared", False):
        expanded["rollout_tokens"] = list(tokens or [])
    elif isinstance(rollout_tokens, list):
        expanded["rollout_tokens"] = rollout_tokens

    for field_name, _dtype in _TRAINING_ARTIFACT_ARRAY_FIELDS:
        if field_name in {"tokens", "rollout_tokens"}:
            continue
        value = _tolist(expanded.get(field_name))
        if isinstance(value, list):
            expanded[field_name] = value
    return expanded


@dataclass(frozen=True)
class GroupExport:
    """Complete exports from every original Session in one Group.

    Session exports preserve their original order through Runtime and Reward.
    """

    group_id: str
    sessions: Tuple[SessionExport, ...]

    @property
    def samples(self) -> list[Sample]:
        """Flatten exported Samples after the complete-group barrier."""

        return [export.sample for session in self.sessions for export in session.exports]
