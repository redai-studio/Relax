# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Typed values crossing Agentic pipeline boundaries."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterator, Tuple

import numpy as np

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


def _can_compact_without_deepcopy_hooks(payload: dict[str, Any]) -> bool:
    """Admit only a bounded-depth, unaliased tree of exact builtin values.

    Inspecting an unknown value must not invoke its iteration, comparison,
    hashing, conversion or copy hooks. NumPy/Torch values, subclasses, cycles
    and repeated container identities therefore use the original path. The scan
    is O(n), with O(depth) iterator storage and O(containers) identities; its
    cost must be measured rather than assumed negligible.
    """
    if type(payload) is not dict:
        return False
    seen: set[int] = set()
    pending: list[Iterator[Any]] = [iter((payload,))]
    while pending:
        try:
            value = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        value_type = type(value)
        # Identity checks avoid custom metaclass equality or hashing during
        # eligibility checks. These exact immutable builtin leaves have no
        # object-defined deepcopy/conversion hooks.
        if (
            value is None
            or value_type is int
            or value_type is float
            or value_type is bool
            or value_type is str
            or value_type is bytes
            or value_type is complex
        ):
            continue
        if value_type is dict:
            if any(type(key) is not str for key in value):
                return False
            children = iter(value.values())
        elif value_type is list or value_type is tuple:
            children = iter(value)
        else:
            return False
        identity = id(value)
        if identity in seen or len(pending) >= 128:
            return False
        seen.add(identity)
        pending.append(children)
    return True


def _compact_sample_payload_original(payload: dict[str, Any]) -> dict[str, Any]:
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
        compacted[field_name] = np.asarray(value, dtype=dtype)
    return compacted


def _compact_sample_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not _can_compact_without_deepcopy_hooks(payload):
        return _compact_sample_payload_original(payload)

    # Numeric lists become independently owned arrays below. Deep-copying
    # them first duplicates every Python element immediately before discarding
    # the intermediate lists. Keep deepcopy for all other (including dynamic)
    # Sample fields so the transport retains its ownership boundary.
    compacted = dict(payload)

    tokens = compacted.get("tokens")
    rollout_tokens = compacted.get("rollout_tokens")
    if isinstance(tokens, list) and isinstance(rollout_tokens, list) and rollout_tokens == tokens:
        compacted["rollout_tokens"] = None
        compacted["_rollout_tokens_shared"] = True

    arrays: dict[str, np.ndarray] = {}
    for field_name, dtype in _TRAINING_ARTIFACT_ARRAY_FIELDS:
        value = compacted.get(field_name)
        if not isinstance(value, list) or not value:
            continue
        arrays[field_name] = np.asarray(value, dtype=dtype)
    compacted = copy.deepcopy({key: value for key, value in compacted.items() if key not in arrays})
    compacted.update(arrays)
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
