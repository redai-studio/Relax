# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Versioned creation, trajectory, stage-event, and training-input records.

Export IDs are int64 inside Relax and decimal strings in JSON (including nested
train samples). Runtime group identity is distinct from group_index, which is
the algorithm's reward-normalization grouping. Each stream is sharded; readers
join by run_id and the corresponding scope identity.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


# Bit layout of sample_export_id. The shard ordinal is bounded by
# _DEFAULT_SESSION_SHARD_COUNT (16), so 8 bits leaves room to grow while the
# 56-bit counter stays far below int64 overflow.
SHARD_ORDINAL_SHIFT = 56
UNASSIGNED_SAMPLE_EXPORT_ID = -1
OBSERVATION_DATA_FIELDS = ("sample_export_ids", "agentic_group_ids")

# A major bump covers incompatible field/type changes; a minor bump is additive.
# Version 2 writes sample_export_id as a JSON string instead of a number.
SCHEMA_VERSION = "3.0.0"


def utc_now() -> str:
    """UTC event time, captured by the producer rather than the writer."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def encode_group_ids(groups: list[str]) -> list[list[int]]:
    """Encode UTF-8 identities with a NUL terminator for TransferQueue."""
    return [list(group.encode("utf-8")) + [0] for group in groups]


def group_ids_to_tensor(rows: list[list[int]], *, device: Any = None) -> Any:
    """Pad encoded identities into one dense uint8 column."""
    import torch

    width = max(map(len, rows), default=1)
    return torch.tensor([row + [0] * (width - len(row)) for row in rows], dtype=torch.uint8, device=device)


def decode_group_ids(value: Any) -> list[str]:
    """Decode dense or TQ-reconstructed jagged UTF-8 identity rows on CPU."""
    import torch

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("agentic_group_ids must stay on CPU")
        value = value.unbind(0)
    result = []
    for row in value:
        if isinstance(row, str):
            result.append(row)
            continue
        if isinstance(row, torch.Tensor):
            if row.device.type != "cpu":
                raise ValueError("agentic_group_ids must stay on CPU")
            row = row.tolist()
        result.append(bytes(row).rstrip(b"\x00").decode("utf-8"))
    return result


_DATA_URI_RE = re.compile(r"^data:([^;,]*)(;base64)?,", re.IGNORECASE)


def pack_sample_export_id(shard_ordinal: int, counter: int) -> int:
    """Pack a shard-local counter into a globally unique int64."""

    if shard_ordinal < 0:
        raise ValueError(f"shard_ordinal must be non-negative, got {shard_ordinal}")
    if counter < 0:
        raise ValueError(f"counter must be non-negative, got {counter}")
    return (shard_ordinal << SHARD_ORDINAL_SHIFT) | counter


def unpack_sample_export_id(sample_export_id: int) -> tuple[int, int]:
    """Split a packed id back into ``(shard_ordinal, counter)``."""

    if sample_export_id < 0:
        raise ValueError(f"cannot unpack an unassigned id: {sample_export_id}")
    return sample_export_id >> SHARD_ORDINAL_SHIFT, sample_export_id & ((1 << SHARD_ORDINAL_SHIFT) - 1)


def strip_media(messages: list[dict[str, Any]], *, media_sink: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Replace inline ``data:`` URIs with a content digest.

    Agentic multimodal observations inline images as base64 data URIs, so a
    single message can carry hundreds of KB. Trajectories keep the digest and
    the byte count; the bytes go to ``media_sink`` only when the caller asked
    for them (``--dump-agentic-obs-media``).
    """

    def _convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_convert(item) for item in value]
        if isinstance(value, (bytes, bytearray)):
            import base64

            digest = hashlib.sha256(value).hexdigest()
            if media_sink is not None:
                media_sink.setdefault(
                    digest, "data:application/octet-stream;base64," + base64.b64encode(value).decode("ascii")
                )
            return f"<media:sha256:{digest}:{len(value)}>"
        if isinstance(value, str) and _DATA_URI_RE.match(value):
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            if media_sink is not None:
                media_sink.setdefault(digest, value)
            return f"<media:sha256:{digest}:{len(value)}>"
        return value

    return [_convert(message) for message in messages]


@dataclass
class LineageNode:
    """One committed SessionForest node on the exported lineage.

    Deliberately narrow: ``agentic_trace.turns[]`` already carries every
    structural field for ``resp`` nodes (parent hash, rollout id, abort count,
    status, response length, timings, request id). This type only adds what the
    trace has no place for — ``obs`` nodes, message content, and the per-node
    weight versions that ``build_sample`` flattens away.
    """

    kind: str
    state_hash: str
    token_count: int
    weight_versions: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptRecord:
    run_id: str
    prompt_id: str
    dataset_id: str | None
    dataset_version: str | None
    split: str
    source_row_id: str | None
    input_payload: dict[str, Any]
    prompt_preview: str
    input_payload_ref: str | None = None
    identity_kind: str = "source_row"

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


@dataclass
class GroupRecord:
    run_id: str
    group_id: str
    group_index: int | None
    prompt_id: str
    expected_group_size: int
    rollout_mode: str
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


@dataclass
class SessionRecord:
    run_id: str
    session_id: str
    group_id: str
    group_index: int | None
    sample_index: int | None
    prompt_id: str
    replicate_index: int
    expected_group_size: int
    sampling_config: dict[str, Any]
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


@dataclass
class ExportTrajectoryRecord:
    """One exported training context, written by the owning SessionShard.

    Written unconditionally at export time. Whether the sample reached training
    is recorded separately by :class:`ExportStageRecord`, so a dropped sample
    still has a full trajectory to inspect.
    """

    sample_export_id: int
    session_id: str
    export_name: str | None
    group_index: int | None
    index: int | None
    label: str | None
    leaf_state_hash: str
    group_id: str | None = None
    rollout_mode: str | None = None
    lineage: list[LineageNode] = field(default_factory=list)
    agentic_trace: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    exported_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        payload["sample_index"] = payload.pop("index")
        return payload


@dataclass
class ExportStageRecord:
    """One boundary of a scoped stage interval, paired by stage_id.

    Events are append-only. is_terminal closes the owner lifecycle,
    independently of whether the stage has ended. Update and rollout events
    carry their own scope identity and group membership instead of
    sample_export_id.
    """

    sample_export_id: int | None
    stage: str
    status: str
    group_id: str
    reason: str | None = None
    partition_id: int | None = None
    reward_value: float | None = None
    reward_detail: dict[str, Any] | None = None
    run_id: str = ""
    scope: str = "trajectory"
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    stage_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    retry_of_stage_id: str | None = None
    transfer_batch_id: str | None = None
    operation: str | None = None
    event_seq: int = 1
    occurred_at: str = field(default_factory=utc_now)
    is_terminal: bool = False
    reason_code: str | None = None
    category: str | None = None
    rollout_id: int | None = None
    rollout_mode: str | None = None
    update_id: str | None = None
    group_ids: list[str] | None = None
    groups_finalized: bool | None = None
    dp_rank: int | None = None
    dp_world_size: int | None = None
    step_id: int | None = None
    train_generation_id: str | None = None
    update_sequence: int | None = None
    optimizer_successful: bool | None = None
    optimizer_step: int | None = None
    input_complete: bool | None = None
    session_id: str | None = None
    wall_ms: float | None = None
    active_ms: float | None = None
    request_count: int | None = None
    attempt_count: int | None = None
    generated_tokens: int | None = None
    token_count_complete: bool | None = None
    export_ids: list[int] | None = None
    log_ref: str | None = None
    request_id: str | None = None
    sampling_config: dict[str, Any] | None = None
    llm_calls: list[dict[str, Any]] | None = None
    llm_calls_complete: bool | None = None
    diagnostics: list[dict[str, Any]] | None = None
    run_config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        for key in (
            "sample_export_id",
            "update_id",
            "rollout_id",
            "group_ids",
            "groups_finalized",
            "optimizer_step",
            "dp_rank",
            "dp_world_size",
            "step_id",
            "update_sequence",
            "train_generation_id",
            "optimizer_successful",
            "input_complete",
            "session_id",
            "wall_ms",
            "active_ms",
            "request_count",
            "attempt_count",
            "generated_tokens",
            "token_count_complete",
            "export_ids",
            "request_id",
            "sampling_config",
            "retry_of_stage_id",
            "transfer_batch_id",
            "operation",
            "llm_calls",
            "llm_calls_complete",
            "diagnostics",
            "run_config",
        ):
            if payload[key] is None:
                del payload[key]
        if self.token_count_complete is False:
            payload["generated_tokens"] = None
        return payload


@dataclass
class TrainScalarRecord:
    """Per-sample training scalars, written by the actor.

    Legacy per-sample value object. ``advantage_*`` are ``None`` under context
    parallelism because each rank holds a zig-zag shard of the sequence, which
    makes a per-sample weighted mean wrong rather than merely partial.
    """

    sample_export_id: int
    rollout_id: int
    dp_rank: int
    effective_tokens: int
    response_length: int
    total_length: int
    advantage_mean: float | None = None
    advantage_absmax: float | None = None
    returns_mean: float | None = None
    group_id: str = ""
    loss_mask: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainRecord:
    """One DP shard of the real inputs consumed by one policy update."""

    run_id: str
    update_id: str
    samples: list[dict[str, Any]]
    dp_rank: int
    dp_world_size: int
    input_complete: bool = True
    capture_phase: str = "training_input"
    stage_id: str | None = None
    train_generation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


__all__ = [
    "SCHEMA_VERSION",
    "UNASSIGNED_SAMPLE_EXPORT_ID",
    "ExportStageRecord",
    "ExportTrajectoryRecord",
    "GroupRecord",
    "LineageNode",
    "PromptRecord",
    "SessionRecord",
    "TrainScalarRecord",
    "TrainRecord",
    "pack_sample_export_id",
    "strip_media",
    "unpack_sample_export_id",
]
