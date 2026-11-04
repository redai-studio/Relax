# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SessionShard-side trajectory emission and Session lifecycle accounting.

Lives in ``relax/agentic`` because it reads ``SessionForest`` internals; the
record contract and the writers are provider-agnostic and live in
``relax.utils.agentic_obs``.

Trajectories are written here, inside the shard, rather than shipped to the
rollout driver. Two reasons:

* The payload is the full message history, and agentic multimodal observations
  inline images as base64 data URIs. ``TrainingFieldArtifact`` compacts the
  seven training arrays precisely to keep the Ray hop cheap; sending the whole
  conversation alongside would cancel that out.
* The driver is a single process driving every Session's orchestration,
  admission, reward and TransferQueue traffic. The shards are 16 actors keyed by
  group hash. File I/O is safer on the shard, counter-intuitive as that reads.
"""

from __future__ import annotations

import itertools
import math
import time
import uuid
from argparse import Namespace
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from relax.utils.agentic_obs import (
    AgenticObsConfig,
    AsyncJsonlAppender,
    ExportTrajectoryRecord,
    LineageNode,
    pack_sample_export_id,
    resolve_agentic_obs_config,
    strip_media,
)
from relax.utils.agentic_obs.config import write_manifest
from relax.utils.agentic_obs.records import ExportStageRecord, utc_now
from relax.utils.agentic_obs.writers import CLOSE_TIMEOUT, best_effort, close_background
from relax.utils.logging_utils import get_logger


if TYPE_CHECKING:
    from relax.agentic.session.state import SessionForest
    from relax.utils.types import Sample


logger = get_logger(__name__)


@dataclass
class SessionObservation:
    started_at: float = field(default_factory=time.monotonic)
    stage_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at_utc: str = field(default_factory=utc_now)
    event_seq: int = 1
    terminal: bool = False
    request_count: int = 0
    attempt_count: int = 0
    generated_tokens: int = 0
    token_count_complete: bool = True
    active_ms: float = 0.0
    log_ref: str | None = None
    export_ids: list[int] = field(default_factory=list)
    failure: tuple[str, str, str] | None = None
    calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def begin_call(self, request_id: str) -> None:
        if request_id in self.calls:
            return
        self.calls[request_id] = {
            "request_id": request_id,
            "call_index": len(self.calls),
            "started_at": utc_now(),
            "finished_at": None,
            "status": "running",
            "parent_state_hash": None,
            "resp_state_hash": None,
            "attempts": [],
        }
        self.request_count = len(self.calls)

    @best_effort
    def record_backend_timing(self, attempt: dict[str, Any], meta_info: dict[str, Any]) -> None:
        # Preserve backend-reported seconds only; no estimates or tensor reads.
        keys = (
            "request_received_ts",
            "api_server_dispatch_finish_ts",
            "forward_entry_time",
            "prefill_finished_time",
            "request_finished_ts",
            "response_sent_to_client_ts",
            "queue_time",
            "prefill_waiting_latency",
            "prefill_launch_latency",
            "inference_time",
        )
        timing = {
            key: value
            for key in keys
            if type(value := meta_info.get(key)) in (int, float)
            and math.isfinite(value)
            and value >= 0
            and (not key.endswith(("_ts", "_time")) or key in ("queue_time", "inference_time") or value > 0)
        }
        if timing:
            attempt["backend_timing"] = timing

    def finish_call(self, request_id: str, status: str, **fields: Any) -> None:
        call = self.calls.get(request_id)
        if call is not None and call["finished_at"] is None:
            call.update(status=status, finished_at=utc_now(), **fields)

    def begin_attempt(self, request_id: str, rollout_id: int) -> dict[str, Any]:
        self.begin_call(request_id)
        attempts = self.calls[request_id]["attempts"]
        attempt = {
            "attempt_index": len(attempts),
            "rollout_id": rollout_id,
            "started_at": utc_now(),
            "finished_at": None,
            "status": "running",
        }
        attempts.append(attempt)
        self.attempt_count += 1
        return attempt


def resolve_shard_ordinal(default: int = 0) -> int:
    """Recover this shard's ordinal from its Ray actor name.

    Shards are created as named actors ``agentic_session_shard_<n>``
    (``agentic_session_shard_name``), so the ordinal is already durable and
    unique across the fleet. Reading it back avoids threading a new constructor
    argument through the Service layer.
    """

    try:
        import ray

        actor_name = ray.get_runtime_context().get_actor_name()
    except Exception:  # noqa: BLE001 — driver-side tests run outside an actor
        return default
    if not actor_name:
        return default
    _, separator, suffix = actor_name.rpartition("_")
    if not separator or not suffix.isdigit():
        logger.warning("agentic_obs: cannot parse shard ordinal from actor name %r", actor_name)
        return default
    return int(suffix)


class ShardTrajectoryRecorder:
    """Allocate export ids and queue trajectory records for one shard.

    Disabled unless ``--dump-agentic-obs`` names a directory, in which case
    both entry points are a single ``None`` check.
    """

    def __init__(self, args: Namespace, *, shard_ordinal: Optional[int] = None) -> None:
        self._config: Optional[AgenticObsConfig] = resolve_agentic_obs_config(args)
        self._shard_ordinal = resolve_shard_ordinal() if shard_ordinal is None else shard_ordinal
        # Counts every export this shard produced, whether or not observability
        # is on, so ids stay comparable across runs that toggle the flag.
        self._counter = itertools.count()
        self._writer: Optional[AsyncJsonlAppender] = None
        self._stage_writer: Optional[AsyncJsonlAppender] = None
        if self._config is not None:
            write_manifest(args, self._config)
            self._stage_writer = AsyncJsonlAppender(self._config.stages_dir / f"shard_{self._shard_ordinal:02d}.jsonl")
            self._writer = AsyncJsonlAppender(
                self._config.trajectory_path(self._shard_ordinal),
                media_dir=self._config.media_dir if self._config.dump_media else None,
            )

    @property
    def enabled(self) -> bool:
        return self._config is not None

    def next_sample_export_id(self) -> int:
        """Allocate the next globally unique export id for this shard."""

        return pack_sample_export_id(self._shard_ordinal, next(self._counter))

    def session_log_path(self, session_id: str) -> Path | None:
        if self._config is None:
            return None
        import hashlib

        # Route tokens are external input; never use them as relative paths.
        name = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self._config.root / "logs" / f"{name}.log"

    @best_effort
    def record_session(
        self,
        observation: SessionObservation,
        *,
        session_id: str,
        group_id: str,
        rollout_mode: str,
        rollout_id: int | None,
        stage: str,
        status: str,
        is_terminal: bool = False,
        reason_code: str | None = None,
        reason: str | None = None,
        category: str | None = None,
        request_id: str | None = None,
        sampling_config: dict[str, Any] | None = None,
    ) -> None:
        if self._config is None or self._stage_writer is None or observation.terminal:
            return
        # Request/cleanup hooks update the local ledger, never the session stream.
        if stage == "request" and request_id is not None:
            if status in ("completed", "failed", "cancelled"):
                observation.finish_call(request_id, status, reason_code=reason_code, reason=reason)
            elif sampling_config is not None and request_id in observation.calls:
                observation.calls[request_id]["sampling_config"] = sampling_config
        if stage == "cleanup" and status == "failed":
            observation.diagnostics.append({"reason_code": reason_code, "reason": reason})
        if not is_terminal:
            return
        for call in observation.calls.values():
            if call["finished_at"] is None:
                observation.finish_call(call["request_id"], "cancelled", reason_code=reason_code or "session_ended")
        observation.event_seq += 1
        observation.terminal = is_terminal
        self._stage_writer.submit(
            ExportStageRecord(
                None,
                "rollout",
                status,
                group_id,
                run_id=self._config.run_id,
                scope="session",
                session_id=session_id,
                rollout_mode=rollout_mode,
                rollout_id=rollout_id,
                event_seq=observation.event_seq,
                stage_id=observation.stage_id,
                is_terminal=is_terminal,
                reason_code=reason_code,
                reason=reason,
                category=category,
                wall_ms=max(
                    0.0,
                    (
                        datetime.now(timezone.utc)
                        - datetime.fromisoformat(observation.started_at_utc.replace("Z", "+00:00"))
                    ).total_seconds()
                    * 1000,
                ),
                active_ms=observation.active_ms,
                request_count=observation.request_count,
                attempt_count=observation.attempt_count,
                generated_tokens=observation.generated_tokens,
                token_count_complete=observation.token_count_complete,
                export_ids=list(observation.export_ids),
                log_ref=observation.log_ref,
                request_id=request_id,
                sampling_config=sampling_config,
                llm_calls=list(observation.calls.values()),
                llm_calls_complete=all(
                    call["finished_at"] is not None
                    and all(attempt["finished_at"] is not None for attempt in call["attempts"])
                    for call in observation.calls.values()
                ),
                diagnostics=observation.diagnostics or None,
            )
        )

    def record(
        self,
        *,
        forest: "SessionForest",
        state_hash: str,
        sample_export_id: int,
        metadata: dict[str, Any],
        export_name: Optional[str],
        group_id: Optional[str] = None,
        rollout_mode: Optional[str] = None,
    ) -> None:
        """Queue one trajectory record. Never blocks, never raises.

        Must be called after the export payload is built: ``build_sample``
        creates ``agentic_trace``, but the explicit-export metadata merge and
        the ``finalize_end_at`` / ``sample_export_ready_at`` marks land
        afterwards. Snapshotting earlier silently loses them.
        """

        if self._config is None or self._writer is None:
            return
        try:
            media: dict[str, str] = {}
            record = build_trajectory_record(
                forest=forest,
                state_hash=state_hash,
                sample_export_id=sample_export_id,
                metadata=metadata,
                export_name=export_name,
                config=self._config,
                media_sink=media,
                group_id=group_id,
                rollout_mode=rollout_mode,
            )
            self._writer.submit(record, media=media)
        except Exception as error:  # noqa: BLE001 — observability must not fail an export
            logger.warning("agentic_obs: failed to build trajectory for %s: %s", forest.session_id, error)
            return

    @best_effort
    def record_export_failure(
        self,
        exports: tuple[dict[str, Any], ...],
        *,
        group_id: str,
        rollout_mode: str,
        reason: str,
        reason_code: str = "session_cleanup_failed",
        category: str = "infra",
    ) -> None:
        if self._stage_writer is None or self._config is None:
            return
        for export in exports:
            export_id = export["sample_payload"].get("sample_export_id", -1)
            if export_id < 0:
                continue
            stage_id = uuid.uuid4().hex
            for sequence, status in enumerate(("start", "filtered" if category == "selection" else "failed"), 1):
                self._stage_writer.submit(
                    ExportStageRecord(
                        sample_export_id=export_id,
                        group_id=group_id,
                        run_id=self._config.run_id,
                        rollout_mode=rollout_mode,
                        stage="filter",
                        operation="export_delivery",
                        stage_id=stage_id,
                        event_seq=sequence,
                        status=status,
                        is_terminal=sequence == 2,
                        reason_code=reason_code if sequence == 2 else None,
                        category=category if sequence == 2 else None,
                        reason=reason if sequence == 2 else None,
                    )
                )

    async def aclose(self) -> None:
        """Drain queued records.

        Shards set ``max_restarts=0``; without this the tail of the run is lost
        on shutdown.
        """

        deadline = time.monotonic() + CLOSE_TIMEOUT
        if self._writer is not None:
            await self._writer.aclose(timeout=max(0.0, deadline - time.monotonic()))
            self._writer = None
        if self._stage_writer is not None:
            await self._stage_writer.aclose(timeout=max(0.0, deadline - time.monotonic()))
            self._stage_writer = None
        await close_background(timeout=max(0.0, deadline - time.monotonic()))


def build_trajectory_record(
    *,
    forest: "SessionForest",
    state_hash: str,
    sample_export_id: int,
    metadata: dict[str, Any],
    export_name: Optional[str],
    config: AgenticObsConfig,
    media_sink: Optional[dict[str, str]] = None,
    group_id: Optional[str] = None,
    rollout_mode: Optional[str] = None,
) -> ExportTrajectoryRecord:
    """Snapshot an exported lineage and optional media without file I/O."""

    from relax.agentic.profile import TRACE_KEY

    collect_media = config.dump_media

    lineage_nodes: list[LineageNode] = []
    for node in forest.lineage(state_hash):
        # lineage[0] is the synthetic root created by SessionForest.create_empty:
        # no messages, no tokens, nothing to show.
        if not node.messages_delta and not node.train_token_delta:
            continue
        messages = strip_media(
            node.messages_delta,
            media_sink=media_sink if collect_media else None,
        )
        lineage_nodes.append(
            LineageNode(
                kind=str(node.kind),
                state_hash=node.state_hash,
                token_count=len(node.train_token_delta),
                weight_versions=list(node.weight_version_delta),
                messages=messages,
            )
        )

    agentic_trace = metadata.get(TRACE_KEY) if isinstance(metadata, dict) else None

    return ExportTrajectoryRecord(
        sample_export_id=sample_export_id,
        session_id=forest.session_id,
        export_name=export_name,
        group_index=forest.group_index,
        index=forest.index,
        label=forest.label,
        leaf_state_hash=state_hash,
        group_id=group_id,
        rollout_mode=rollout_mode,
        lineage=lineage_nodes,
        agentic_trace=agentic_trace if isinstance(agentic_trace, dict) else {},
        run_id=config.run_id,
    )


def sample_reward_fields(
    args: Namespace, sample: "Sample", *, rollout_mode: str = "train"
) -> tuple[Optional[float], Optional[dict[str, Any]]]:
    """Split ``Sample.reward`` into the trained scalar and the raw detail.

    ``Sample.reward`` is ``float | dict | None``. With ``--reward-key`` set it
    is a dict and only one key reaches training, so keep both: the projection
    for sorting, the dict for seeing where the reward models disagreed.
    """

    reward = sample.reward
    if reward is None:
        return None, None
    if isinstance(reward, dict):
        detail = dict(reward)
        key = getattr(args, "reward_key", None)
        if rollout_mode == "eval":
            key = getattr(args, "eval_reward_key", None) or key
        value = reward.get(key) if key else None
        return (float(value) if isinstance(value, (int, float)) else None), detail
    if isinstance(reward, (int, float)):
        return float(reward), None
    return None, None


__all__ = [
    "ShardTrajectoryRecorder",
    "build_trajectory_record",
    "resolve_shard_ordinal",
    "sample_reward_fields",
]
