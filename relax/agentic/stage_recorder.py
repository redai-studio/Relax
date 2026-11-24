# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Driver-owned interval boundaries for exported trajectories.

Reward and filter contexts enclose real work; transfer starts before conversion
and ends at the queue acknowledgement. Selection/disposal decisions use filter
with an operation identifying their cause. Each sample carries its event sequence
and terminal flag so later cleanup cannot overwrite a completed decision.

Writes use a bounded daemon queue; cancellation never waits for persistence.
The recorder is disabled unless --dump-agentic-obs names a directory.
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from argparse import Namespace
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable, Literal, Mapping, Optional

from relax.agentic.observability import sample_reward_fields
from relax.utils.agentic_obs import (
    AgenticObsConfig,
    ExportStageRecord,
    JsonlAppender,
    resolve_agentic_obs_config,
)
from relax.utils.agentic_obs.config import write_manifest
from relax.utils.agentic_obs.writers import BoundedWorker, atomic_write_json, best_effort, snapshot_size
from relax.utils.logging_utils import get_logger


if TYPE_CHECKING:
    from relax.agentic.pipeline import GroupExport
    from relax.agentic.pipeline.runtime import RuntimeGroupStream
    from relax.utils.types import Sample


logger = get_logger(__name__)


STAGE_FILTER = "filter"
STAGE_CUSTOM_ADVANTAGE = "custom_advantage"
STAGE_OVERSAMPLING = "oversampling"
STAGE_SHUTDOWN = "shutdown"
STAGE_TRANSFER = "transfer"
STAGE_EVAL = "eval"

STATUS_DROPPED = "failed"
STATUS_QUEUED = "queued_for_train"
STATUS_COMPLETED = "completed"

# Steps are written contiguously, but the transfer domain keeps a previous and a
# current partition open at once, so more than one step can be live. A small
# window keeps the descriptor count flat over a long run instead of growing by
# one file per step until the process hits its ulimit.
_MAX_OPEN_FILES = 4


class StageRecorder:
    """Append stage events, one file per driver step."""

    def __init__(self, args: Namespace, *, rollout_mode: Literal["train", "eval"] = "train") -> None:
        self._args = args
        self._rollout_mode = rollout_mode
        self._config: Optional[AgenticObsConfig] = resolve_agentic_obs_config(args)
        self._appenders: "OrderedDict[int, JsonlAppender]" = OrderedDict()
        self._producer_id = uuid.uuid4().hex
        self._worker = BoundedWorker(on_close=self._close_files)
        self._active: dict[tuple[int, str], tuple[str, dict[str, Any]]] = {}
        if self._config is not None:
            write_manifest(args, self._config)

    @property
    def enabled(self) -> bool:
        return self._config is not None

    def _appender(self, rollout_id: int) -> JsonlAppender:
        """Return the appender for one step, evicting the least recently
        used."""

        appender = self._appenders.get(rollout_id)
        if appender is not None:
            self._appenders.move_to_end(rollout_id)
            return appender
        assert self._config is not None
        appender = JsonlAppender(self._config.stage_path(rollout_id, rollout_mode=self._rollout_mode))
        self._appenders[rollout_id] = appender
        while len(self._appenders) > _MAX_OPEN_FILES:
            _evicted_id, evicted = self._appenders.popitem(last=False)
            evicted.close()
        return appender

    async def record_group_dropped(
        self,
        group: "GroupExport",
        *,
        rollout_id: int,
        stage: str,
        reason: Optional[str] = None,
        category: str = "selection",
        reason_code: Optional[str] = None,
        status: str = STATUS_DROPPED,
    ) -> None:
        """Record one event per Sample of a dropped Group."""

        if self._config is None:
            return
        await self._record_samples(
            group.samples,
            rollout_id=rollout_id,
            group_ids={sample.sample_export_id: group.group_id for sample in group.samples},
            stage=stage,
            status=status,
            reason=reason,
            partition_id=None,
            category=category,
            reason_code=reason_code,
        )

    async def record_groups_dropped(
        self,
        groups: Iterable["GroupExport"],
        *,
        rollout_id: int,
        stage: str,
        reason: Optional[str] = None,
        category: str = "selection",
        reason_code: Optional[str] = None,
        status: str = STATUS_DROPPED,
    ) -> None:
        if self._config is None:
            return
        for group in tuple(groups):
            await self.record_group_dropped(
                group,
                rollout_id=rollout_id,
                stage=stage,
                reason=reason,
                category=category,
                reason_code=reason_code,
                status=status,
            )

    @best_effort
    async def record_stream_dropped(
        self,
        stream: "RuntimeGroupStream",
        *,
        rollout_id: int,
        stage: str,
        reason: Optional[str] = None,
        category: str = "selection",
        reason_code: Optional[str] = None,
        status: str = STATUS_DROPPED,
    ) -> None:
        """Record already exported Sessions after their Group task is
        joined."""

        if self._config is None:
            return
        samples = []
        for result in stream.session_results:
            completion = result.completion
            if not completion.done() or completion.cancelled() or completion.exception() is not None:
                continue
            session = completion.result()
            if session is not None:
                samples.extend(export.sample for export in session.exports)
        await self._record_samples(
            samples,
            rollout_id=rollout_id,
            group_ids={sample.sample_export_id: stream.group_id for sample in samples},
            stage=stage,
            status=status,
            reason=reason,
            partition_id=None,
            category=category,
            reason_code=reason_code,
        )

    @best_effort
    async def record_queued(
        self,
        samples: list["Sample"],
        *,
        rollout_id: int,
        partition_id: int,
        group_ids: Optional[Mapping[int, str]] = None,
    ) -> None:
        """Record a successful TransferQueue write.

        ``rollout_id`` names the file (the driver step); ``partition_id`` is
        the TransferQueue partition, which can differ because the transfer
        domain keeps a previous and a current partition open at the same time.
        """

        if self._config is None:
            return
        await self._record_samples(
            samples,
            rollout_id=rollout_id,
            group_ids=group_ids or {},
            stage=STAGE_TRANSFER,
            status=STATUS_QUEUED,
            reason=None,
            partition_id=partition_id,
        )

    @best_effort
    async def _record_samples(
        self,
        samples: Iterable["Sample"],
        *,
        rollout_id: int,
        group_ids: Mapping[int, str],
        stage: str,
        status: str,
        reason: Optional[str],
        partition_id: Optional[int],
        is_terminal: Optional[bool] = None,
        reason_code: Optional[str] = None,
        category: Optional[str] = None,
        operation: str | None = None,
        transfer_batch_id: str | None = None,
    ) -> None:
        if self._config is None:
            return
        operation = operation or stage
        if status == STATUS_DROPPED:
            reason_code = reason_code or f"{operation}_rejected"
            category = category or "selection"
        if stage not in ("reward", "filter", "transfer"):
            stage = "filter"
            if status == "completed":
                status, is_terminal = "passed", True
        if stage == "filter" and status == "failed" and category == "selection":
            status = "cancelled" if operation == STAGE_SHUTDOWN else "filtered"
        records: list[ExportStageRecord] = []
        for sample in samples:
            export_id = getattr(sample, "sample_export_id", -1)
            if export_id < 0 or getattr(sample, "agentic_terminal", False):
                # Not an agentic export; nothing to join against.
                continue
            key = (export_id, stage)
            active = self._active.get(key)
            if status == "start" and active is not None:
                continue
            if active is None:
                active = (uuid.uuid4().hex, {"operation": operation, "transfer_batch_id": transfer_batch_id})
                self._active[key] = active
                sample.agentic_event_seq = getattr(sample, "agentic_event_seq", 0) + 1
                records.append(
                    ExportStageRecord(
                        export_id,
                        stage,
                        "start",
                        group_ids.get(export_id, ""),
                        run_id=self._config.run_id,
                        stage_id=active[0],
                        event_seq=sample.agentic_event_seq,
                        rollout_id=rollout_id,
                        rollout_mode=self._rollout_mode,
                        partition_id=partition_id,
                        **active[1],
                    )
                )
            if status == "start":
                continue
            self._active.pop(key)
            terminal = status in ("failed", "cancelled", "filtered") if is_terminal is None else is_terminal
            sample.agentic_event_seq += 1
            sample.agentic_terminal = terminal
            reward_value, reward_detail = (None, None)
            if stage != "reward" or status == "completed":
                reward_value, reward_detail = sample_reward_fields(self._args, sample, rollout_mode=self._rollout_mode)
            records.append(
                ExportStageRecord(
                    sample_export_id=export_id,
                    stage=stage,
                    status=status,
                    group_id=group_ids.get(export_id, ""),
                    reason=reason,
                    partition_id=partition_id,
                    reward_value=reward_value,
                    reward_detail=reward_detail,
                    run_id=self._config.run_id,
                    event_seq=sample.agentic_event_seq,
                    stage_id=active[0],
                    is_terminal=terminal,
                    reason_code=reason_code,
                    category=category,
                    rollout_id=rollout_id,
                    rollout_mode=self._rollout_mode,
                    **active[1],
                )
            )
        if not records:
            return
        await self._append_records(rollout_id, records)

    @best_effort
    async def _append_records(self, rollout_id: int, records: list[ExportStageRecord]) -> None:
        payloads = copy.deepcopy([record.to_dict() for record in records])
        self._worker.submit(self._write_records, rollout_id, payloads, size_bytes=snapshot_size(payloads))

    def _write_records(self, rollout_id: int, records: list[dict[str, Any]]) -> None:
        self._appender(rollout_id).append_many(records)

    @best_effort
    async def begin_samples(
        self,
        samples: Iterable["Sample"],
        *,
        rollout_id: int,
        group_ids: Mapping[int, str],
        stage: str,
        partition_id: int | None = None,
        operation: str | None = None,
        transfer_batch_id: str | None = None,
    ) -> None:
        await self._record_samples(
            samples,
            rollout_id=rollout_id,
            group_ids=group_ids,
            stage=stage,
            status="start",
            reason=None,
            partition_id=partition_id,
            operation=operation,
            transfer_batch_id=transfer_batch_id,
        )

    @asynccontextmanager
    async def observe(
        self,
        samples: Iterable["Sample"],
        *,
        rollout_id: int,
        group_id: str,
        stage: str,
        operation: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        samples = tuple(samples)
        fields = dict(
            rollout_id=rollout_id,
            group_ids={s.sample_export_id: group_id for s in samples},
            stage=stage,
            operation=operation,
            partition_id=None,
        )
        result = {"status": "passed" if stage == "filter" else "completed", "reason": None}
        try:
            await self.begin_samples(samples, **fields)
            yield result
        except BaseException as error:
            cancelled = isinstance(error, asyncio.CancelledError)
            await self._record_samples(
                samples,
                **fields,
                status="cancelled" if cancelled else "failed",
                reason=f"{type(error).__name__}: {error}",
                reason_code=f"{operation or stage}_{'cancelled' if cancelled else 'failed'}",
                category="selection" if cancelled else "execution",
                is_terminal=True,
            )
            raise
        else:
            await self._record_samples(samples, **fields, **result)

    @best_effort
    async def record_reward(self, sample: "Sample", *, rollout_id: int, group_id: str) -> None:
        if self._config is None:
            return
        await self._record_samples(
            [sample],
            rollout_id=rollout_id,
            group_ids={sample.sample_export_id: group_id},
            stage="reward",
            status=STATUS_COMPLETED,
            reason=None,
            partition_id=None,
        )

    @best_effort
    async def record_transfer_unknown(
        self,
        groups: Iterable["GroupExport"],
        *,
        rollout_id: int,
        partition_id: int,
        reason: str,
        cancelled: bool = False,
    ) -> None:
        """A failed RPC acknowledgement does not prove that TQ rejected a
        write."""
        if self._config is None:
            return
        groups = tuple(groups)
        await self._record_samples(
            [sample for group in groups for sample in group.samples],
            rollout_id=rollout_id,
            group_ids={sample.sample_export_id: group.group_id for group in groups for sample in group.samples},
            stage=STAGE_TRANSFER,
            status="cancelled" if cancelled else "failed",
            reason=reason,
            partition_id=partition_id,
            is_terminal=False,
            reason_code="transfer_outcome_unknown",
            category="infra",
        )

    @best_effort
    async def record_rollout(
        self,
        rollout_id: int,
        *,
        group_ids: Iterable[str],
        event_seq: int,
        status: str = "start",
        stage_id: str | None = None,
        groups_finalized: bool = False,
        reason: Optional[str] = None,
    ) -> None:
        if self._config is None:
            return
        record = ExportStageRecord(
            sample_export_id=None,
            group_id="",
            run_id=self._config.run_id,
            scope="rollout",
            stage="execution",
            status=status,
            is_terminal=status != "start",
            stage_id=stage_id or f"rollout:{self._rollout_mode}:{rollout_id}:execution",
            event_seq=event_seq,
            rollout_id=rollout_id,
            rollout_mode=self._rollout_mode,
            group_ids=sorted(set(group_ids)),
            groups_finalized=groups_finalized,
            reason=reason,
            reason_code="rollout_failed" if status == "failed" else None,
            category="execution" if status == "failed" else None,
        )
        await self._append_records(rollout_id, [record])

    @best_effort
    async def record_evaluated(self, group: "GroupExport", *, rollout_id: int) -> None:
        """Record the final Eval reward, retaining its dataset-scoped group
        id."""

        if self._config is None:
            return
        await self._record_samples(
            group.samples,
            rollout_id=rollout_id,
            group_ids={sample.sample_export_id: group.group_id for sample in group.samples},
            stage=STAGE_EVAL,
            status=STATUS_COMPLETED,
            is_terminal=True,
            reason=None,
            partition_id=None,
        )

    def close(self) -> None:
        self._worker.close()
        self._active.clear()

    async def aclose(self) -> None:
        await self._worker.aclose()
        self._active.clear()

    def _close_files(self) -> None:
        for appender in self._appenders.values():
            appender.close()
        self._appenders.clear()
        if self._config is not None:
            atomic_write_json(
                self._config.stages_dir / f"producer_{self._producer_id}.status.json",
                {"batches_dropped": self._worker.dropped_count, "write_errors": self._worker.error_count},
            )


__all__ = [
    "STAGE_CUSTOM_ADVANTAGE",
    "STAGE_EVAL",
    "STAGE_FILTER",
    "STAGE_OVERSAMPLING",
    "STAGE_SHUTDOWN",
    "STAGE_TRANSFER",
    "STATUS_DROPPED",
    "STATUS_COMPLETED",
    "STATUS_QUEUED",
    "StageRecorder",
]
