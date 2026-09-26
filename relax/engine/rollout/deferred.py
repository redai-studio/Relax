# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Deferred scoring: move teacher prefill out of generation, gate publication.

The immediate path scores each sample inline while it is generated, which keeps
the teacher resident for the whole generation. Deferred scoring instead stages
whole batches, scores them once generation has finished, validates the result and
only then publishes -- so the same GPUs can run student, then teacher, then
trainer, one at a time.

Two rules shape everything here:

* Nothing half-scored is publishable. A batch either has every required training
  field for every eligible sample, or it fails. Publishing the successful part
  would put rows in the queue that the loss treats as complete distillation
  targets when they are not.
* Staging happens while generation is still in flight, scoring does not. The
  student cannot be offloaded for the teacher until the last request is done,
  which is why batches are staged during the step and flushed after it.

This executor belongs to the rollout workload: the Manager owns models and
switches the shared GPUs between phases, and this owns samples and the queue
handoff.
"""

import asyncio
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Sequence

from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


logger = get_logger(__name__)


class DeferredState(str, Enum):
    """A deferred batch goes staged, scoring, validating, publishing and ends
    completed or failed."""

    STAGED = "staged"
    SCORING = "scoring"
    VALIDATING = "validating"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class SampleRef:
    """The sealed description of one sample's scoring contract.

    Sealed at submit time and kept until a terminal state: it is what results
    are correlated against, so a late or duplicated response cannot be mistaken
    for the sample it claims to be.
    """

    sample_index: Any
    group_index: int | None
    response_length: int
    prompt_length: int
    route_key: str | None = None
    has_multimodal: bool = False
    # Set when ``sample_index`` is not unique in the batch, e.g. the several
    # exports of one Agentic Session that keep their training index.
    scoring_id: Any = None

    @property
    def eligible(self) -> bool:
        """An empty response has nothing to score and no fields to validate."""
        return self.response_length > 0

    @property
    def identity(self) -> Any:
        """The key results are correlated against within the batch."""
        return self.sample_index if self.scoring_id is None else self.scoring_id


@dataclass(frozen=True)
class BatchRef:
    """A batch as sealed at submit time."""

    batch_id: str
    rollout_id: int
    policy_version: str | None
    token_selection: str
    required_fields: tuple[str, ...]
    samples: tuple[SampleRef, ...]

    @property
    def eligible_count(self) -> int:
        return sum(1 for sample in self.samples if sample.eligible)


@dataclass(frozen=True)
class DeferredHandle:
    operation_id: str
    batch_id: str


@dataclass(frozen=True)
class DeferredSnapshot:
    operation_id: str
    batch_id: str
    state: DeferredState
    scored: int = 0
    published: int = 0
    missing: tuple[Any, ...] = ()
    error: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in (DeferredState.COMPLETED, DeferredState.FAILED)


@dataclass
class _Record:
    handle: DeferredHandle
    batch_ref: BatchRef
    plan_id: str | None
    # Flat samples for scoring and validation, and the caller's original nesting
    # for publication: the queue helper derives its ordering from the groups.
    samples: list[Sample]
    payload: Any
    is_last: bool
    snapshot: DeferredSnapshot
    task: asyncio.Task | None = None


def _leading_length(value: Any) -> int | None:
    """Rows in a scoring field, for list, tuple and array-like values."""
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        return int(shape[0]) if len(shape) else 0
    try:
        return len(value)
    except TypeError:
        return None


def validate_scored_batch(batch_ref: BatchRef, samples: Sequence[Sample]) -> tuple[tuple[Any, ...], list[str]]:
    """Check the scored batch against the sealed reference.

    Returns the sample identities that are not publishable and the reasons. It
    verifies presence, order, per-field row counts against the response length,
    and that the loss mask still matches -- a field that is one row short would
    otherwise silently misalign every token after it.
    """
    problems: list[str] = []
    missing: list[Any] = []
    if len(samples) != len(batch_ref.samples):
        problems.append(f"batch size changed: sealed {len(batch_ref.samples)}, scored {len(samples)}")
        return tuple(ref.identity for ref in batch_ref.samples), problems

    seen: set[Any] = set()
    seen_objects: set[int] = set()
    for ref, sample in zip(batch_ref.samples, samples, strict=True):
        identity = ref.identity
        if identity in seen or id(sample) in seen_objects:
            problems.append(f"sample {identity} appears twice in the scored batch")
            missing.append(identity)
            continue
        seen.add(identity)
        seen_objects.add(id(sample))
        if sample.index != ref.sample_index:
            problems.append(f"sample order changed: sealed {ref.sample_index}, scored {sample.index}")
            missing.append(identity)
            continue
        response_length = int(sample.response_length or 0)
        if response_length != ref.response_length:
            problems.append(
                f"sample {identity} response length changed: sealed {ref.response_length}, now {response_length}"
            )
            missing.append(identity)
            continue
        if not ref.eligible:
            continue
        mask = getattr(sample, "loss_mask", None)
        if mask is not None and len(mask) != response_length:
            problems.append(f"sample {identity} loss mask covers {len(mask)} of {response_length} tokens")
            missing.append(identity)
            continue
        failed_field = None
        for name in batch_ref.required_fields:
            rows = _leading_length(getattr(sample, name, None))
            if rows is None:
                failed_field = f"sample {identity} is missing {name}"
                break
            if rows != response_length:
                failed_field = f"sample {identity} field {name} has {rows} rows, expected {response_length}"
                break
        if failed_field is not None:
            problems.append(failed_field)
            missing.append(identity)
    return tuple(missing), problems


def seal_batch(
    args: Any,
    samples: Sequence[Sample],
    *,
    batch_id: str,
    rollout_id: int,
    required_fields: Sequence[str],
    policy_version: str | None = None,
    scoring_ids: Sequence[Any] | None = None,
) -> BatchRef:
    """Seal what the scorer must produce for this batch.

    ``scoring_ids`` replaces ``sample.index`` as the per-batch identity when
    several samples legitimately share a training index.
    """
    if scoring_ids is not None and (len(scoring_ids) != len(samples) or len(set(scoring_ids)) != len(scoring_ids)):
        raise ValueError(f"Batch {batch_id} needs one unique scoring ID per sample")
    route_key_field = getattr(args, "opd_teacher_key", None) or "data_source"
    refs = []
    for position, sample in enumerate(samples):
        metadata = getattr(sample, "metadata", None) or {}
        multimodal = getattr(sample, "multimodal_inputs", None)
        response_length = int(sample.response_length or 0)
        refs.append(
            SampleRef(
                sample_index=sample.index,
                group_index=getattr(sample, "group_index", None),
                response_length=response_length,
                prompt_length=max(len(sample.tokens or ()) - response_length, 0),
                route_key=metadata.get(route_key_field) if isinstance(metadata, dict) else None,
                has_multimodal=bool(multimodal),
                scoring_id=None if scoring_ids is None else scoring_ids[position],
            )
        )
    return BatchRef(
        batch_id=batch_id,
        rollout_id=rollout_id,
        policy_version=policy_version,
        token_selection=str(getattr(args, "opd_token_selection", "")),
        required_fields=tuple(required_fields),
        samples=tuple(refs),
    )


class DeferredExecutor:
    """Run one deferred batch at a time, and never publish an incomplete
    one."""

    def __init__(self, args: Any) -> None:
        self.args = args
        self._records: dict[str, _Record] = {}
        # One batch at a time: the first version deliberately has no cross-batch
        # pipeline, because two batches would contend for the scoring phase.
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public contract.
    # ------------------------------------------------------------------
    def submit_deferred(
        self,
        batch_ref: BatchRef,
        plan_id: str | None,
        *,
        operation_id: str,
        samples: Sequence[Sample],
        payload: Any = None,
        is_last: bool = False,
    ) -> DeferredHandle:
        """Register a staged batch.

        This does not make it trainable.
        """
        if not operation_id:
            raise ValueError("A deferred operation ID is required")
        existing = self._records.get(operation_id)
        if existing is not None:
            if existing.batch_ref != batch_ref or existing.plan_id != plan_id:
                raise ValueError(f"Deferred operation {operation_id} was submitted with different inputs")
            return existing.handle
        handle = DeferredHandle(operation_id, batch_ref.batch_id)
        self._records[operation_id] = _Record(
            handle=handle,
            batch_ref=batch_ref,
            plan_id=plan_id,
            samples=list(samples),
            payload=payload if payload is not None else list(samples),
            is_last=is_last,
            snapshot=DeferredSnapshot(operation_id, batch_ref.batch_id, DeferredState.STAGED),
        )
        return handle

    async def wait_deferred(self, handle: DeferredHandle, *, timeout_s: float | None = None) -> DeferredSnapshot:
        """Wait for a terminal state; a timeout ends the wait, not the
        batch."""
        record = self._records[handle.operation_id]
        if record.task is None:
            return record.snapshot
        try:
            await asyncio.wait_for(asyncio.shield(record.task), timeout=timeout_s)
        except asyncio.TimeoutError:
            return record.snapshot
        return record.snapshot

    # ------------------------------------------------------------------
    # Execution.
    # ------------------------------------------------------------------
    def start(
        self,
        handle: DeferredHandle,
        *,
        score: Callable[[list[Sample]], Any],
        publish: Callable[[Any, bool], Any],
    ) -> asyncio.Task:
        record = self._records[handle.operation_id]
        if record.task is None:
            record.task = asyncio.create_task(self._run(record, score, publish))
        return record.task

    async def cancel(self) -> None:
        """Cancel every unfinished batch and wait until its task has stopped.

        ``wait_deferred`` shields the task, so cancelling a waiter alone would
        leave the batch scoring and publishing on its own.
        """
        records = [record for record in self._records.values() if record.task is not None and not record.task.done()]
        for record in records:
            record.task.cancel()
        await asyncio.gather(*(record.task for record in records), return_exceptions=True)
        for record in records:
            if not record.snapshot.terminal:
                self._fail(record, "cancelled")

    def _advance(self, record: _Record, state: DeferredState, **changes: Any) -> None:
        record.snapshot = replace(record.snapshot, state=state, **changes)

    def _fail(self, record: _Record, message: str, **changes: Any) -> None:
        record.snapshot = replace(record.snapshot, state=DeferredState.FAILED, error=message, **changes)

    async def _run(
        self,
        record: _Record,
        score: Callable[[list[Sample]], Any],
        publish: Callable[[Any, bool], Any],
    ) -> DeferredSnapshot:
        async with self._lock:
            try:
                self._advance(record, DeferredState.SCORING)
                failures = await score(record.samples)
            except Exception as exc:
                self._fail(record, f"{type(exc).__name__}: {exc}")
                logger.exception(f"Deferred scoring failed for batch {record.batch_ref.batch_id}")
                return record.snapshot
            self._advance(record, DeferredState.VALIDATING)
            missing, problems = validate_scored_batch(record.batch_ref, record.samples)
            unscored = tuple(dict.fromkeys(tuple(failures or ()) + missing))
            if unscored or problems:
                # Successful samples are kept in the record for diagnosis, but
                # the batch does not publish: a partially scored batch would
                # train rows whose distillation targets are absent.
                detail = "; ".join(problems[:5]) if problems else f"{len(unscored)} sample(s) were not scored"
                self._fail(
                    record,
                    f"Deferred scoring is incomplete: {detail}",
                    scored=record.batch_ref.eligible_count - len(unscored),
                    missing=unscored,
                )
                logger.error(
                    f"Deferred batch {record.batch_ref.batch_id} not published: "
                    f"{len(unscored)} of {record.batch_ref.eligible_count} samples unscored; {problems[:5]}"
                )
                return record.snapshot
            self._advance(record, DeferredState.PUBLISHING, scored=record.batch_ref.eligible_count)
            try:
                await publish(record.payload, record.is_last)
            except Exception as exc:
                self._fail(record, f"publication failed: {type(exc).__name__}: {exc}")
                return record.snapshot
            self._advance(record, DeferredState.COMPLETED, published=len(record.samples))
            return record.snapshot


__all__ = [
    "BatchRef",
    "DeferredExecutor",
    "DeferredHandle",
    "DeferredSnapshot",
    "DeferredState",
    "SampleRef",
    "seal_batch",
    "validate_scored_batch",
]
