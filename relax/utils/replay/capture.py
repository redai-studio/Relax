# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Production capture: post-forward tensors → replay bundle.

Enabled via maybe_enable_from_env. Hooks live in capture_hooks.
CaptureManager.submit detaches and clones tensors (GPU→GPU copy, no CPU sync);
GPU→CPU copy and serialization run on a bounded writer thread that never back-
pressures training.
"""

from __future__ import annotations

import atexit
import queue
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from relax.utils.replay.bundle import BundleWriter, try_finalize_cohort, write_cohort_shard
from relax.utils.replay.schema import (
    FORMAT_VERSION,
    STAGE_ORDER,
    BundleIndex,
    ComparisonPolicy,
    Identity,
    Manifest,
    ProducerInfo,
    RecomputeConfig,
    SampleRecord,
    StageContract,
    StageId,
)
from relax.utils.replay.stages import REIMPLEMENTED_STAGES, V1_STAGE_VERSIONS, capability_for


logger = get_logger(__name__)


@dataclass
class CaptureConfig:
    """Runtime configuration for production capture."""

    enabled: bool = False
    output_dir: str | Path = ""
    queue_capacity: int = 16
    # Actor steps (rollout_id, step_id) to capture; None captures every step.
    selected_steps: set[tuple[int, int]] | None = None
    # Rollouts to capture at rollout level (reward/advantage); None captures every rollout.
    selected_rollouts: set[int] | None = None
    redaction: dict[str, str] = field(default_factory=dict)
    # Capturing process rank and the last-PP ranks that must publish
    # COMPLETE.<rank> before the shared cohort is finalized. None means this
    # rank only (single-rank bundle, no cross-rank coordinator).
    rank: int = 0
    expected_ranks: list[int] | None = None


@dataclass
class CaptureRecord:
    """Everything one capture cohort needs to be replayed.

    A record is anchored by exactly one of actor_step_id (per-step, loss) or
    rollout_id (per-rollout, reward/advantage). tensors holds the post-forward
    tensor payloads (inputs and expected outputs); expected holds the JSON-
    serializable expected outputs.
    """

    identity: Identity
    samples: list[SampleRecord]
    config: RecomputeConfig
    tensors: dict[str, torch.Tensor]
    expected: dict[str, Any]
    bundle_id: str
    actor_step_id: tuple[int, int] | None = None
    rollout_id: int | None = None
    producer: ProducerInfo = field(default_factory=ProducerInfo)
    redaction: dict[str, str] = field(default_factory=dict)
    # Stages actually captured by this record; None means "derive from the
    # frozen matrix" (declare every matrix stage). A partial capture (e.g. loss
    # only) sets this explicitly so the manifest does not promise stages whose
    # payloads are absent.
    stages: set[StageId] | None = None
    # Raw per-sample metadata used to build SampleRecord on the writer
    # thread. The mask/reward/group tensors stay detached through the hot path;
    # their GPU→CPU conversion happens only in build_bundle_from_record.
    response_lengths: list[int] | None = None
    total_lengths: list[int] | None = None
    loss_masks_tensor: torch.Tensor | None = None
    group_indices_tensor: torch.Tensor | None = None
    raw_rewards_tensor: torch.Tensor | None = None
    rewards_tensor: torch.Tensor | None = None
    # Per-sample DataIterator micro-batch ids (mb-0000, ...). Parallel to
    # response_lengths. captured_micro_batch_ids is the first-writer-wins set
    # used to ignore activation-checkpoint recomputes of the same micro-batch.
    sample_micro_batch_ids: list[str] | None = None
    captured_micro_batch_ids: set[str] = field(default_factory=set)
    capture_rank: int = 0
    expected_ranks: list[int] | None = None

    @property
    def anchor(self) -> str:
        """Human-readable cohort anchor for logging."""
        if self.actor_step_id is not None:
            return f"step{self.actor_step_id}"
        if self.rollout_id is not None:
            return f"rollout-{self.rollout_id}"
        return "<unset>"


def _snapshot_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Detach and clone so later in-place writes cannot mutate the snapshot.

    clone() is a device-local copy (GPU→GPU when the tensor is CUDA) and does
    not synchronize to CPU. GPU→CPU transfer stays on the writer thread.
    """
    return tensor.detach().clone()


def _snapshot_value(value: Any) -> Any:
    """Snapshot tensors in a nested dict/list; leave other values unchanged."""
    if isinstance(value, torch.Tensor):
        return _snapshot_tensor(value)
    if isinstance(value, dict):
        return {key: _snapshot_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_value(item) for item in value]
    return value


def _snapshot_record(record: CaptureRecord) -> CaptureRecord:
    """Immutable snapshot of record tensors (no GPU→CPU sync)."""
    return replace(
        record,
        tensors={name: _snapshot_tensor(tensor) for name, tensor in record.tensors.items()},
        expected=_snapshot_value(record.expected),
        loss_masks_tensor=_snapshot_value(record.loss_masks_tensor),
        group_indices_tensor=_snapshot_value(record.group_indices_tensor),
        raw_rewards_tensor=_snapshot_value(record.raw_rewards_tensor),
        rewards_tensor=_snapshot_value(record.rewards_tensor),
        sample_micro_batch_ids=(
            list(record.sample_micro_batch_ids) if record.sample_micro_batch_ids is not None else None
        ),
        captured_micro_batch_ids=set(record.captured_micro_batch_ids),
        expected_ranks=list(record.expected_ranks) if record.expected_ranks is not None else None,
    )


def build_manifest_for_record(record: CaptureRecord) -> Manifest:
    """Derive the manifest from the frozen V1 capability matrix.

    Each stage in STAGE_ORDER is declared with the capability the matrix grants
    for the bundle's topology (record.config.advantage_estimator and
    record.identity.rank["cp"]). Stages outside the matrix are unsupported and
    will be skipped — never silently recomputed. When record.stages is set,
    only those stages are declared (the rest get no contract and are skipped),
    so a partial capture never promises absent payloads.
    """
    context_parallel = record.identity.rank.get("cp", 1)
    declared = record.stages if record.stages is not None else set(STAGE_ORDER)
    stage_contracts: dict[StageId, StageContract] = {}
    for stage in STAGE_ORDER:
        if stage not in declared:
            continue
        capability = capability_for(
            stage,
            advantage_estimator=record.config.advantage_estimator,
            context_parallel=context_parallel,
        )
        stage_contracts[stage] = StageContract(
            stage=stage,
            version=V1_STAGE_VERSIONS[stage],
            capability=capability,
            implementation="reimplemented" if stage in REIMPLEMENTED_STAGES else "reuse",
        )
    return Manifest(
        format_version=FORMAT_VERSION,
        bundle_id=record.bundle_id,
        producer=record.producer,
        stage_contracts=stage_contracts,
        payloads={},
        comparison_policy=ComparisonPolicy(),
        redaction=dict(record.redaction),
    )


def _normalize_expected(expected: dict[str, Any]) -> dict[str, Any]:
    """Convert detached scalar tensors in expected to plain floats.

    Runs on the writer thread so the GPU→CPU sync (.item()) never happens in
    the training hot path. Handles nested dicts/lists (e.g. loss.policy metric
    maps).
    """

    def _convert(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        if isinstance(value, dict):
            return {key: _convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_convert(item) for item in value]
        return value

    return {key: _convert(value) for key, value in expected.items()}


def _build_samples_from_raw(record: CaptureRecord) -> list[SampleRecord]:
    """Build the SampleRecord list from raw per-sample metadata.

    Runs on the writer thread; the flat loss_masks_tensor is split back per
    sample using response_lengths, and the optional group/reward tensors are
    converted here so their GPU→CPU sync never happens in the hot path. When
    the group/reward tensors are absent (loss-only capture) placeholders are
    used.
    """
    if record.response_lengths is None or record.total_lengths is None or record.loss_masks_tensor is None:
        raise ValueError("capture record has no samples and incomplete raw sample metadata")

    count = len(record.response_lengths)
    flat_masks = record.loss_masks_tensor.detach().cpu().tolist()
    masks: list[list[int]] = []
    offset = 0
    for response_length in record.response_lengths:
        masks.append([int(value) for value in flat_masks[offset : offset + response_length]])
        offset += response_length

    group_indices = (
        [int(value) for value in record.group_indices_tensor.detach().cpu().tolist()]
        if record.group_indices_tensor is not None
        else [0] * count
    )
    raw_rewards = (
        [float(value) for value in record.raw_rewards_tensor.detach().cpu().tolist()]
        if record.raw_rewards_tensor is not None
        else [0.0] * count
    )
    rewards = (
        [float(value) for value in record.rewards_tensor.detach().cpu().tolist()]
        if record.rewards_tensor is not None
        else raw_rewards
    )
    return [
        SampleRecord(
            sample_id=f"s-{index}",
            group_index=group_indices[index],
            response_length=int(record.response_lengths[index]),
            total_length=int(record.total_lengths[index]),
            loss_mask=masks[index],
            raw_reward=raw_rewards[index],
            reward=rewards[index],
            label_hash="",
            micro_batch_id=_sample_micro_batch_id(record, index),
        )
        for index in range(count)
    ]


def format_micro_batch_id(index: int) -> str:
    """Stable DataIterator micro-batch id (mb-0000, mb-0001, ...)."""
    return f"mb-{index:04d}"


def _sample_micro_batch_id(record: CaptureRecord, index: int) -> str | None:
    """Per-sample micro-batch id, or a single-mb fallback for loss-only
    records."""
    if record.sample_micro_batch_ids is not None and index < len(record.sample_micro_batch_ids):
        return record.sample_micro_batch_ids[index]
    return _micro_batch_id_for(record)


def _micro_batch_id_for(record: CaptureRecord) -> str | None:
    """Fallback id when the hook did not stamp per-sample membership.

    Step-level records default to mb-0000 (a single captured micro-batch).
    Rollout-level records leave it unset so --batch stays fail-closed.
    """
    if record.actor_step_id is None:
        return None
    if record.captured_micro_batch_ids:
        return sorted(record.captured_micro_batch_ids)[0]
    return format_micro_batch_id(0)


def _identity_micro_batch_ids(record: CaptureRecord, samples: list[SampleRecord]) -> list[str]:
    """Unique micro-batch ids in first-seen order, matching SampleRecord."""
    seen: list[str] = []
    for sample in samples:
        if sample.micro_batch_id and sample.micro_batch_id not in seen:
            seen.append(sample.micro_batch_id)
    if seen:
        return seen
    fallback = _micro_batch_id_for(record)
    return [fallback] if fallback is not None else []


def _resolved_ranks(record: CaptureRecord) -> tuple[int, list[int]]:
    rank = record.capture_rank
    expected = list(record.expected_ranks) if record.expected_ranks is not None else [rank]
    return rank, expected


def _publish_cohort_and_finalize(
    cohort_dir: Path,
    record: CaptureRecord,
    *,
    relpath: str | None,
    payloads: dict[str, str] | None = None,
    metadata: dict[str, str] | None = None,
) -> None:
    rank, expected = _resolved_ranks(record)
    write_cohort_shard(
        cohort_dir,
        rank=rank,
        identity=record.identity,
        expected_ranks=expected,
        payloads=payloads or {},
        metadata=metadata,
        relpath=relpath,
    )
    try_finalize_cohort(cohort_dir, expected)


def build_bundle_from_record(record: CaptureRecord, output_dir: str | Path) -> Path:
    """Serialize one CaptureRecord into a replay bundle.

    Pure and synchronous; used directly by tests and by the async writer
    thread. A multi-rank capture writes rank-local bundles under
    <dir>/<bundle_id>/rank-<rank>/ and try-finalizes a shared COMPLETE without
    waiting for other ranks.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rank, expected = _resolved_ranks(record)
    multi = len(expected) > 1
    cohort_dir = output_dir / record.bundle_id

    manifest = build_manifest_for_record(record)
    samples = record.samples if record.samples else _build_samples_from_raw(record)
    identity = record.identity
    micro_batch_ids = _identity_micro_batch_ids(record, samples)
    # When samples are reconstructed from raw step metadata, stamp matching
    # identity.micro_batch_ids so --batch selection agrees with SampleRecord.
    if not record.samples and micro_batch_ids:
        identity = replace(identity, micro_batch_ids=micro_batch_ids)
    index = BundleIndex(bundle_id=record.bundle_id, identity=identity, samples=samples, config=record.config)

    expected_outputs = _normalize_expected(record.expected)
    # Derive per-sample reward expectations from the deferred tensors (writer
    # thread), so the reward adapters have JSON-serializable lists to compare.
    if record.raw_rewards_tensor is not None:
        expected_outputs.setdefault(StageId.REWARD_RAW.value, {})["raw_rewards"] = [
            float(value) for value in record.raw_rewards_tensor.detach().cpu().tolist()
        ]
    if record.rewards_tensor is not None:
        expected_outputs.setdefault(StageId.REWARD_POST_PROCESS.value, {})["rewards"] = [
            float(value) for value in record.rewards_tensor.detach().cpu().tolist()
        ]

    rank_path = (cohort_dir / f"rank-{rank}") if multi else cohort_dir
    writer = BundleWriter(rank_path, manifest, index, expected_outputs, rank=rank)
    for name, tensor in record.tensors.items():
        writer.write_payload(name, tensor.detach().cpu().contiguous())
    rank_complete = writer.finalize(ranks=[rank], expected_ranks=expected)
    if multi:
        _publish_cohort_and_finalize(
            cohort_dir,
            record,
            relpath=f"rank-{rank}",
            payloads=dict(rank_complete["payloads"]),
            metadata=dict(rank_complete["metadata"]) if rank_complete.get("metadata") else None,
        )
    return rank_path


class CaptureManager:
    """Bounded, asynchronous capture sink.

    Owns the writer thread and the bounded queue. Disabled by default: when
    config.enabled is False, submit is a no-op that never inspects tensors, so
    the disabled path is a single branch.
    """

    def __init__(self, config: CaptureConfig) -> None:
        self._config = config
        self._dropped_count = 0
        self._error_count = 0
        self._queue: queue.Queue[CaptureRecord | None] | None = None
        self._thread: threading.Thread | None = None
        if config.enabled:
            self._queue = queue.Queue(maxsize=config.queue_capacity)
            self._thread = threading.Thread(target=self._writer_loop, name="replay-capture-writer", daemon=True)
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def output_dir(self) -> str:
        return str(self._config.output_dir)

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def rank(self) -> int:
        return self._config.rank

    @property
    def expected_ranks(self) -> list[int] | None:
        return self._config.expected_ranks

    def is_selected(self, actor_step_id: tuple[int, int]) -> bool:
        if self._config.selected_steps is None:
            return True
        return actor_step_id in self._config.selected_steps

    def is_rollout_selected(self, rollout_id: int) -> bool:
        if self._config.selected_rollouts is None:
            return True
        return rollout_id in self._config.selected_rollouts

    def _record_selected(self, record: CaptureRecord) -> bool:
        if record.actor_step_id is not None:
            return self.is_selected(record.actor_step_id)
        if record.rollout_id is not None:
            return self.is_rollout_selected(record.rollout_id)
        return False

    def _enqueue(self, record: CaptureRecord, *, drop_label: str) -> bool:
        assert self._queue is not None
        try:
            self._queue.put_nowait(record)
            return True
        except queue.Full:
            self._dropped_count += 1
            logger.warning(
                "replay capture queue full (capacity=%d); dropping %s (dropped=%d)",
                self._config.queue_capacity,
                drop_label,
                self._dropped_count,
            )
            return False

    def submit(self, record: CaptureRecord) -> bool:
        """Enqueue a cloned snapshot of record; returns True if queued.

        detach()+clone() happens on the caller thread (no GPU→CPU sync, no
        collective). On queue overflow the record is dropped and False
        returned.
        """
        if not self.enabled or not self._record_selected(record):
            return False
        snapshot = _snapshot_record(record)
        snapshot.capture_rank = self._config.rank
        snapshot.expected_ranks = (
            list(self._config.expected_ranks) if self._config.expected_ranks is not None else None
        )
        return self._enqueue(snapshot, drop_label=record.anchor)

    def _writer_loop(self) -> None:
        assert self._queue is not None
        while True:
            record = self._queue.get()
            try:
                if record is None:  # shutdown sentinel
                    return
                build_bundle_from_record(record, self._config.output_dir)
            except Exception as exc:  # noqa: BLE001 — writer failure must never crash training
                self._error_count += 1
                logger.error("replay capture failed for %s: %s", record.anchor, exc)
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        """Block until queued records have been written."""
        if self._queue is not None:
            self._queue.join()

    def close(self, timeout: float | None = 10.0) -> None:
        """Signal the writer thread to stop and join it.

        The shutdown sentinel is enqueued with the same timeout as the join, so
        a stuck writer cannot hang process exit (atexit / actor teardown).
        """
        if self._thread is None or self._queue is None:
            return
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            logger.warning("replay capture writer did not accept shutdown sentinel")
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("replay capture writer still running after shutdown timeout; abandoning remaining records")
        self._thread = None


# ---------------------------------------------------------------------------
# Process-global capture state (the thin, hot-path-safe hook surface).
#
# Training files call the tiny functions below; everything heavier lives on the
# CaptureManager / writer thread. active() is a single global read and
# current_step() a single global read — no GPU→CPU sync, no collective.
# ---------------------------------------------------------------------------

_active: CaptureManager | None = None
_current_step: CaptureRecord | None = None
_atexit_registered: bool = False


def _parse_selected_steps(raw: str | None) -> set[tuple[int, int]] | None:
    """Parse RELAX_REPLAY_CAPTURE_STEPS (ROLLOUT_ID:STEP_ID,...)."""
    if raw is None or not raw.strip():
        return None
    steps: set[tuple[int, int]] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            rollout_id, step_id = token.split(":")
            steps.add((int(rollout_id), int(step_id)))
        except ValueError as exc:
            raise ValueError(
                f"invalid RELAX_REPLAY_CAPTURE_STEPS item {token!r} (expected ROLLOUT_ID:STEP_ID)"
            ) from exc
    return steps or None


def _parse_selected_rollouts(raw: str | None) -> set[int] | None:
    """Parse RELAX_REPLAY_CAPTURE_ROLLOUTS (ID,ID,...)."""
    if raw is None or not raw.strip():
        return None
    rollouts: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            rollouts.add(int(token))
        except ValueError as exc:
            raise ValueError(f"invalid RELAX_REPLAY_CAPTURE_ROLLOUTS item {token!r} (expected int)") from exc
    return rollouts or None


def config_from_env() -> CaptureConfig | None:
    """Build a CaptureConfig from RELAX_REPLAY_CAPTURE* env vars.

    Returns None when capture is unset/false, so the caller can no-op.
    RELAX_REPLAY_CAPTURE=1 without RELAX_REPLAY_CAPTURE_DIR logs a warning and
    also returns None — never write to an implicit path.
    """
    if not Envs.RELAX_REPLAY_CAPTURE:
        return None
    output_dir = Envs.RELAX_REPLAY_CAPTURE_DIR
    if not output_dir:
        logger.warning("RELAX_REPLAY_CAPTURE is set but RELAX_REPLAY_CAPTURE_DIR is empty; capture stays disabled")
        return None
    return CaptureConfig(
        enabled=True,
        output_dir=output_dir,
        selected_steps=_parse_selected_steps(Envs.RELAX_REPLAY_CAPTURE_STEPS),
        selected_rollouts=_parse_selected_rollouts(Envs.RELAX_REPLAY_CAPTURE_ROLLOUTS),
    )


def maybe_enable_from_env(
    *, rank: int | None = None, expected_ranks: list[int] | None = None
) -> CaptureManager | None:
    """Enable capture from env vars. No-op when capture is not requested.

    rank is this process. expected_ranks is the last-PP producer set that must
    publish COMPLETE.<rank>; None means this rank only. Output stays under
    RELAX_REPLAY_CAPTURE_DIR; multi-rank writers coordinate via
    try_finalize_cohort.
    """
    config = config_from_env()
    if config is None:
        return None
    if rank is not None:
        config = replace(
            config,
            rank=rank,
            expected_ranks=list(expected_ranks) if expected_ranks is not None else None,
        )
    logger.info("enabling trajectory-replay capture under %s", config.output_dir)
    return enable(config)


def enable(config: CaptureConfig) -> CaptureManager:
    """Enable capture process-wide and return the active manager.

    Replaces any previously active manager (flushing it first). This is the
    Python-API injection point: the training actor calls it from
    maybe_enable_for_actor — no argument-parsing changes required.
    """
    global _active, _atexit_registered
    if _active is not None:
        _active.close()
    _active = CaptureManager(config)
    if not _atexit_registered:
        atexit.register(disable)
        _atexit_registered = True
    return _active


def disable() -> None:
    """Disable capture process-wide and flush the active manager."""
    global _active
    if _active is not None:
        _active.close()
        _active = None


def active() -> CaptureManager | None:
    """Return the active manager, or None when capture is disabled."""
    return _active


def _open_cohort(
    *,
    selected: bool,
    identity: Identity,
    config: RecomputeConfig,
    bundle_id: str,
    actor_step_id: tuple[int, int] | None = None,
    rollout_id: int | None = None,
    producer: ProducerInfo | None = None,
) -> None:
    global _current_step
    if not selected:
        _current_step = None
        return
    _current_step = CaptureRecord(
        identity=identity,
        samples=[],
        config=config,
        tensors={},
        expected={},
        bundle_id=bundle_id,
        actor_step_id=actor_step_id,
        rollout_id=rollout_id,
        producer=producer if producer is not None else ProducerInfo(),
    )


def begin_step(
    actor_step_id: tuple[int, int],
    *,
    identity: Identity,
    config: RecomputeConfig,
    bundle_id: str,
    producer: ProducerInfo | None = None,
) -> None:
    """Open a per-step capture accumulator (no-op when disabled/unselected)."""
    _open_cohort(
        selected=_active is not None and _active.is_selected(actor_step_id),
        identity=identity,
        config=config,
        bundle_id=bundle_id,
        actor_step_id=actor_step_id,
        producer=producer,
    )


def begin_rollout(
    rollout_id: int,
    *,
    identity: Identity,
    config: RecomputeConfig,
    bundle_id: str,
    producer: ProducerInfo | None = None,
) -> None:
    """Open a per-rollout capture accumulator (no-op when
    disabled/unselected)."""
    _open_cohort(
        selected=_active is not None and _active.is_rollout_selected(rollout_id),
        identity=identity,
        config=config,
        bundle_id=bundle_id,
        rollout_id=rollout_id,
        producer=producer,
    )


def should_capture(actor_step_id: tuple[int, int]) -> bool:
    """Cheap disabled/unselected guard (two global reads, no allocation)."""
    return _active is not None and _active.is_selected(actor_step_id)


def should_capture_rollout(rollout_id: int) -> bool:
    """Cheap disabled/unselected guard for the rollout-level hot path."""
    return _active is not None and _active.is_rollout_selected(rollout_id)


def current_step() -> CaptureRecord | None:
    """Return the current cohort's accumulator, or None when disabled."""
    return _current_step


def end_step() -> None:
    """Submit the current accumulator, or drop it if no hook filled stages."""
    global _current_step
    step = _current_step
    _current_step = None
    if step is None or _active is None or not step.stages:
        return
    _active.submit(step)


def end_rollout() -> None:
    end_step()
