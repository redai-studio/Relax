# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Capture real policy-update inputs before packing; serialize off-thread.

CUDA snapshots use asynchronous copies. Only the writer waits for them. DP
owners emit local outcomes; readers merge membership without training
collectives.
"""

from __future__ import annotations

import asyncio
import copy
import threading
import time
import uuid
from argparse import Namespace
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from relax.utils.agentic_obs.config import AgenticObsConfig, resolve_agentic_obs_config, write_manifest
from relax.utils.agentic_obs.records import ExportStageRecord, TrainRecord, decode_group_ids
from relax.utils.agentic_obs.writers import (
    CLOSE_TIMEOUT,
    BoundedWorker,
    JsonlAppender,
    atomic_write_json,
    best_effort,
    snapshot_size,
)
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_MAX_UPDATE_BYTES = 256 * 1024 * 1024
_PENDING = threading.BoundedSemaphore(4)
_EXECUTOR: BoundedWorker | None = None
_METADATA_EXECUTOR: BoundedWorker | None = None
_STAGE_EXECUTOR: BoundedWorker | None = None
_UPDATE_EXECUTOR_LOCK = threading.Lock()
_STAGE_EXECUTOR_LOCK = threading.Lock()
_METADATA_PENDING = threading.BoundedSemaphore(128)
_UPDATE_SEQUENCES: dict[tuple[str, str | None, int], int] = {}
_PRODUCER_STATUS: dict[tuple[str, int], dict[str, int]] = {}


@dataclass
class _Snapshot:
    identities: list[tuple[int, str, int, int]]
    values: Any
    event: Any
    source: Any
    advantage_available: bool
    returns_available: bool


@dataclass
class _Update:
    args: Namespace
    config: AgenticObsConfig
    rollout_id: int
    update_id: str
    dp_rank: int
    dp_size: int
    cp_size: int
    step_id: int = 0
    train_generation_id: str | None = None
    update_sequence: int = 0
    stage_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    snapshots: list[_Snapshot] = field(default_factory=list)
    group_ids: set[str] = field(default_factory=set)
    capture_ok: bool = True
    membership_ok: bool = True
    size_bytes: int = 0
    stage: ExportStageRecord | None = None
    reserved: bool = False
    metadata_reserved: bool = False


_CURRENT: ContextVar[_Update | None] = ContextVar("agentic_obs_update", default=None)


def _cpu_values(value: Any) -> list:
    import torch

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("observability identities and lengths must stay on CPU")
        return value.tolist()
    result = []
    for item in value:
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu":
                raise ValueError("observability identities and lengths must stay on CPU")
            item = item.tolist()
        result.append(item)
    return result


@best_effort
def _begin_update(args: Namespace, rollout_id: int, step_id: int, role: str) -> _Update | None:
    config = resolve_agentic_obs_config(args)
    if config is None or role != "actor":
        return None
    from megatron.core import mpu

    if not (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.is_pipeline_last_stage(ignore_virtual=True)
        and mpu.get_context_parallel_rank() == 0
    ):
        return None
    rank = mpu.get_data_parallel_rank(with_context_parallel=False)
    generation = getattr(args, "agentic_obs_train_generation_id", None)
    key = (str(config.root), generation, rank)
    sequence = _UPDATE_SEQUENCES.get(key, 0)
    _UPDATE_SEQUENCES[key] = sequence + 1
    state = _Update(
        args=args,
        config=config,
        rollout_id=rollout_id,
        step_id=step_id,
        update_id=f"actor:{rollout_id}:{step_id}",
        dp_rank=rank,
        dp_size=mpu.get_data_parallel_world_size(with_context_parallel=False),
        cp_size=1 if getattr(args, "dynamic_context_parallel", False) else mpu.get_context_parallel_world_size(),
        train_generation_id=generation,
        update_sequence=sequence,
    )
    write_manifest(args, config)
    start = _stage_record(state, "start")
    state.reserved = _PENDING.acquire(blocking=False)
    state.capture_ok = state.reserved
    _submit_stage(state, start)
    return state


def _stage_record(state: _Update, status: str, **fields: Any) -> ExportStageRecord:
    return ExportStageRecord(
        None,
        "train",
        status,
        "",
        run_id=state.config.run_id,
        scope="update_fragment",
        stage_id=state.stage_id,
        update_id=state.update_id,
        rollout_id=state.rollout_id,
        step_id=state.step_id,
        train_generation_id=state.train_generation_id,
        update_sequence=state.update_sequence,
        dp_rank=state.dp_rank,
        dp_world_size=state.dp_size,
        event_seq=1 if status == "start" else 2,
        is_terminal=status != "start",
        group_ids=sorted(state.group_ids),
        groups_finalized=status != "start" and state.membership_ok,
        input_complete=status != "start" and state.capture_ok,
        **fields,
    )


@best_effort
def _record_failure(state: _Update, error: BaseException) -> None:
    state.capture_ok = False
    if state.stage is None:
        state.membership_ok = False
        state.stage = _stage_record(
            state,
            "cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
            reason_code="train_step_failed",
            reason=f"{type(error).__name__}: {error}",
            category="training",
        )
        _submit_stage(state, state.stage)


@contextmanager
def observe_train_step(args: Namespace, rollout_id: int, step_id: int, *, role: str = "actor") -> Iterator[None]:
    state = _begin_update(args, rollout_id, step_id, role)
    if state is None:
        yield
        return
    token = _CURRENT.set(state)
    try:
        yield
    except BaseException as error:
        _record_failure(state, error)
        raise
    finally:
        _CURRENT.reset(token)
        _submit_update(state)


@best_effort
def capture_training_input(batch: Any) -> None:
    state = _CURRENT.get()
    if state is None or batch.get("__is_dummy__", False):
        return
    from megatron.core import mpu

    if not mpu.is_pipeline_last_stage():
        return
    try:
        _capture(state, batch)
    except Exception as error:  # noqa: BLE001
        state.capture_ok = state.membership_ok = False
        logger.warning("agentic_obs: update %s input capture failed: %s", state.update_id, error)


def _capture(state: _Update, batch: Any) -> None:
    import torch

    ids = _cpu_values(batch.get("sample_export_ids", []))
    if not ids:
        raise ValueError("missing sample_export_ids at the training input")
    groups = decode_group_ids(batch.get("agentic_group_ids", []))
    responses = _cpu_values(batch["response_lengths"])
    totals = _cpu_values(batch["total_lengths"])
    masks = batch["loss_masks"]
    if not (len(ids) == len(groups) == len(responses) == len(totals) == len(masks)):
        raise ValueError("training input columns disagree")
    identities = []
    for export_id, group, response, total in zip(ids, groups, responses, totals):
        if export_id >= 0:
            if not group:
                raise ValueError("missing runtime group identity")
            state.group_ids.add(group)
        else:
            state.capture_ok = state.membership_ok = False
        identities.append((int(export_id), group, int(response), int(total)))
    if not state.reserved:
        return
    size_bytes = (sum(int(mask.numel()) for mask in masks) + 3 * len(ids)) * 4
    if state.size_bytes + size_bytes > _MAX_UPDATE_BYTES:
        state.capture_ok = False
        return
    advantages, returns = batch.get("advantages"), batch.get("returns")
    adv_ok = state.cp_size == 1 and advantages is not None and len(advantages) == len(ids)
    ret_ok = state.cp_size == 1 and returns is not None and len(returns) == len(ids)
    scalars, flat_masks = [], []
    for index, mask in enumerate(masks):
        mask = mask.detach().float().reshape(-1)
        if mask.numel() != responses[index]:
            raise ValueError("loss_mask is not in response-token space")
        flat_masks.append(mask)
        denominator = mask.sum().clamp(min=1)
        zero = mask.new_zeros(())
        weighted = advantages[index].detach() * mask if adv_ok else None
        scalars.extend(
            (
                weighted.sum() / denominator if adv_ok else zero,
                weighted.abs().max() if adv_ok and weighted.numel() else zero,
                (returns[index].detach() * mask).sum() / denominator if ret_ok else zero,
            )
        )
    source = torch.cat([torch.stack(scalars), *flat_masks]).detach()
    event = None
    if source.is_cuda:
        values = torch.empty_like(source, device="cpu", pin_memory=True)
        # Same-stream copy precedes subsequent training mutations. Only the
        # writer synchronizes the event; source owns independent storage.
        values.copy_(source, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(source.device))
    else:
        values = source.clone()
    state.snapshots.append(_Snapshot(identities, values, event, source, adv_ok, ret_ok))
    state.size_bytes += size_bytes


@best_effort
def record_optimizer_result(successful: bool) -> None:
    state = _CURRENT.get()
    if state is None:
        return
    state.stage = _stage_record(
        state,
        "completed" if successful else "failed",
        optimizer_successful=bool(successful),
        reason_code=None if successful else "optimizer_step_skipped",
        category=None if successful else "training",
    )
    # Small local facts have a separate bounded queue from tensor snapshots.
    _submit_stage(state, state.stage)


def _write_update(state: _Update) -> None:
    try:
        samples = []
        for snapshot in state.snapshots:
            if snapshot.event is not None:
                snapshot.event.synchronize()
            snapshot.source = None
            values = snapshot.values.tolist()
            cursor = 3 * len(snapshot.identities)
            for index, (export_id, group, response, total) in enumerate(snapshot.identities):
                mask = [int(value) for value in values[cursor : cursor + response]]
                cursor += response
                if export_id < 0:
                    continue
                samples.append(
                    {
                        "sample_export_id": export_id,
                        "group_id": group,
                        "dp_rank": state.dp_rank,
                        "effective_tokens": int(sum(mask)),
                        "response_length": response,
                        "total_length": total,
                        "advantage_mean": values[3 * index] if snapshot.advantage_available else None,
                        "advantage_absmax": values[3 * index + 1] if snapshot.advantage_available else None,
                        "returns_mean": values[3 * index + 2] if snapshot.returns_available else None,
                    }
                )
        path = state.config.train_scalar_path(state.rollout_id, state.dp_rank)
        if state.metadata_reserved:
            path = path.with_name(f"overflow_{path.name}")
        appender = JsonlAppender(path)
        appender.append(
            TrainRecord(
                run_id=state.config.run_id,
                update_id=state.update_id,
                stage_id=state.stage_id,
                train_generation_id=state.train_generation_id,
                samples=samples,
                dp_rank=state.dp_rank,
                dp_world_size=state.dp_size,
                input_complete=state.capture_ok,
            )
        )
        appender.close()
    except Exception as error:  # noqa: BLE001
        key = (str(state.config.root), state.dp_rank)
        status = _PRODUCER_STATUS.setdefault(key, {"updates_dropped": 0})
        status["write_errors"] = status.get("write_errors", 0) + 1
        logger.warning("agentic_obs: update %s write failed: %s", state.update_id, error)
    finally:
        try:
            _write_producer_status(state)
        finally:
            if state.reserved:
                _PENDING.release()
            if state.metadata_reserved:
                _METADATA_PENDING.release()


@best_effort
def _submit_update(state: _Update) -> None:
    global _EXECUTOR, _METADATA_EXECUTOR
    accepted = False
    try:
        with _UPDATE_EXECUTOR_LOCK:
            if state.reserved:
                if _EXECUTOR is None:
                    _EXECUTOR = BoundedWorker(capacity=4, max_bytes=8 * _MAX_UPDATE_BYTES)
                accepted = _EXECUTOR.submit(
                    _write_update,
                    state,
                    size_bytes=2 * state.size_bytes + snapshot_size(state.group_ids),
                    on_discard=lambda: _discard_update(state),
                )
            else:
                state.metadata_reserved = _METADATA_PENDING.acquire(blocking=False)
                if state.metadata_reserved:
                    if _METADATA_EXECUTOR is None:
                        _METADATA_EXECUTOR = BoundedWorker(capacity=128)
                    accepted = _METADATA_EXECUTOR.submit(
                        _write_update,
                        state,
                        size_bytes=snapshot_size(state.group_ids),
                        on_discard=lambda: _discard_update(state),
                    )
    finally:
        if not accepted:
            if state.reserved:
                _PENDING.release()
            if state.metadata_reserved:
                _METADATA_PENDING.release()
    if not accepted or not state.reserved:
        status = _PRODUCER_STATUS.setdefault((str(state.config.root), state.dp_rank), {"updates_dropped": 0})
        status["updates_dropped"] += 1


def _discard_update(state: _Update) -> None:
    status = _PRODUCER_STATUS.setdefault((str(state.config.root), state.dp_rank), {"updates_dropped": 0})
    key = "updates_dropped" if state.reserved else "events_dropped"
    status[key] = status.get(key, 0) + 1
    if state.reserved:
        _PENDING.release()
    if state.metadata_reserved:
        _METADATA_PENDING.release()


@best_effort
def _submit_stage(state: _Update, record: ExportStageRecord) -> None:
    """Snapshot local boundaries independently of the tensor-copy backlog."""
    global _STAGE_EXECUTOR
    payload = copy.deepcopy(record.to_dict())
    path = state.config.stages_dir / f"actor_{state.rollout_id:08d}_dp_{state.dp_rank:03d}.jsonl"
    with _STAGE_EXECUTOR_LOCK:
        if _STAGE_EXECUTOR is None:
            _STAGE_EXECUTOR = BoundedWorker(capacity=256)
        if not _STAGE_EXECUTOR.submit(_write_stage, path, payload, size_bytes=snapshot_size(payload)):
            status = _PRODUCER_STATUS.setdefault((str(state.config.root), state.dp_rank), {"updates_dropped": 0})
            status["events_dropped"] = status.get("events_dropped", 0) + 1


def _write_stage(path: Any, payload: dict[str, Any]) -> None:
    appender = JsonlAppender(path)
    try:
        appender.append(payload)
    finally:
        appender.close()


def _write_producer_status(state: _Update) -> None:
    key = (str(state.config.root), state.dp_rank)
    status = _PRODUCER_STATUS.setdefault(key, {"updates_dropped": 0})
    atomic_write_json(state.config.train_debug_dir / f"producer_dp_{state.dp_rank:03d}.status.json", dict(status))


def emit_train_scalars(args: Namespace, rollout_id: int, rollout_data: Any) -> None:
    """Compatibility entry point: captures only inside an active real
    update."""
    capture_training_input(rollout_data)


def close_train_appenders(timeout: float = CLOSE_TIMEOUT) -> None:
    """Drain within one deadline; stalled workers retain ownership of
    handles."""
    global _EXECUTOR, _METADATA_EXECUTOR, _STAGE_EXECUTOR
    deadline = time.monotonic() + timeout
    with _UPDATE_EXECUTOR_LOCK, _STAGE_EXECUTOR_LOCK:
        workers = [worker for worker in (_EXECUTOR, _METADATA_EXECUTOR, _STAGE_EXECUTOR) if worker is not None]
        for worker in workers:
            worker.request_close()
    for worker in workers:
        worker.close(max(0.0, deadline - time.monotonic()))
    with _UPDATE_EXECUTOR_LOCK, _STAGE_EXECUTOR_LOCK:
        if _EXECUTOR is not None and _EXECUTOR.done.is_set():
            _EXECUTOR = None
        if _METADATA_EXECUTOR is not None and _METADATA_EXECUTOR.done.is_set():
            _METADATA_EXECUTOR = None
        if _STAGE_EXECUTOR is not None and _STAGE_EXECUTOR.done.is_set():
            _STAGE_EXECUTOR = None


__all__ = [
    "capture_training_input",
    "close_train_appenders",
    "emit_train_scalars",
    "observe_train_step",
    "record_optimizer_result",
]
