# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import asyncio
import copy
import inspect
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class DeferredBatchState(str, Enum):
    GENERATED = "GENERATED"
    SCORING = "SCORING"
    COMPLETE = "COMPLETE"
    PUBLISHING = "PUBLISHING"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class PendingSampleKey:
    run_id: str
    rollout_id: int
    group_ordinal: int
    sample_ordinal: int
    sample_index: Any


_WRITEBACK_FIELDS = (
    "reward",
    "custom_advantage",
    "remove_sample",
    "teacher_log_probs",
    "teacher_topk_token_ids",
    "teacher_topk_log_probs",
    "teacher_at_student_topk_log_probs",
    "student_at_teacher_topk_log_probs",
    "opd_topk_token_ids",
    "opd_topk_student_log_probs",
    "opd_topk_teacher_log_probs",
    "opd_topk_ksz",
    "teacher_tokens",
    "teacher_prompt_length",
    "teacher_image_data",
    "teacher_image_b64_list",
    "teacher_image_grid_thw",
)
Stage = Callable[[list[Any]], Awaitable[Any]]


class DeferredBatch:
    def __init__(self, run_id: str, *, max_samples: int = 65536, timeout: float = 1800.0) -> None:
        if not isinstance(run_id, str) or not run_id or max_samples <= 0 or timeout <= 0:
            raise ValueError("Deferred batch requires a run identity and positive limits")
        self.run_id = run_id
        self.max_samples = max_samples
        self.timeout = timeout
        self.state = DeferredBatchState.GENERATED
        self.groups: list[list[Any]] = []
        self.samples: list[Any] = []
        self.keys: tuple[PendingSampleKey, ...] = ()
        self.key: tuple[str, int, int] | None = None
        self.policy_version: str | None = None
        self.failure: str | None = None
        self._lock = asyncio.Lock()
        self._source_identity: tuple[tuple[int, ...], ...] | None = None

    def _capture(self, rollout_id: int, groups: Sequence[Sequence[Any]], needs_student: bool) -> list[list[Any]]:
        self.groups = [list(group) for group in groups]
        self.samples = [sample for group in self.groups for sample in group]
        if len(self.samples) > self.max_samples:
            raise ValueError("Deferred batch exceeds its sample capacity")
        if len({id(sample) for sample in self.samples}) != len(self.samples):
            raise ValueError("A pending Sample object cannot occupy multiple export rows")
        self.key = (self.run_id, rollout_id, 0)
        self.keys = tuple(
            PendingSampleKey(self.run_id, rollout_id, group_ordinal, sample_ordinal, getattr(sample, "index", None))
            for group_ordinal, group in enumerate(self.groups)
            for sample_ordinal, sample in enumerate(group)
        )
        if needs_student:
            versions = set()
            for sample in self.samples:
                if int(getattr(sample, "response_length", 0) or 0) <= 0 or getattr(sample, "remove_sample", False):
                    continue
                recorded = getattr(sample, "weight_versions", None) or []
                if not recorded or any(not isinstance(version, str) or not version for version in recorded):
                    raise ValueError("Deferred student scoring requires a known generation weight version")
                versions.update(recorded)
            if len(versions) > 1:
                raise ValueError("Deferred student scoring cannot restore mixed policy versions")
            self.policy_version = next(iter(versions), None)
        staged = []
        for group in self.groups:
            staged_group = []
            for sample in group:
                clone = copy.copy(sample)
                for field in ("tokens", "rollout_tokens", "loss_mask", "weight_versions", "metadata"):
                    if hasattr(sample, field):
                        setattr(clone, field, copy.deepcopy(getattr(sample, field)))
                for field in _WRITEBACK_FIELDS:
                    if hasattr(sample, field):
                        setattr(clone, field, copy.deepcopy(getattr(sample, field)))
                staged_group.append(clone)
            staged.append(staged_group)
        return staged

    async def complete(
        self,
        rollout_id: int,
        groups: Sequence[Sequence[Any]],
        *,
        offload_rollout: Callable[[], Awaitable[Any]],
        activate_role: Callable[[str], Awaitable[Any]],
        deactivate_role: Callable[[str], Awaitable[Any]],
        reward_stage: Callable[[list[list[Any]]], Awaitable[Any]] | None = None,
        prepare_teacher: Stage | None = None,
        teacher_stage: Stage | None = None,
        student_stage: Stage | None = None,
        assemble_validate: Callable[[list[Any]], Any] | None = None,
        restore_student: Callable[[str], Awaitable[Any]] | None = None,
        commit: Callable[[DeferredBatch], Awaitable[Any]] | None = None,
    ) -> None:
        async with self._lock:
            identity = tuple(tuple(id(sample) for sample in group) for group in groups)
            if self.key is not None:
                if self.key[:2] != (self.run_id, rollout_id) or identity != self._source_identity:
                    raise ValueError("Deferred batch reentry has a different batch identity")
                if self.state == DeferredBatchState.COMMITTED:
                    return
                raise RuntimeError(f"Deferred batch is already {self.state.value}; create a new revision explicitly")
            self._source_identity = identity
            try:
                staged_groups = self._capture(rollout_id, groups, student_stage is not None)
                if student_stage is not None and self.policy_version is not None and restore_student is None:
                    raise ValueError("Deferred student scoring requires a pinned-weight restore callback")
                if teacher_stage is not None and assemble_validate is None:
                    raise ValueError("Deferred Teacher scoring requires full result validation")
                await asyncio.wait_for(
                    self._complete(
                        staged_groups,
                        offload_rollout,
                        activate_role,
                        deactivate_role,
                        reward_stage,
                        prepare_teacher,
                        teacher_stage,
                        student_stage,
                        assemble_validate,
                        restore_student,
                        commit,
                    ),
                    timeout=self.timeout,
                )
            except BaseException as exc:
                self.state = DeferredBatchState.FAILED
                self.failure = f"{type(exc).__name__}: {exc}"
                raise

    async def _complete(
        self,
        groups,
        offload_rollout,
        activate_role,
        deactivate_role,
        reward_stage,
        prepare_teacher,
        teacher_stage,
        student_stage,
        assemble_validate,
        restore_student,
        commit,
    ) -> None:
        samples = [sample for group in groups for sample in group]
        self.state = DeferredBatchState.SCORING
        if prepare_teacher is not None:
            await prepare_teacher(samples)
        await offload_rollout()

        async def score_role(role, stage, values):
            await activate_role(role)
            try:
                await stage(values)
            finally:
                await deactivate_role(role)

        if reward_stage is not None:
            await score_role("genrm", reward_stage, groups)
            for ordinal, sample in enumerate(samples):
                reward = getattr(sample, "reward", None)
                if getattr(sample, "remove_sample", False):
                    continue
                values = list(reward.values()) if isinstance(reward, dict) else [reward]
                if not values or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value) for value in values
                ):
                    raise ValueError(f"Deferred reward missing or nonfinite at sample ordinal {ordinal}")
        if teacher_stage is not None:
            await score_role("teacher", teacher_stage, samples)
        if student_stage is not None and self.policy_version is not None:
            try:
                await restore_student(self.policy_version)
                await student_stage(samples)
            finally:
                await deactivate_role("rollout")
        if assemble_validate is not None:
            result = assemble_validate(samples)
            if inspect.isawaitable(result):
                await result
        self.state = DeferredBatchState.COMPLETE
        for original, scored in zip(self.samples, samples, strict=True):
            for field in _WRITEBACK_FIELDS:
                if hasattr(scored, field):
                    setattr(original, field, getattr(scored, field))
        self.state = DeferredBatchState.PUBLISHING
        if commit is not None:
            await commit(self)
        self.state = DeferredBatchState.COMMITTED
