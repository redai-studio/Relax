# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""One complete, ordered OSWorld task set per synchronous rollout."""

from relax.engine.rollout.data_source import RolloutDataSourceWithBuffer
from relax.utils.types import Sample


class FixedTaskDataSource(RolloutDataSourceWithBuffer):
    """Bound each rollout to one dataset pass; never replace a dropped
    group."""

    def __init__(self, args):
        if not args.rollout_global_dataset or args.use_streaming_dataset or args.rollout_shuffle:
            raise ValueError("Fixed OSWorld tasks require an eager, unshuffled global dataset.")
        super().__init__(args)
        if len(self.dataset) != args.rollout_batch_size:
            raise ValueError(
                f"Fixed OSWorld tasks require dataset size == rollout_batch_size, "
                f"got {len(self.dataset)} tasks and {args.rollout_batch_size} groups."
            )
        task_ids = [sample.metadata.get("task_id") for sample in self.dataset.samples]
        if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
            raise ValueError("Every fixed OSWorld task must have a nonempty metadata.task_id.")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Fixed OSWorld dataset contains duplicate task_id values.")
        self.task_ids = tuple(task_ids)
        self._active_rollout_id = None
        self._issued_groups = 0

    def begin_rollout_step(self, rollout_id: int) -> tuple[str, ...]:
        if self._active_rollout_id is not None:
            raise RuntimeError(f"Fixed OSWorld rollout {self._active_rollout_id} has not finished.")
        if self.buffer:
            raise RuntimeError("Fixed OSWorld tasks cannot reuse buffered groups.")
        # Old checkpoints may contain a streaming cursor in the middle of an epoch.
        # Keep the restored session/group counters, but always begin at the first task.
        self.sample_offset = 0
        self._issued_groups = 0
        self._active_rollout_id = rollout_id
        return self.task_ids

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        if self._active_rollout_id is None:
            raise RuntimeError("Call begin_rollout_step before requesting fixed OSWorld tasks.")
        if num_samples < 0:
            raise ValueError("num_samples must be nonnegative.")
        remaining = len(self.task_ids) - self._issued_groups
        if num_samples > remaining:
            raise RuntimeError(
                f"Fixed OSWorld rollout {self._active_rollout_id} requested {num_samples} groups "
                f"with only {remaining} unissued tasks remaining. A group was dropped or filtered; "
                "refusing to substitute another task. Check agent preparation/export failures."
            )
        groups = super().get_samples(num_samples)
        self._issued_groups += len(groups)
        return groups

    def add_samples(self, samples: list[list[Sample]]):
        if samples:
            raise RuntimeError("Fixed OSWorld tasks do not support buffered or partial rollouts.")

    def finish_rollout_step(self, rollout_id: int):
        if self._active_rollout_id != rollout_id or self._issued_groups != len(self.task_ids):
            raise RuntimeError(
                f"Incomplete fixed OSWorld rollout {rollout_id}: issued "
                f"{self._issued_groups}/{len(self.task_ids)} task groups."
            )
        self._active_rollout_id = None
        self.epoch_id += 1
