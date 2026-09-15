# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fixed OSWorld task coverage through existing rollout and reward hooks.

Keep the normal small transfer batches: validate every complete task group
before its transfer, then validate the full task set before the final transfer
can mark the partition complete. No framework rollout or reward algorithm is
replaced.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class _StepValidation:
    rollout_id: int
    expected_task_ids: tuple[str, ...]
    group_size: int
    groups: dict[int, tuple[str, frozenset[int]]] = field(default_factory=dict)
    task_groups: dict[str, int] = field(default_factory=dict)
    sample_indices: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if (
            not self.expected_task_ids
            or any(not isinstance(task, str) or not task.strip() for task in self.expected_task_ids)
            or len(set(self.expected_task_ids)) != len(self.expected_task_ids)
            or self.group_size <= 0
        ):
            raise ValueError("Fixed-task rollout requires unique nonempty task IDs and a positive group size")

    def accept_batch(self, samples: Sequence[Any]) -> None:
        """Validate complete groups, counting a multi-turn trajectory only
        once."""
        if not samples:
            raise RuntimeError("Fixed-task rollout cannot transfer an empty sample batch")
        grouped: dict[int, tuple[str, set[int]]] = {}
        for sample in samples:
            metadata = sample.metadata
            task_id = metadata.get("task_id") if isinstance(metadata, dict) else None
            group_index = sample.group_index
            sample_index = sample.index
            if task_id not in self.expected_task_ids:
                raise RuntimeError(f"Fixed-task rollout {self.rollout_id}: unexpected task_id {task_id!r}")
            if not isinstance(group_index, int) or isinstance(group_index, bool) or group_index < 0:
                raise RuntimeError("Fixed-task rollout requires a nonnegative integer Sample.group_index")
            if not isinstance(sample_index, int) or isinstance(sample_index, bool) or sample_index < 0:
                raise RuntimeError("Fixed-task rollout requires a nonnegative integer Sample.index")
            if group_index not in grouped:
                grouped[group_index] = (task_id, set())
            group_task, indices = grouped[group_index]
            if group_task != task_id:
                raise RuntimeError(f"Fixed-task group {group_index} mixes multiple task IDs")
            indices.add(sample_index)

        pending_tasks: set[str] = set()
        pending_indices: set[int] = set()
        for group_index, (task_id, indices) in grouped.items():
            if group_index in self.groups:
                raise RuntimeError(f"Fixed-task group {group_index} was transferred more than once")
            if task_id in self.task_groups or task_id in pending_tasks:
                raise RuntimeError(f"Fixed-task rollout {self.rollout_id}: duplicate task {task_id!r}")
            if len(indices) != self.group_size:
                raise RuntimeError(
                    f"Fixed-task group {group_index} has {len(indices)} unique trajectories; "
                    f"expected {self.group_size}"
                )
            if indices & (self.sample_indices | pending_indices):
                raise RuntimeError("Fixed-task rollout reuses Sample.index across task groups")
            pending_tasks.add(task_id)
            pending_indices.update(indices)
        for group_index, (task_id, indices) in grouped.items():
            self.groups[group_index] = (task_id, frozenset(indices))
            self.task_groups[task_id] = group_index
        self.sample_indices.update(pending_indices)
        if len(self.groups) == len(self.expected_task_ids):
            # This runs inside convert_samples_to_train_data, before the final
            # async_put can seal the training partition, even with small batches.
            self.assert_complete()

    def assert_complete(self) -> None:
        missing = set(self.expected_task_ids) - self.task_groups.keys()
        if missing or len(self.groups) != len(self.expected_task_ids):
            raise RuntimeError(
                f"Fixed-task rollout {self.rollout_id} is incomplete: "
                f"{len(self.groups)}/{len(self.expected_task_ids)} groups; missing={sorted(missing)}"
            )
        if len(self.sample_indices) != len(self.expected_task_ids) * self.group_size:
            raise RuntimeError("Fixed-task rollout has an incorrect total trajectory count")

    def validate_output(self, output_groups: Sequence[Sequence[Any]]) -> None:
        if len(output_groups) != len(self.expected_task_ids):
            raise RuntimeError("Fixed-task rollout output has an incorrect number of task groups")
        observed = _StepValidation(self.rollout_id, self.expected_task_ids, self.group_size)
        for samples in output_groups:
            if not samples or len({sample.group_index for sample in samples}) != 1:
                raise RuntimeError("Fixed-task rollout output must contain one complete group per task")
            observed.accept_batch(samples)
        observed.assert_complete()
        if observed.groups != self.groups:
            raise RuntimeError("Fixed-task rollout output differs from the validated training groups")


_ENTRY_LOCK = threading.Lock()
_VALIDATION_LOCK = threading.Lock()
_ACTIVE_STEP: _StepValidation | None = None


def _log_reward_environment(rollout_id: int, stage: str) -> None:
    import json
    import os

    keys = (
        "OSWORLD_RUN_TAG",
        "OSWORLD_PER_TRAJ_NORM",
        "OSWORLD_MAX_STEPS_REF",
        "OSWORLD_LENGTH_PENALTY_COEF",
        "OSWORLD_CAP_PENALTY",
        "OSWORLD_ACC_LO",
        "OSWORLD_ACC_HI",
        "OSWORLD_ADV_CLIP",
        "OSWORLD_REWARD_DUMP",
    )
    values = {key: os.environ.get(key) for key in keys}
    values.update(pid=os.getpid(), rollout_id=rollout_id, stage=stage, module=__file__)
    print("[osworld_reward_environment] " + json.dumps(values), flush=True)


def post_process_rewards(args: Any, samples: list[Any]) -> Any:
    """Validate this transfer, then preserve the existing OSWorld reward
    behavior."""
    from examples.osworld_agentic.app.reward_post_process import post_process_rewards as original_post_process

    with _VALIDATION_LOCK:
        if _ACTIVE_STEP is None:
            raise RuntimeError("Fixed-task reward validation requires the fixed-task rollout wrapper")
        _ACTIVE_STEP.accept_batch(samples)
        _log_reward_environment(_ACTIVE_STEP.rollout_id, "reward")
        return original_post_process(args, samples)


def generate_rollout(
    args: Any,
    rollout_id: int,
    data_source: Any,
    data_system_client: Any = None,
    evaluation: bool = False,
) -> Any:
    """Open one bounded dataset pass and delegate generation to the normal
    runtime."""
    from relax.agentic.rollout import generate_rollout as original_generate

    if evaluation:
        return original_generate(args, rollout_id, data_source, data_system_client, evaluation=True)

    import ray

    global _ACTIVE_STEP
    with _ENTRY_LOCK:
        expected = tuple(ray.get(data_source.begin_rollout_step.remote(rollout_id)))
        state = _StepValidation(rollout_id, expected, args.n_samples_per_prompt)
        with _VALIDATION_LOCK:
            if _ACTIVE_STEP is not None:
                raise RuntimeError("A fixed-task rollout step is already active")
            _ACTIVE_STEP = state
        try:
            _log_reward_environment(rollout_id, "rollout_start")
            from examples.osworld_agentic.training.sandbox import install_prepare_retry
            from relax.agentic.rollout import get_agentic_resident_pipeline

            install_prepare_retry(get_agentic_resident_pipeline().runtime_domain)
            output = original_generate(args, rollout_id, data_source, data_system_client, evaluation=False)
            with _VALIDATION_LOCK:
                state.assert_complete()
                state.validate_output(output.samples)
            ray.get(data_source.finish_rollout_step.remote(rollout_id))
            from relax.utils.logging_utils import get_logger

            get_logger(__name__).info(
                "Fixed OSWorld rollout %s: %s unique tasks, %s logical sessions; coverage validated",
                rollout_id,
                len(state.task_groups),
                len(state.sample_indices),
            )
            return output
        finally:
            with _VALIDATION_LOCK:
                _ACTIVE_STEP = None
