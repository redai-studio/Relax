# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run configuration snapshots after rollout schedule resolution."""

from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path

from relax.utils.agentic_obs.config import resolve_agentic_obs_config, write_manifest
from relax.utils.agentic_obs.records import ExportStageRecord
from relax.utils.agentic_obs.writers import JsonlAppender, best_effort, snapshot_size, submit_background


# Explicitly select task metadata instead of dumping args (which can include
# credentials, environment variables, and executable agent commands).
_RUN_CONFIG_FIELDS = (
    "num_epoch",
    "num_rollout",
    "num_rollout_per_epoch",
    "rollout_batch_size",
    "n_samples_per_prompt",
    "global_batch_size",
    "micro_batch_size",
    "num_steps_per_rollout",
    "use_dynamic_batch_size",
    "use_dynamic_global_batch_size",
    "train_backend",
    "hf_checkpoint",
    "prompt_data",
    "rollout_global_dataset",
    "seed",
    "loss_type",
    "advantage_estimator",
    "reward_key",
    "custom_rm_path",
    "agentic_custom_advantage_path",
    "fully_async",
    "partial_rollout",
    "max_staleness",
    "resource",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "dynamic_context_parallel",
    "rollout_num_gpus_per_engine",
    "rollout_max_response_len",
    "rollout_temperature",
    "eval_interval",
    "save_interval",
    "debug_rollout_only",
    "debug_train_only",
)


@best_effort
def record_run_startup(args: Namespace) -> None:
    """Record the resolved plan before rollout starts; no-op when disabled.

    Steps here are rollout iterations, not optimizer updates. Epoch equivalents
    describe the full capped plan, including any steps restored from a
    checkpoint. Dynamic batching and filtering can change the actual number of
    updates. Each initialization gets its own stage_id, including service
    restarts.
    """
    config = resolve_agentic_obs_config(args)
    if config is None:
        return
    write_manifest(args, config)
    start = ExportStageRecord(None, "startup", "start", "", run_id=config.run_id, scope="run")
    run_config = {name: getattr(args, name, None) for name in _RUN_CONFIG_FIELDS}
    total_steps = run_config["num_rollout"]
    steps_per_epoch = run_config["num_rollout_per_epoch"]
    run_config.update(
        step_unit="rollout",
        planned_epoch_equivalents=(
            total_steps / steps_per_epoch if total_steps is not None and steps_per_epoch else None
        ),
        # The actor may still restore a different step from its checkpoint.
        configured_start_rollout_id=getattr(args, "start_rollout_id", None),
    )
    completed = ExportStageRecord(
        None,
        "startup",
        "completed",
        "",
        run_id=config.run_id,
        scope="run",
        stage_id=start.stage_id,
        event_seq=2,
        run_config=run_config,
    )
    records = copy.deepcopy([start.to_dict(), completed.to_dict()])
    submit_background(_write_startup, config.stages_dir / "startup.jsonl", records, size_bytes=snapshot_size(records))


@best_effort
def _include_clearml_task(path: Path, records: list[dict]) -> None:
    try:
        task = json.loads((path.parent.parent / "clearml_task.json").read_text())
    except FileNotFoundError:
        return
    if isinstance(task, dict) and all(isinstance(task.get(key), str) and task[key] for key in ("task_id", "url")):
        records[-1]["run_config"]["clearml_task"] = {key: task[key] for key in ("task_id", "url")}


def _write_startup(path: Path, records: list[dict]) -> None:
    _include_clearml_task(path, records)
    writer = JsonlAppender(path)
    try:
        writer.append_many(records)
    finally:
        writer.close()
