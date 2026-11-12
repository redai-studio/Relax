# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Save a local ClearML snapshot for the existing startup writer."""

from argparse import Namespace
from pathlib import Path
from typing import Any

from relax.utils.agentic_obs.config import resolve_agentic_obs_config
from relax.utils.agentic_obs.writers import atomic_write_json, best_effort, submit_background


@best_effort
def record_clearml_task(args: Namespace, task: Any) -> None:
    config = resolve_agentic_obs_config(args)
    if config is not None:
        submit_background(_write_clearml_task, config.root / "clearml_task.json", task)


def _write_clearml_task(path: Path, task: Any) -> None:
    # Keep the exact task from this initialization, not Task.current_task().
    atomic_write_json(path, {"task_id": task.id, "url": task.get_output_log_web_page()})
