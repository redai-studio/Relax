# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Lightweight GRS configuration shared by launchers, workers and the agent."""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def load_task_config(task: str, path: str | Path | None = None) -> dict:
    """Resolve a task's prompts and settings without importing the agent
    runtime."""
    path = Path(path or os.environ.get("GAGAR_CONFIG") or Path(__file__).resolve().parents[1] / "grader.yaml")
    config = yaml.safe_load(path.read_text())
    selected = config["tasks"][task]
    return {
        "model": dict(selected.get("model", {})),
        "agent": {
            **config.get("agent", {}),
            **selected.get("agent", {}),
            "system_template": selected["system_template"],
            "instance_template": selected["instance_template"],
        },
        "environment": {**config.get("environment", {}), **selected["environment"]},
        "genrm": dict(selected["genrm"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read one task's GRS configuration.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--genrm-args", action="store_true")
    args = parser.parse_args()
    config = load_task_config(args.task)
    if args.genrm_args:
        # Each line becomes one argument, without shell evaluation.
        genrm = config["genrm"]
        values = [
            "--genrm-engine-config",
            json.dumps(genrm["engine"]),
            "--genrm-sampling-config",
            json.dumps(genrm["sampling"]),
        ]
        sys.stdout.write("\n".join(values) + "\n")
    else:
        sys.stdout.write(json.dumps(config) + "\n")


if __name__ == "__main__":
    main()
