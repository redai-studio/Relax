# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Select the task's grader process without changing its PID or resource
owner."""

import argparse
import json
import os
import sys
from pathlib import Path

from examples.gagar.app.config import load_task_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and run one configured grader session.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.input.read_text())
    config = manifest.get("grader_config")
    if config is None:
        config = load_task_config(args.task)
    # exec keeps the worker PID used by leases, cancellation and orphan cleanup.
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            *config["environment"]["entrypoint"],
            "--input",
            str(args.input),
            "--output",
            str(args.output),
        ],
    )


if __name__ == "__main__":
    main()
