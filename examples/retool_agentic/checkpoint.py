# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check ReTool sampling state before starting or resuming an experiment."""

import argparse
import json
import sys
from pathlib import Path


def validate_checkpoint(root: Path) -> int | None:
    if not root.exists() or not any(root.iterdir()):
        return None
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        # Native initialization may create these before the first optimizer checkpoint.
        initialization_files = {
            "rollout_result",
            "tensorboard_log",
            "transformer_config.json",
            "transformer_config.pkl",
        }
        if all(path.name in initialization_files for path in root.iterdir()):
            if not any((root / "rollout_result").rglob("*.jsonl")):
                return None
            raise ValueError("Uncheckpointed rollout results exist; choose a new experiment instead of resetting")
        raise ValueError("Nonempty training directory lacks a committed native checkpoint; choose a new experiment")
    step = int(tracker.read_text().strip())
    if step < 0:
        raise ValueError("ReTool resume requires a non-negative training iteration")
    state_path = root / "dataset" / f"retool_{step}.json"
    if not state_path.is_file():
        raise ValueError("Resume requires the matching ReTool sampling cursor")
    state = json.loads(state_path.read_text())
    if state.get("cursor") != step + 1 or state.get("rng_state_version") != 1 or not state.get("rng_state"):
        raise ValueError("Resume requires the matching ReTool cursor and native sampling RNG snapshot")
    return step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--fresh", action="store_true", help="Require an empty experiment directory; never resume")
    args = parser.parse_args()
    if args.fresh:
        if args.root.exists() and any(args.root.iterdir()):
            parser.error("Fresh training requires an empty experiment directory; change SAVE_DIR or EXP_NAME")
        step = None
    else:
        step = validate_checkpoint(args.root)
    sys.stdout.write(str(step + 1 if step is not None else 0) + "\n")


if __name__ == "__main__":
    main()
