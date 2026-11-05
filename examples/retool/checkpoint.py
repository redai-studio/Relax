# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Reject incomplete native resumes before Relax can fall back to HF
weights."""

import argparse
import json
import sys
from pathlib import Path


def validate_checkpoint(root: Path, *, sft: bool = False) -> int | None:
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
        if not sft and all(path.name in initialization_files for path in root.iterdir()):
            if not any((root / "rollout_result").rglob("*.jsonl")):
                return None
            raise ValueError("Uncheckpointed rollout results exist; choose a new experiment instead of resetting")
        raise ValueError("Nonempty training directory lacks a committed native checkpoint; choose a new experiment")
    step = int(tracker.read_text().strip())
    checkpoint = root / f"iter_{step:07d}"
    if step < 0 or not (checkpoint / ".metadata").is_file():
        raise ValueError("Incomplete native checkpoint; refusing an HF-only restart")
    from megatron.core.dist_checkpointing import load_common_state_dict
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(checkpoint).read_metadata()
    if not metadata.storage_data:
        raise ValueError("Native checkpoint has no tensor storage metadata")
    for item in metadata.storage_data.values():
        path = checkpoint / item.relative_path
        if not path.resolve().is_relative_to(checkpoint.resolve()) or not path.is_file():
            raise ValueError("Native checkpoint tensor storage is missing or outside the checkpoint")
        if path.stat().st_size < item.offset + item.length:
            raise ValueError("Native checkpoint tensor storage is truncated")
    keys = [str(key) for key in metadata.state_dict_metadata]
    if not any("optimizer" in key for key in keys) or not any("rng_state" in key for key in keys):
        raise ValueError("Resume requires saved optimizer and RNG state")
    common = load_common_state_dict(checkpoint)
    if "opt_param_scheduler" not in common or common.get("iteration") != step:
        raise ValueError("Resume requires the matching scheduler and native iteration")
    if not sft:
        state_path = root / "dataset" / f"retool_{step}.json"
        if not state_path.is_file():
            raise ValueError("Resume requires the matching ReTool sampling cursor")
        state = json.loads(state_path.read_text())
        if state.get("cursor") != step + 1 or state.get("rng_state_version") != 1 or not state.get("rng_state"):
            raise ValueError("Resume requires the matching ReTool cursor and native sampling RNG snapshot")
    if sft:
        if not (root / "sft_data_state" / f"{step + 1:07d}.json").is_file():
            raise ValueError("Resume requires the matching SFT data state")
    return step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--sft", action="store_true")
    args = parser.parse_args()
    step = validate_checkpoint(args.root, sft=args.sft)
    sys.stdout.write(str(step + 1 if step is not None else 0) + "\n")


if __name__ == "__main__":
    main()
