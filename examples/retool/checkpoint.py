# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Reject incomplete native resumes before Relax can fall back to HF
weights."""

import argparse
import sys
from pathlib import Path


def validate_checkpoint(root: Path, *, sft: bool = False) -> int | None:
    if not root.exists():
        return None
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise ValueError("Training directory lacks a committed native checkpoint; choose a new experiment")
    step = int(tracker.read_text().strip())
    checkpoint = root / f"iter_{step:07d}"
    if step < 0 or not (checkpoint / ".metadata").is_file():
        raise ValueError("Incomplete native checkpoint; refusing an HF-only restart")
    from megatron.core.dist_checkpointing import load_common_state_dict
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(checkpoint).read_metadata()
    keys = [str(key) for key in metadata.state_dict_metadata]
    if not any("optimizer" in key for key in keys) or not any("rng_state" in key for key in keys):
        raise ValueError("Resume requires saved optimizer and RNG state")
    common = load_common_state_dict(checkpoint)
    if "opt_param_scheduler" not in common or common.get("iteration") != step:
        raise ValueError("Resume requires the matching scheduler and native iteration")
    if not sft and not (root / "dataset" / f"retool_{step}.json").is_file():
        raise ValueError("Resume requires the matching ReTool sampling cursor")
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
