# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared completed-step triggers for training checkpoints and offline eval."""

from argparse import Namespace


def interval_is_due(args: Namespace, interval: int | str | None, completed_steps: int) -> bool:
    if interval is None or completed_steps <= 0:
        return False
    if interval == "epoch":
        size = getattr(args, "_sft_train_dataset_size", None)
        if size is None or size <= 0:
            raise ValueError("Epoch intervals require the resolved SFT training dataset size.")
        batch_size = args.global_batch_size
        return completed_steps * batch_size // size > (completed_steps - 1) * batch_size // size
    return interval > 0 and completed_steps % interval == 0


def should_save_checkpoint(args: Namespace, completed_steps: int) -> bool:
    if getattr(args, "save", None) is None or completed_steps <= 0:
        return False
    interval = getattr(args, "save_interval", None)
    return bool(
        getattr(args, "rotate_ckpt", False)
        or interval is not None
        and (interval_is_due(args, interval, completed_steps) or completed_steps == args.num_rollout)
    )
