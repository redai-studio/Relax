# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Optimizer progress within one source process generation."""


def record_optimizer_step(args: object, rollout_id: int, step_id: int, updated: bool) -> None:
    previous = getattr(args, "_weight_source_progress", {})
    args._weight_source_progress = {
        "optimizer_attempts": previous.get("optimizer_attempts", 0) + 1,
        "optimizer_updates": previous.get("optimizer_updates", 0) + int(updated),
        "last_step_updated": bool(updated),
        "rollout_id": int(rollout_id),
        "step_in_rollout": int(step_id),
    }


def source_progress(args: object) -> dict:
    return dict(
        getattr(
            args,
            "_weight_source_progress",
            {
                "optimizer_attempts": 0,
                "optimizer_updates": 0,
                "last_step_updated": False,
                "rollout_id": None,
                "step_in_rollout": None,
            },
        )
    )
