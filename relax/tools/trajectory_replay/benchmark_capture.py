# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Microbenchmark for the trajectory-replay capture hook overhead.

Measures the per-call cost of capture_hooks.capture_policy_loss in the three
modes that matter for the hot-path budget:

- disabled: capture never enabled (default) — must be ~0.
- unselected: capture enabled but the step is not selected.
- selected: capture enabled and the step is captured.

This isolates the hook overhead only. The full step-latency budget (spec 7.3:
disabled <0.5%, selected-step <3%) must be measured on a real training run,
because this microbenchmark does not include forward/backward.

Usage:
    python -m relax.tools.trajectory_replay.benchmark_capture [--iters N] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import time

import torch

from relax.utils.logging_utils import get_logger
from relax.utils.replay import capture_hooks
from relax.utils.replay.capture import CaptureConfig, disable, enable


logger = get_logger(__name__)


def _synthetic_loss_args(device: torch.device) -> dict:
    num_tokens = 8
    return {
        "old_log_probs": torch.zeros(num_tokens, device=device),
        "log_probs": torch.zeros(num_tokens, device=device),
        "entropy": torch.full((num_tokens,), 0.5, device=device),
        "advantages": torch.ones(num_tokens, device=device),
        "loss_masks": [torch.ones(2, device=device) for _ in range(4)],
        "response_lengths": [2, 2, 2, 2],
        "total_lengths": [3, 3, 3, 3],
        "reported_loss": {
            "loss": torch.tensor(0.0, device=device),
            "pg_loss": torch.tensor(0.0, device=device),
            "entropy_loss": torch.tensor(0.0, device=device),
            "pg_clipfrac": torch.tensor(0.0, device=device),
            "ppo_kl": torch.tensor(0.0, device=device),
        },
    }


def _time_hook(iters: int, device: torch.device) -> float:
    kwargs = _synthetic_loss_args(device)
    start = time.perf_counter()
    for _ in range(iters):
        capture_hooks.capture_policy_loss(**kwargs)
    return (time.perf_counter() - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10_000)
    parser.add_argument("--output-dir", type=str, default="/tmp/replay-benchmark")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s iters=%s", device, args.iters)

    disable()
    disabled_ns = _time_hook(args.iters, device) * 1e9

    enable(CaptureConfig(enabled=True, output_dir=args.output_dir, selected_steps={(0, 0)}))
    unselected_ns = _time_hook(args.iters, device) * 1e9

    enable(CaptureConfig(enabled=True, output_dir=args.output_dir))
    from relax.utils.replay import capture as capture_module

    capture_module.begin_step((0, 0), identity=_identity(), config=_config(), bundle_id="bench")
    selected_ns = _time_hook(args.iters, device) * 1e9
    capture_module.end_step()
    disable()

    logger.info("disabled   : %8.1f ns/call", disabled_ns)
    logger.info("unselected : %8.1f ns/call", unselected_ns)
    logger.info("selected   : %8.1f ns/call", selected_ns)
    logger.info("full step-latency budget must be measured on a real training run")


def _identity():
    from relax.utils.replay.schema import ActorStepId, Identity

    return Identity(actor_step_id=ActorStepId(rollout_id=0, step_id=0), rank={"dp": 1, "tp": 1, "pp": 1, "cp": 1})


def _config():
    from relax.utils.replay.schema import RecomputeConfig

    return RecomputeConfig(advantage_estimator="grpo", n_samples_per_prompt=2)


if __name__ == "__main__":
    main()
