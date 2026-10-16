# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Log original outcome accuracy without changing GAGAR training rewards."""

from argparse import Namespace

from relax.utils import tracking_utils
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_utils import compute_rollout_step, dict_add_prefix
from relax.utils.metrics.rollout_metrics import compute_metrics_from_samples, compute_perf_metrics_from_samples
from relax.utils.types import Sample


def log_rollout(
    rollout_id: int,
    args: Namespace,
    samples: list[Sample],
    rollout_extra_metrics: dict | None,
    rollout_time: float,
) -> bool:
    if args.load_debug_rollout_data:
        return True
    metrics = dict(rollout_extra_metrics or {})
    if samples:
        metrics |= dict_add_prefix(compute_metrics_from_samples(args, samples, rollout_id=rollout_id), "rollout/")
        metrics |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
        outcomes = [sample.reward["test_reward"] for sample in samples if sample.reward is not None]
        if outcomes:
            metrics["rollout/group_reward_mean"] = metrics["rollout/raw_reward"]
            metrics["rollout/raw_reward"] = sum(outcomes) / len(outcomes)
    get_logger(__name__).info("rollout %s: %s", rollout_id, metrics)
    step = compute_rollout_step(args, rollout_id)
    metrics["rollout/step"] = step
    tracking_utils.log(args, metrics, step_key="rollout/step")
    tracking_utils.flush_metrics(args, step)
    return True
