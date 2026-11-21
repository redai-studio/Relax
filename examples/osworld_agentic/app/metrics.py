# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Preserve the handoff OSWorld reward curve on the resident rollout API."""

from argparse import Namespace
from typing import Any

from examples.osworld_agentic.app.reward_post_process import compute_legacy_raw_reward
from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


logger = get_logger(__name__)


def log_rollout_data(
    rollout_id: int,
    args: Namespace,
    samples: list[Sample],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    from relax.distributed.ray.rollout import compute_metrics_from_samples, compute_perf_metrics_from_samples
    from relax.utils import tracking_utils
    from relax.utils.metrics.metric_utils import compute_rollout_step, dict_add_prefix

    metrics = dict(rollout_extra_metrics or {})
    if samples:
        metrics.update(dict_add_prefix(compute_metrics_from_samples(args, samples, rollout_id=rollout_id), "rollout/"))
        metrics.update(dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/"))
        # Override the generic 0/1 outcome metric with the handoff's shaped reward.
        # Session success rate remains under osworld/success_rate/mean.
        metrics["rollout/raw_reward"] = compute_legacy_raw_reward(samples)
    logger.info("rollout %s: %s", rollout_id, {k: v for k, v in metrics.items() if not k.startswith("perf")})
    logger.info("perf %s: %s", rollout_id, {k: v for k, v in metrics.items() if k.startswith("perf")})
    step = compute_rollout_step(args, rollout_id)
    metrics["rollout/step"] = step
    tracking_utils.log(args, metrics, step_key="rollout/step")
    tracking_utils.flush_metrics(args, step)
    return True
