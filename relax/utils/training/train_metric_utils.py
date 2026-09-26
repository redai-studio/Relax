# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import math
from argparse import Namespace
from copy import deepcopy
from typing import TYPE_CHECKING

from relax.utils import tracking_utils
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_utils import compute_rollout_step
from relax.utils.timer import Timer


if TYPE_CHECKING:
    from relax.utils.training.flops_counter import FlopsCounter


logger = get_logger(__name__)


def log_perf_data_raw(
    rollout_id: int,
    args: Namespace,
    is_primary_rank: bool,
    flops_counter: FlopsCounter | None = None,
    world_size: int = 1,
) -> None:
    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()

    if not is_primary_rank:
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}

    if timer_instance.seq_lens:
        log_dict["perf/actor_train_tokens"] = sum(timer_instance.seq_lens)

    per_gpu_tflops: float | None = None
    if ("perf/actor_train_time" in log_dict) and (flops_counter is not None):
        seq_lens = timer_instance.seq_lens
        images_seqlens = getattr(timer_instance, "images_seqlens", None) or None
        audio_seqlens = getattr(timer_instance, "audio_seqlens", None) or None
        estimated_tflops, peak_tflops = flops_counter.estimate(
            batch_seqlens=seq_lens, delta_time=1.0, images_seqlens=images_seqlens, audio_seqlens=audio_seqlens
        )
        # estimated_tflops is total fwd+bwd TFLOPS at delta_time=1 => raw TFLOPS count
        # Normalize to per-GPU
        per_gpu_tflops = estimated_tflops / world_size

        if "perf/log_probs_time" in log_dict:
            # Forward only = fwd+bwd / 3
            log_dict["perf/log_probs_tflops"] = per_gpu_tflops / 3 / log_dict["perf/log_probs_time"]

        if "perf/ref_log_probs_time" in log_dict:
            log_dict["perf/ref_log_probs_tflops"] = per_gpu_tflops / 3 / log_dict["perf/ref_log_probs_time"]

        if log_dict["perf/actor_train_time"] > 0:
            # Training includes fwd+bwd, use full 6N flops
            log_dict["perf/actor_train_tflops"] = per_gpu_tflops / log_dict["perf/actor_train_time"]
            log_dict["perf/actor_train_tok_per_s"] = sum(seq_lens) / log_dict["perf/actor_train_time"]

        # MFU = achieved_per_gpu_tflops / device_peak_tflops
        if math.isfinite(peak_tflops) and peak_tflops > 0:
            log_dict["perf/device_peak_tflops"] = peak_tflops
            if "perf/actor_train_tflops" in log_dict:
                log_dict["perf/mfu/actor_train"] = log_dict["perf/actor_train_tflops"] / peak_tflops
            if "perf/log_probs_tflops" in log_dict:
                log_dict["perf/mfu/actor_infer"] = log_dict["perf/log_probs_tflops"] / peak_tflops
            if "perf/ref_log_probs_tflops" in log_dict:
                log_dict["perf/mfu/ref_infer"] = log_dict["perf/ref_log_probs_tflops"] / peak_tflops

    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            log_dict["perf/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time
            if timer_instance.seq_lens:
                log_dict["perf/step_token_per_s"] = sum(timer_instance.seq_lens) / total_time
            response_lens = getattr(timer_instance, "response_lens", None)
            if response_lens:
                log_dict["perf/step_resp_token_per_s"] = sum(response_lens) / total_time

    logger.info(f"perf {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")
