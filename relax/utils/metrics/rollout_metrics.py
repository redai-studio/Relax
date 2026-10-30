# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure sample statistics shared by rollout training and evaluation logging."""

from argparse import Namespace

import numpy as np

from relax.utils.metrics.metric_utils import (
    compute_rollout_reward_metrics,
    compute_rollout_stop_reason_and_turn_metrics,
    compute_statistics,
    dict_add_prefix,
    has_repetition,
)
from relax.utils.metrics.speculative_metrics import compute_speculative_log_metrics
from relax.utils.misc import group_by
from relax.utils.multimodal.stats import get_sample_multimodal_stats
from relax.utils.types import Sample


def compute_metrics_from_samples(
    args: Namespace,
    samples: list[Sample],
    *,
    rollout_id: int | None = None,
    include_rloo_diagnostics: bool = True,
) -> dict[str, float]:
    if not samples:
        return {}

    rewarded_samples = [sample for sample in samples if sample.reward is not None]
    reward_cat_key = args.log_reward_category
    reward_category_samples = (
        [sample for sample in rewarded_samples if isinstance(sample.reward, dict) and reward_cat_key in sample.reward]
        if reward_cat_key is not None
        else rewarded_samples
    )
    response_lengths = [sample.effective_response_length for sample in samples]
    multimodal_stats = [get_sample_multimodal_stats(sample) for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_min_mean_max_stats([s["image_count"] for s in multimodal_stats], "image_count/")
    log_dict |= _compute_min_mean_max_stats(
        [s["multimodal_token_count"] for s in multimodal_stats], "multimodal_token_count/"
    )
    log_dict |= compute_rollout_reward_metrics(
        args,
        rewarded_samples,
        include_rloo_diagnostics=include_rloo_diagnostics,
    )
    log_dict |= _compute_zero_std_metrics(args, rewarded_samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, reward_category_samples)
    log_dict |= compute_mopd_metrics(args, rewarded_samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()
    log_dict |= compute_rollout_stop_reason_and_turn_metrics(samples)
    if rollout_id is not None and args.partial_rollout and not args.fully_async:
        staleness_gaps = [rollout_id - sample.metadata.get("start_rollout_id", rollout_id) for sample in samples]
        log_dict["staleness/avg"] = np.mean(staleness_gaps).item()
        log_dict["staleness/max"] = np.max(staleness_gaps).item()
        log_dict["staleness/min"] = np.min(staleness_gaps).item()
        log_dict["global_batch_size"] = len({sample.index for sample in samples})
    return log_dict


def _compute_min_mean_max_stats(values: list[int], prefix: str) -> dict[str, float]:
    if not values:
        return {}
    return {
        f"{prefix}mean": np.mean(values).item(),
        f"{prefix}max": np.max(values).item(),
        f"{prefix}min": np.min(values).item(),
    }


def compute_perf_metrics_from_samples(args: Namespace, samples: list[Sample], rollout_time: float) -> dict[str, float]:
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths: list[int], non_generation_time: list[float], key: str = "") -> None:
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args: Namespace, all_samples: list[Sample]) -> dict[str, int]:
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]) -> bool:
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args: Namespace, all_samples: list[Sample]) -> dict[str, int | float]:
    return compute_speculative_log_metrics(
        all_samples,
        enabled=getattr(args, "sglang_speculative_algorithm", None) is not None,
    )


def _compute_prefix_cache_metrics(args: Namespace, all_samples: list[Sample]) -> dict[str, float]:
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args: Namespace, all_samples: list[Sample]) -> dict[str, float]:
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}


def compute_mopd_metrics(args: Namespace, all_samples: list[Sample]) -> dict[str, float]:
    """Per-data-source accuracy and OPD distillation metrics for MOPD.

    Groups samples by ``metadata[opd_teacher_key]`` and reports for each source:
      - accuracy     : fraction of samples with reward == 1
      - mean_reward  : average reward value
      - teacher_logp : sequence-mean teacher log-prob
      - student_logp : sequence-mean student log-prob
      - logp_gap     : student_logp - teacher_logp (→ 0 as student converges)
      - rkl_approx   : E_student[log p_s - log p_t] per token, averaged across samples

    Returns ``{}`` when OPD is disabled (``opd_teacher_key`` absent).
    """
    opd_teacher_key = getattr(args, "opd_teacher_key", None)
    if not opd_teacher_key:
        return {}

    by_source = group_by(all_samples, lambda s: (s.metadata or {}).get(opd_teacher_key, "unknown"))
    log_dict: dict = {}

    for source, samples in by_source.items():
        n = len(samples)
        if n == 0:
            continue

        label = source.replace("/", "_").replace("-", "_")
        prefix = f"by_source/{label}/"

        raw_rewards = [s.get_reward_value(args) for s in samples]
        log_dict[prefix + "accuracy"] = sum(1 for r in raw_rewards if r == 1) / n
        log_dict[prefix + "mean_reward"] = sum(raw_rewards) / n

        teacher_lp_seq: list[float] = []
        student_lp_seq: list[float] = []
        gap_seq: list[float] = []
        rkl_seq: list[float] = []

        for s in samples:
            t_lp = s.teacher_log_probs
            s_lp = s.rollout_log_probs
            t_mean = sum(t_lp) / len(t_lp) if t_lp and len(t_lp) > 0 else None
            s_mean = sum(s_lp) / len(s_lp) if s_lp and len(s_lp) > 0 else None
            if t_mean is not None:
                teacher_lp_seq.append(t_mean)
            if s_mean is not None:
                student_lp_seq.append(s_mean)
            # Pair gap/rkl per-sample: only when THIS sample has both teacher and
            # student log probs, so the values stay aligned to one sample (the
            # marginal teacher_lp_seq/student_lp_seq lists can differ in membership).
            if t_mean is not None and s_mean is not None:
                gap_seq.append(s_mean - t_mean)
            if t_lp and s_lp:
                n_tok = min(len(t_lp), len(s_lp))
                if n_tok > 0:
                    rkl_seq.append(sum(s_lp[i] - t_lp[i] for i in range(n_tok)) / n_tok)

        if teacher_lp_seq:
            log_dict[prefix + "teacher_logp"] = sum(teacher_lp_seq) / len(teacher_lp_seq)
        if student_lp_seq:
            log_dict[prefix + "student_logp"] = sum(student_lp_seq) / len(student_lp_seq)
        if gap_seq:
            log_dict[prefix + "logp_gap"] = sum(gap_seq) / len(gap_seq)
        if rkl_seq:
            log_dict[prefix + "rkl_approx"] = sum(rkl_seq) / len(rkl_seq)

    return log_dict
