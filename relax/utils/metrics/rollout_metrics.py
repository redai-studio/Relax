# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Production rollout metrics, callable on CPU without starting Ray engines.

The distributed rollout module re-exports this aggregation entry point. Keep
observability independent of engine initialization so offline diagnostics and
integration tests exercise the same implementation as training.
"""

import numpy as np

from relax.utils.metrics.metric_utils import (
    compute_rollout_reward_metrics,
    compute_statistics,
    dict_add_prefix,
    has_repetition,
)
from relax.utils.misc import group_by
from relax.utils.multimodal.stats import get_sample_multimodal_stats
from relax.utils.opd.opd_utils import compute_mopd_metrics
from relax.utils.types import Sample


def compute_metrics_from_samples(
    args,
    samples,
    *,
    rollout_id: int | None = None,
    include_rloo_diagnostics: bool = True,
):
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
    log_dict["num_turn/mean"] = np.mean([s.metadata.get("rollout_turns", 1) for s in samples]).item()
    log_dict["num_turn/max"] = np.max([s.metadata.get("rollout_turns", 1) for s in samples]).item()
    log_dict["num_turn/min"] = np.min([s.metadata.get("rollout_turns", 1) for s in samples]).item()
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


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if getattr(args, "sglang_speculative_algorithm", None) is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
