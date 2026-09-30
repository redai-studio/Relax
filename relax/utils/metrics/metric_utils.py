# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import logging
import math
from typing import Any, Literal

import numpy as np
import torch

from relax.algorithms.spec import get_algorithm
from relax.utils.repetition import detect_repetition
from relax.utils.types import Sample


logger = logging.getLogger(__name__)


def dict_add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def compute_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
):
    if group_size == 1:
        return {}

    if num_groups is None:
        # Caller doesn't know the group structure (e.g. streaming consumer whose
        # per-DP slice may not be group-aligned).  Derive whole groups and drop
        # any ragged remainder instead of asserting, so a non-multiple count
        # never crashes the caller.  An empty result means nothing to report.
        num_groups = len(flat_rewards) // group_size
        if num_groups == 0:
            logger.warning(
                "compute_pass_rate: %d rewards < group_size %d; skipping pass@k",
                len(flat_rewards),
                group_size,
            )
            return {}
        usable = num_groups * group_size
        if usable != len(flat_rewards):
            logger.warning(
                "compute_pass_rate: %d rewards not a multiple of group_size %d; truncating to %d for pass@k",
                len(flat_rewards),
                group_size,
                usable,
            )
            flat_rewards = flat_rewards[:usable]

    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    log_dict = {}
    for k in pass_rate_name_list:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        num_samples = np.full(num_groups, group_size)

        pass_k_estimates = _estimate_pass_at_k(num_samples, num_correct, k)

        pass_k = np.mean(pass_k_estimates)
        log_dict[f"pass@{k}"] = pass_k

    return log_dict


def _estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n, c, k):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct, strict=False)])


def compute_statistics(values: list[float]) -> dict[str, float]:
    values = np.array(values)
    return {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }


def is_rollout_numeric_metric_value(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def append_rollout_numeric_metric_values(metric_values: dict[str, list[float]], *, key: str, value) -> None:
    if isinstance(value, (list, tuple)):
        flattened = [float(item) for item in value if is_rollout_numeric_metric_value(item)]
        if flattened:
            metric_values.setdefault(key, []).extend(flattened)
        return
    if is_rollout_numeric_metric_value(value):
        metric_values.setdefault(key, []).append(float(value))


def finalize_rollout_explicit_metric_values(metric_values: dict[str, list[float]]) -> dict[str, float]:
    log_dict: dict[str, float] = {}
    for metric_name, values in metric_values.items():
        if values:
            log_dict |= dict_add_prefix(compute_statistics(values), f"{metric_name}/")
    return log_dict


def compute_num_turn_metrics(samples: list[Sample]) -> dict[str, float]:
    """Pure aggregation of per-sample rollout turn counts.

    Turns are read from ``sample.metadata["rollout_turns"]`` and normalized
    (see ``_normalize_rollout_turns``): non-negative integers are kept, finite
    non-negative integral floats are converted to int, and every other value —
    including missing or None metadata — falls back to the historical
    single-turn default of 1. ``num_turn/mean|max|min`` therefore keep their
    semantics for normal inputs, and the added tail percentiles surface
    long-tail trajectories.

    The four percentiles come from a single ``np.percentile`` call with the
    default linear interpolation; single-sample batches report the same Python
    scalar for all seven statistics.

    Returns {} for empty input; reads only the samples it is given.
    """
    if not samples:
        return {}
    turns = [_normalize_rollout_turns(sample) for sample in samples]
    quantiles = np.percentile(turns, [50, 90, 95, 99])
    metrics = {
        "num_turn/mean": np.mean(turns).item(),
        "num_turn/max": np.max(turns).item(),
        "num_turn/min": np.min(turns).item(),
    }
    metrics |= {f"num_turn/p{p}": q.item() for p, q in zip((50, 90, 95, 99), quantiles)}
    return metrics


def compute_stop_reason_metrics(samples: list[Sample]) -> dict[str, float]:
    """Pure aggregation of the stop-reason distribution over rollout samples.

    Each sample lands in exactly one bucket resolved by
    ``_resolve_rollout_stop_reason`` — the legacy ``rollout_stop_reason``
    metadata field first, the agentic ``stop_reason`` metadata field second,
    and an explicit ``unknown`` fallback — so the emitted
    ``stop_reason/{reason}/ratio`` values sum to 1 over the whole batch. The
    denominator is every input sample: no filtering by reward, status, group,
    or session. Counts are reported alongside as float integers for
    small-batch debugging, and only categories present in the current batch
    are emitted (nothing is accumulated across calls).

    Keys embed the raw reason strings (e.g. ``stop_reason/max_turns/count``).
    Relax ships explicit metric dicts with no key-substring-based reduction,
    so reasons containing "max"/"min" are safe here, unlike aggregators that
    infer the reduction from key names.

    Returns {} for empty input; reads only the samples it is given.
    """
    if not samples:
        return {}
    counts: dict[str, int] = {}
    for sample in samples:
        reason = _resolve_rollout_stop_reason(sample)
        counts[reason] = counts.get(reason, 0) + 1
    total = len(samples)
    metrics: dict[str, float] = {}
    for reason in sorted(counts):
        count = counts[reason]
        metrics[f"stop_reason/{reason}/count"] = float(count)
        metrics[f"stop_reason/{reason}/ratio"] = count / total
    return metrics


def _resolve_rollout_stop_reason(sample: Sample) -> str:
    """Resolve the stop-reason bucket for one sample (first valid wins).

    1. ``metadata["rollout_stop_reason"]`` — legacy field written by the
       DeepEyes rollout (e.g. ``max_turns`` / ``env_done``).
    2. ``metadata["stop_reason"]`` — field returned by agentic apps
       (e.g. ``examples/deepeyes_agentic/app/agent.py``) and merged into
       sample metadata by the session service.
    3. ``"unknown"`` when neither field holds a valid value.

    Valid values are strings with non-blank content; surrounding whitespace is
    stripped while case and inner content are preserved. The stop reason is
    never derived from ``Sample.status`` or per-request finish reasons.
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    for key in ("rollout_stop_reason", "stop_reason"):
        reason = metadata.get(key)
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
    return "unknown"


def _normalize_rollout_turns(sample: Sample) -> int:
    """Normalize ``metadata["rollout_turns"]`` to a committed-response count.

    Non-negative Python/NumPy integers are kept as-is (0 is valid), finite non-
    negative integral floats (e.g. ``2.0``) are converted to int, and every
    other value — None, bools, strings, fractional floats, negatives, NaN/inf,
    containers, or missing/None metadata — falls back to the historical single-
    turn default of 1 without parsing or truncation.
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else None
    value = metadata.get("rollout_turns") if metadata else None
    if value is None or isinstance(value, (bool, str, list, tuple, dict)):
        return 1
    if isinstance(value, (int, np.integer)):
        return int(value) if value >= 0 else 1
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value >= 0 and float(value).is_integer():
        return int(value)
    return 1


def _compute_rloo_group_diagnostics(args, samples: list[Sample]) -> dict[str, float]:
    """Compute leave-one-out diagnostics from training rollout samples.

    Gated on the algorithm registry's ``reward_normalizer`` rather than on the
    estimator's name: what makes these numbers meaningful is that the reward
    stage produced a leave-one-out baseline, so any algorithm declaring that
    normalizer gets them. The ``rloo/`` key prefix stays as-is -- those metric
    names are already published to dashboards.

    These keys are returned with an ``rloo/`` prefix. The training rollout
    logger adds the outer ``rollout/`` prefix before publishing them.

    These are purely observational rollout statistics and do not feed back into
    the training path. ``no_signal_frac`` here means effective loss tokens whose
    RLOO advantage is exactly zero (the leave-one-out baseline is zero whenever
    all rewards in a group are equal); it is not a zero-token training-step
    concept. ``empty_response_frac`` intentionally uses the literal response
    length and therefore remains distinct from a sample whose response exists
    but is fully masked.
    """
    spec = get_algorithm(getattr(args, "advantage_estimator", None) or "grpo")
    if spec.reward_normalizer != "group_leave_one_out":
        return {}
    if (
        getattr(args, "custom_reward_post_process_path", None) is not None
        or getattr(args, "agentic_custom_advantage_path", None) is not None
    ):
        # These hooks can replace the advantages consumed by training. Raw
        # rewards are insufficient to reconstruct that custom signal here, so
        # suppress the standard RLOO diagnostics instead of publishing values
        # that may disagree with the optimizer input.
        return {}

    from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards

    group_size = args.n_samples_per_prompt
    groups: dict[int, list[tuple[Sample, float]]] = {}
    for sample in samples:
        group_index = sample.group_index
        if group_index is None:
            continue
        groups.setdefault(group_index, []).append((sample, sample.get_reward_value(args)))

    baseline_means: list[float] = []
    advantage_abs_means: list[float] = []
    zero_adv_groups = 0
    complete_groups = 0
    dropped_groups = 0
    zero_adv_tokens = 0
    effective_response_tokens = 0

    for items in groups.values():
        if len(items) != group_size:
            dropped_groups += 1
            continue
        complete_groups += 1
        rewards = torch.tensor([reward for _, reward in items], dtype=torch.float32)
        advantages = compute_rloo_leave_one_out_rewards(rewards)
        baselines = rewards - advantages
        baseline_means.append(baselines.mean().item())
        advantage_abs_means.append(advantages.abs().mean().item())
        if bool(advantages.abs().max().item() < 1e-12):
            zero_adv_groups += 1
        for (sample, _), advantage in zip(items, advantages.tolist(), strict=True):
            effective_length = sample.effective_response_length or 0
            effective_response_tokens += effective_length
            if abs(advantage) < 1e-12:
                zero_adv_tokens += effective_length

    total_samples = len(samples)
    observed_groups = complete_groups + dropped_groups
    empty_responses = sum((sample.response_length or 0) == 0 for sample in samples)
    return {
        "rloo/baseline_mean": sum(baseline_means) / len(baseline_means) if baseline_means else 0.0,
        "rloo/adv_abs_mean": sum(advantage_abs_means) / len(advantage_abs_means) if advantage_abs_means else 0.0,
        "rloo/no_signal_frac": (zero_adv_tokens / effective_response_tokens if effective_response_tokens > 0 else 0.0),
        "rloo/empty_response_frac": empty_responses / total_samples if total_samples > 0 else 0.0,
        "rloo/zero_adv_group_frac": zero_adv_groups / complete_groups if complete_groups > 0 else 0.0,
        "rloo/dropped_group_frac": dropped_groups / observed_groups if observed_groups > 0 else 0.0,
    }


def compute_rollout_reward_metrics(
    args,
    samples: list[Sample],
    *,
    include_rloo_diagnostics: bool = True,
) -> dict[str, float]:
    reward_values = [sample.get_reward_value(args) for sample in samples if sample.reward is not None]
    log_dict = {"raw_reward": np.mean(reward_values).item()} if reward_values else {}
    reward_metric_values: dict[str, list[float]] = {}
    primary_reward_key = getattr(args, "reward_key", None)
    for sample in samples:
        reward = sample.reward
        if not isinstance(reward, dict):
            continue
        for key, value in reward.items():
            if (
                not isinstance(key, str)
                or not key
                or key == primary_reward_key
                or key == "raw_reward"
                or key.startswith("_")
            ):
                continue
            append_rollout_numeric_metric_values(reward_metric_values, key=key, value=value)
    log_dict |= finalize_rollout_explicit_metric_values(reward_metric_values)
    if args.log_passrate and reward_values:
        log_dict |= dict_add_prefix(
            compute_pass_rate(flat_rewards=reward_values, group_size=args.n_samples_per_prompt),
            "passrate/",
        )

    if include_rloo_diagnostics:
        log_dict |= _compute_rloo_group_diagnostics(args, samples)

    return log_dict


compute_rollout_explicit_reward_metrics = compute_rollout_reward_metrics


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    if isinstance(data, str):
        raw = data.encode(encoding)
    else:
        raw = data

    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    if algorithm == "zlib":
        import zlib

        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def has_repetition(text: str) -> bool:
    return detect_repetition(text, stop_after_first_hit=True).has_repetition


def compute_rollout_step(args, rollout_id):
    if args.wandb_always_use_train_step:
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id
