# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import logging
import math
from typing import Any, Literal

import numpy as np
import torch

from relax.algorithms.spec import get_algorithm
from relax.utils.repetition import detect_repetition
from relax.utils.types import SPEC_COUNTER_KEYS, Sample, normalize_spec_counter_values


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


def _record_spec_node(node_counts: dict[str, dict[str, int]], node_key: str, counters: dict[str, int]) -> None:
    """Record one submitted generation node under its own identity.

    Counters belong to the submitted node, not to the sample that exports it,
    so a node exported by several samples (``A -> B`` and ``A -> C``) is
    recorded once instead of being added again per sample. An occurrence that
    carries no counters never overwrites counters another occurrence reported.
    Partial copies can supply missing fields without adding duplicate work.
    """
    record = node_counts.setdefault(node_key, {})
    for key, value in normalize_spec_counter_values(counters).items():
        record.setdefault(key, value)


def compute_spec_decoding_metrics(samples: list[Sample]) -> dict[str, float]:
    """Aggregate speculative decoding counters of one rollout metric batch.

    Counters are deduplicated per submitted generation node and summed before any
    rate is derived, so a node shared by several exported samples (``A -> B`` and
    ``A -> C`` both export ``A``) counts once, and a batch whose samples carry very
    different counter magnitudes is weighted by counters instead of averaging ratios.

    A sample without node identities -- a non-agentic rollout, or a dump written
    before node identities were exported -- falls back to its own totals. Totals of
    a legacy dump carry no report flag, so they count only when a draft or verify
    counter is non-zero. ``spec_legacy_samples`` counts legacy payloads whose
    generation identities cannot be recovered. Each ratio uses only records with
    both of its counters available. Missing and null fields remain unknown even
    after serialization, so they cannot become a zero numerator.

    Returns an empty mapping when the batch carries no submitted generation node.

    Keys (``compute_metrics_from_samples`` logs them verbatim):
        ``spec_accept_rate``: accepted drafts over proposed drafts, summed per node.
        ``spec_accept_length``: completion tokens over verify calls, summed over the
        nodes with at least one verify call.
        ``spec_nodes_total``: counted generation nodes, shared nodes counted once.
        ``spec_nodes_missing_counts``: nodes missing at least one of the four counters.
        ``spec_coverage``: share of counted nodes with all four counters available.
        ``spec_accept_rate_coverage``: share with accepted and proposed available.
        ``spec_accept_length_coverage``: share with completion and verify available.
        ``spec_legacy_samples``: only when any sample had to fall back to its totals.
    """
    node_counts: dict[str, dict[str, int]] = {}
    sample_records: list[dict[str, int]] = []
    legacy_samples = 0

    for sample in samples:
        info = sample.spec_info
        if info.nodes:
            for node_key, counters in info.nodes.items():
                _record_spec_node(node_counts, node_key, counters)
            continue
        # One sample owns these totals, so no cross-sample deduplication applies.
        counters = info.counter_values()
        if info.counts_reported is None:
            # A legacy dump has no report flag. ``completion_token_num`` is filled for
            # every response, so only a draft or verify counter proves a report.
            if not (counters.get("spec_draft_token_num", 0) > 0 or counters.get("spec_verify_ct", 0) > 0):
                counters = {}
            legacy_samples += 1
        sample_records.append(counters)

    records = [*node_counts.values(), *sample_records]
    if not records:
        return {}

    counters_reported = sum(all(key in counters for key in SPEC_COUNTER_KEYS) for counters in records)
    nodes_total = len(records)
    # Each ratio sums only the records that carry its own denominator, so completion
    # tokens of a record without verify steps cannot inflate the accept length.
    accepted = proposed = completion = verify = 0
    acceptance_covered = length_covered = 0
    for counters in records:
        if "spec_accept_token_num" in counters and "spec_draft_token_num" in counters:
            acceptance_covered += 1
            accepted += counters["spec_accept_token_num"]
            proposed += counters["spec_draft_token_num"]
        if "completion_token_num" in counters and "spec_verify_ct" in counters:
            length_covered += 1
            if counters["spec_verify_ct"] > 0:
                completion += counters["completion_token_num"]
                verify += counters["spec_verify_ct"]

    metrics: dict[str, float] = {}
    if proposed > 0:
        metrics["spec_accept_rate"] = accepted / proposed
    if verify > 0:
        metrics["spec_accept_length"] = completion / verify
    metrics["spec_nodes_total"] = float(nodes_total)
    metrics["spec_nodes_missing_counts"] = float(nodes_total - counters_reported)
    metrics["spec_coverage"] = counters_reported / nodes_total
    metrics["spec_accept_rate_coverage"] = acceptance_covered / nodes_total
    metrics["spec_accept_length_coverage"] = length_covered / nodes_total
    if legacy_samples:
        metrics["spec_legacy_samples"] = float(legacy_samples)
    return metrics


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
