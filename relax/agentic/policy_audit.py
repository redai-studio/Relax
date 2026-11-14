# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Policy provenance for rows consumed by fully asynchronous Agentic
training."""

import json
from argparse import Namespace
from typing import Any

from relax.agentic.profile import TRACE_KEY
from relax.utils.types import Sample


POLICY_AUDIT_FIELD = "agentic_policy_audit"


def policy_audit_enabled(args: Namespace | None) -> bool:
    return (
        getattr(args, "fully_async", False)
        and getattr(args, "use_agentic_rollout", False)
        and not getattr(args, "hybrid", False)
        and not getattr(args, "debug_train_only", False)
        and getattr(args, "custom_convert_samples_to_train_data_path", None) is None
    )


def encode_policy_audit(sample: Sample, loss_mask: list[int]) -> str:
    trace = sample.metadata.get(TRACE_KEY) or {}
    fragments = []
    for turn in trace.get("turns", []):
        for start, end, version in turn.get("weight_version_spans", []):
            fragments.append(
                {
                    "turn_idx": turn["turn_idx"],
                    "request_id": turn.get("request_id"),
                    "resp_state_hash": turn["resp_state_hash"],
                    "start": start,
                    "end": end,
                    "weight_version": version,
                    "training_tokens": sum(bool(mask) for mask in loss_mask[start:end]),
                }
            )
    return json.dumps(
        {
            "session_id": trace.get("session_id", sample.session_id),
            "leaf_state_hash": trace.get("leaf_state_hash"),
            "sample_index": sample.index,
            "training_tokens": sum(bool(mask) for mask in loss_mask),
            "fragments": fragments,
        },
        separators=(",", ":"),
    )


def audit_consumed_rows(
    rows: list[str], published_versions: dict[str, int]
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, int]]:
    """Produce row diagnostics and DP-reducible means with their
    denominators."""
    reports = []
    totals = dict(
        generation_version_span=0,
        post_generation_version_lag=0,
        token_weighted_version_lag=0,
        missing_version_fraction=0,
    )
    counts = dict.fromkeys(totals, 0)
    consume = len(published_versions) or None
    for row in rows:
        report = json.loads(row)
        reports.append(report)
        report["consume_version_ordinal"] = consume
        fragments = report["fragments"]
        known_ordinals = []
        known_tokens = 0
        age_sum = 0
        for fragment in fragments:
            ordinal = published_versions.get(fragment["weight_version"])
            age = consume - ordinal if consume is not None and ordinal is not None else None
            fragment["version_ordinal"] = ordinal
            fragment["age_at_consumption"] = age
            if age is not None and fragment["training_tokens"]:
                known_ordinals.append(ordinal)
                known_tokens += fragment["training_tokens"]
                age_sum += age * fragment["training_tokens"]

        training_tokens = report["training_tokens"]
        missing = training_tokens - known_tokens
        span = max(known_ordinals) - min(known_ordinals) if known_ordinals and missing == 0 else None
        wait = fragments[-1]["age_at_consumption"] if fragments and training_tokens else None
        report["generation_version_span"] = span
        report["post_generation_version_lag"] = wait
        report["token_weighted_version_lag"] = age_sum / known_tokens if known_tokens else None
        report["missing_version_fraction"] = missing / training_tokens if training_tokens else None

        if training_tokens:
            totals["missing_version_fraction"] += missing
            counts["missing_version_fraction"] += training_tokens
            if wait is not None:
                totals["post_generation_version_lag"] += wait
                counts["post_generation_version_lag"] += 1
        if span is not None:
            totals["generation_version_span"] += span
            counts["generation_version_span"] += 1
        totals["token_weighted_version_lag"] += age_sum
        counts["token_weighted_version_lag"] += known_tokens

    return reports, {key: totals[key] / counts[key] if counts[key] else 0.0 for key in totals}, counts
