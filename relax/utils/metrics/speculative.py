# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Batch-local accounting for exported speculative generations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from relax.utils.types import Sample

_PAIRS = (
    (
        "spec_accept_token_num",
        "spec_draft_token_num",
        "accepted_tokens",
        "proposed_tokens",
        "acceptance",
        "spec_accept_rate",
    ),
    ("completion_token_num", "spec_verify_ct", "completion_tokens", "verify_count", "length", "spec_accept_length"),
)


def compute_spec_metrics(args: Any, samples: list[Sample]) -> dict[str, int | float]:
    """Deduplicate committed generations and divide paired counter sums.

    Legacy samples have no reliable generation identity or coverage
    information; report their count without inventing a ratio.
    """
    if getattr(args, "sglang_speculative_algorithm", None) is None:
        return {}

    generations = {}
    blocks = []
    legacy_count = 0
    for sample in samples:
        records = sample.spec_generations
        if records is None:
            info = sample.spec_info
            if info.missing_fields is None:
                legacy_count += 1
            else:
                blocks.append(info.observed_counts())
            continue
        for record in records:
            key = (record["session_id"], record["request_id"])
            generations.setdefault(key, record)

    counts = blocks + list(generations.values())
    metrics: dict[str, int | float] = {
        "spec/generation_count": len(generations),
        "spec/sample_block_count": len(blocks),
        "spec/legacy_sample_count": legacy_count,
    }
    for numerator, denominator, num_name, den_name, coverage_name, metric in _PAIRS:
        paired = [record for record in counts if record[numerator] is not None and record[denominator] is not None]
        total_num = sum(record[numerator] for record in paired)
        total_den = sum(record[denominator] for record in paired)
        metrics[f"spec/{num_name}"] = total_num
        metrics[f"spec/{den_name}"] = total_den
        metrics[f"spec/{coverage_name}_observed_count"] = len(paired)
        if total_den > 0:
            metrics[metric] = total_num / total_den

    return metrics
