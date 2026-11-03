# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from typing import TYPE_CHECKING, Iterable

from relax.utils.speculative import SpeculativeCounts, SpeculativeGeneration


if TYPE_CHECKING:
    from relax.utils.types import Sample


def _counter_metrics(
    counts: list[SpeculativeCounts],
    total: int,
    prefix: str,
    *,
    coverage_known: bool = True,
) -> dict[str, int | float]:
    """Keep all known counter totals; compute ratios over complete pairs."""
    metrics: dict[str, int | float] = {}
    for numerator, denominator, ratio_name, cohort in (
        ("accepted", "proposed", "accept_rate", "accept"),
        ("completion", "verify", "tokens_per_verify", "verify"),
    ):
        pairs = [
            (getattr(item, numerator), getattr(item, denominator))
            for item in counts
            if getattr(item, numerator) is not None and getattr(item, denominator) is not None
        ]
        metrics[f"{prefix}{numerator}_total"] = sum(getattr(item, numerator) or 0 for item in counts)
        metrics[f"{prefix}{denominator}_total"] = sum(getattr(item, denominator) or 0 for item in counts)
        metrics[f"{prefix}{cohort}_covered_count"] = len(pairs)
        metrics[f"{prefix}{cohort}_uncovered_count"] = total - len(pairs)
        if total and coverage_known:
            metrics[f"{prefix}{cohort}_count_coverage"] = len(pairs) / total
        paired_denominator_total = sum(bottom for _, bottom in pairs)
        if paired_denominator_total > 0:
            metrics[f"{prefix}{ratio_name}"] = sum(top for top, _ in pairs) / paired_denominator_total
    return metrics


def _aggregate(
    samples: Iterable["Sample"],
) -> tuple[dict[str, int | float], list[SpeculativeCounts | None], bool]:
    """Decode export records once for batch deduplication and sample ratios."""
    generations: dict[tuple[str, str], SpeculativeGeneration] = {}
    conflicts: set[tuple[str, str]] = set()
    ordinary_counts: list[SpeculativeCounts] = []
    sample_counts: list[SpeculativeCounts | None] = []
    agentic_sample_count = legacy_sample_count = invalid_record_count = record_occurrence_count = 0
    has_speculative_fields = False

    for sample in samples:
        info = sample.spec_info
        has_speculative_fields |= any(
            value > 0 for value in (info.spec_accept_token_num, info.spec_draft_token_num, info.spec_verify_ct)
        )
        records = sample.spec_generations
        if records is None:
            if info.counts is not None and not info.legacy_counts and "agentic_trace" not in sample.metadata:
                counts = info.counts
                ordinary_counts.append(counts)
                has_speculative_fields |= any(
                    value is not None for value in (counts.accepted, counts.proposed, counts.verify)
                )
            else:
                legacy_sample_count += 1
                counts = info.known_counts
            sample_counts.append(counts)
            continue

        agentic_sample_count += 1
        if not isinstance(records, list):
            invalid_record_count += 1
            sample_counts.append(None)
            continue
        sample_generations: dict[tuple[str, str], SpeculativeGeneration] = {}
        sample_valid = bool(records)
        for raw_record in records:
            record_occurrence_count += 1
            record = SpeculativeGeneration.from_dict(raw_record)
            if record is None or (sample.session_id is not None and sample.session_id != record.session_id):
                invalid_record_count += 1
                sample_valid = False
                continue
            key = (record.session_id, record.generation_id)
            if key in sample_generations and sample_generations[key] != record:
                sample_valid = False
            sample_generations[key] = record
            existing = generations.get(key)
            if existing is not None and existing != record:
                conflicts.add(key)
            else:
                generations[key] = record

        counts = None
        if sample_valid:
            counts = SpeculativeCounts(0, 0, 0, 0)
            for record in sample_generations.values():
                counts = counts.plus(record.counts)
            has_speculative_fields |= any(
                value is not None for value in (counts.accepted, counts.proposed, counts.verify)
            )
        sample_counts.append(counts)

    metrics: dict[str, int | float] = {
        "spec/agentic_sample_count": agentic_sample_count,
        "spec/ordinary_sample_count": len(ordinary_counts),
        "spec/legacy_sample_count": legacy_sample_count,
        "spec/record_occurrence_count": record_occurrence_count,
        "spec/unique_generation_count": len(generations),
        "spec/conflicting_generation_count": len(conflicts),
        "spec/invalid_record_count": invalid_record_count,
    }
    metrics.update(
        _counter_metrics(
            [record.counts for key, record in generations.items() if key not in conflicts],
            len(generations),
            "spec/",
            coverage_known=invalid_record_count == 0,
        )
    )
    if ordinary_counts:
        metrics.update(_counter_metrics(ordinary_counts, len(ordinary_counts), "spec/sample/"))
    return metrics, sample_counts, has_speculative_fields


def compute_speculative_metrics(samples: Iterable["Sample"]) -> dict[str, int | float]:
    """Aggregate one exported batch, deduplicating committed generation IDs."""
    return _aggregate(samples)[0]


def compute_speculative_log_metrics(samples: list["Sample"], *, enabled: bool = False) -> dict[str, int | float]:
    """Return new batch metrics plus complete legacy aliases when provable."""
    metrics, sample_counts, has_speculative_fields = _aggregate(samples)
    if not enabled and not has_speculative_fields:
        return {}
    if not samples or metrics["spec/conflicting_generation_count"] or metrics["spec/invalid_record_count"]:
        return metrics

    for numerator, denominator, key in (
        ("accepted", "proposed", "spec_accept_rate"),
        ("completion", "verify", "spec_accept_length"),
    ):
        pairs = [
            (getattr(counts, numerator), getattr(counts, denominator))
            for counts in sample_counts
            if counts is not None
        ]
        if len(pairs) == len(sample_counts) and all(
            top is not None and bottom is not None and bottom > 0 for top, bottom in pairs
        ):
            metrics[key] = sum(top / bottom for top, bottom in pairs) / len(pairs)
    return metrics
