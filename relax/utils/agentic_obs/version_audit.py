# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Offline weight version staleness audit for agentic asynchronous training.

Joins the trajectory stream (per-span weight versions captured during
generation, ``lineage[].weight_version_spans``) with the train stream
(``consumed_weight_version`` stamped on consumed samples) by
``sample_export_id``, and reports the staleness metrics defined in
``docs/superpowers/specs/2026-11-10-weight-version-tracking-design.md`` §6.

Fail-closed throughout: a span whose version is null or non-numeric is
explicitly unknown — it never enters a lag mean and is never treated as 0 or
as the newest version; missing versions are reported through dedicated
missing-rate metrics instead.
"""

from __future__ import annotations

import statistics
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _version_or_none(version: str | None) -> int | None:
    """Numeric version strings parse to ints; anything else is unknown.

    ``"7"`` is the actor publication ordinal carried in the JSONL streams.
    ``None`` means the engine did not report a version; a non-numeric string
    (or any non-string) is equally unknown — this audit never guesses.
    """

    if not isinstance(version, str):
        return None
    try:
        return int(version)
    except ValueError:
        return None


def compute_sample_stats(spans: list[dict[str, Any]], consumed_weight_version: str | None) -> dict[str, Any]:
    """Per-sample staleness stats over that sample's weight version spans.

    ``spans`` are the ``weight_version_spans`` dicts of one trajectory
    (``version``/``token_start``/``token_end``). ``consumed_weight_version``
    is the train-side stamp; when it is missing every lag is ``None`` rather
    than 0. Known spans are those whose version parses as an int; unknown
    spans (null or non-numeric version) are excluded from lags and counted in
    the missing rates.
    """

    known: list[tuple[int, str, int]] = []  # (version int, original string, token count)
    known_tokens = 0
    unknown_tokens = 0
    for span in spans:
        tokens = max(span.get("token_end", 0) - span.get("token_start", 0), 0)
        version_int = _version_or_none(span.get("version"))
        if version_int is None:
            unknown_tokens += tokens
        else:
            known_tokens += tokens
            known.append((version_int, span.get("version"), tokens))
    total_spans = len(spans)
    total_tokens = known_tokens + unknown_tokens

    stats: dict[str, Any] = {"known_token_count": known_tokens, "unknown_token_count": unknown_tokens}
    if known:
        min_version = min(item[0] for item in known)
        max_version = max(item[0] for item in known)
        stats["oldest_version"] = next(item[1] for item in known if item[0] == min_version)
        stats["newest_version"] = next(item[1] for item in known if item[0] == max_version)
        stats["generation_span"] = max_version - min_version
    else:
        stats["oldest_version"] = None
        stats["newest_version"] = None
        stats["generation_span"] = None
    stats["span_missing_rate"] = (total_spans - len(known)) / total_spans if total_spans else None
    stats["token_missing_rate"] = unknown_tokens / total_tokens if total_tokens else None

    current = _version_or_none(consumed_weight_version)
    if known and current is not None:
        stats["oldest_lag"] = current - min_version
        stats["newest_lag"] = current - max_version
        stats["post_generation_lag"] = current - max_version
        stats["token_weighted_lag"] = sum((current - version) * tokens for version, _, tokens in known) / known_tokens
    else:
        stats["oldest_lag"] = None
        stats["newest_lag"] = None
        stats["post_generation_lag"] = None
        stats["token_weighted_lag"] = None
    return stats


def _mean(values: list[float]) -> float | None:
    """Arithmetic mean; ``None`` for an empty list so absence stays absence."""

    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    """Median via :mod:`statistics`; ``None`` for an empty list."""

    return statistics.median(values) if values else None


def compute_run_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run-level aggregation over per-sample stat rows.

    Each row carries :func:`compute_sample_stats` output plus ``spans`` /
    ``unknown_spans`` span counts. ``None`` values never enter a mean, median,
    or max: lag distributions cover only samples with a known consumed
    version, the generation-span mean covers only samples with known spans,
    and missing data is quantified by the three missing rates instead.
    """

    def values(key: str) -> list[Any]:
        return [row[key] for row in rows if row.get(key) is not None]

    oldest_lags = values("oldest_lag")
    newest_lags = values("newest_lag")
    spans_total = sum(row.get("spans", 0) for row in rows)
    unknown_spans = sum(row.get("unknown_spans", 0) for row in rows)
    known_tokens = sum(row.get("known_token_count", 0) for row in rows)
    unknown_tokens = sum(row.get("unknown_token_count", 0) for row in rows)
    all_unknown = [row for row in rows if row.get("spans", 0) > 0 and row.get("unknown_spans", 0) == row.get("spans")]
    tokens_total = known_tokens + unknown_tokens
    return {
        "sample_count": len(rows),
        "lag_sample_count": len(oldest_lags),
        "oldest_lag_mean": _mean(oldest_lags),
        "oldest_lag_median": _median(oldest_lags),
        "oldest_lag_max": max(oldest_lags) if oldest_lags else None,
        "newest_lag_mean": _mean(newest_lags),
        "newest_lag_median": _median(newest_lags),
        "newest_lag_max": max(newest_lags) if newest_lags else None,
        "token_weighted_lag_mean": _mean(values("token_weighted_lag")),
        "generation_span_mean": _mean(values("generation_span")),
        "span_missing_rate": unknown_spans / spans_total if spans_total else None,
        "token_missing_rate": unknown_tokens / tokens_total if tokens_total else None,
        "sample_missing_rate": len(all_unknown) / len(rows) if rows else None,
    }


__all__ = ["compute_run_summary", "compute_sample_stats"]
