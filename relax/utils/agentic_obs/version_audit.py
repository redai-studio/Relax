# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Offline weight version staleness audit for agentic asynchronous training.

Joins the trajectory stream (per-span weight versions captured during
generation, ``lineage[].weight_version_spans``) with the train stream
(``consumed_weight_version`` and ``loss_mask`` stamped on consumed samples) by
``sample_export_id``, and reports the staleness metrics defined in
``docs/en/guide/weight-version-audit.md``.

The reported population is what training actually consumed: an exported
trajectory with no train record (filtered out, evaluation-only, or the run
crashed before the update was written) is reported separately as unconfirmed
and never enters the lag means or missing rates.

Fail-closed throughout: a span whose version is null or non-numeric is
explicitly unknown — it never enters a lag mean and is never treated as 0 or
as the newest version; missing versions are reported through dedicated
missing-rate metrics instead. Token-weighted metrics count only tokens the
train loss mask keeps, so masked-out samples cannot dilute the average.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
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


def _effective_token_count(span: dict[str, Any], effective_ranges: list[tuple[int, int]] | None) -> int:
    """Tokens of one span the train loss mask actually keeps.

    ``effective_ranges`` are half-open response-coordinate intervals with a
    nonzero mask value, or ``None`` when no mask evidence exists (every token
    the span covers then counts). A masked-out sample still occupies response
    coordinates, so counting its span length would credit tokens training
    never used to the lag average.
    """

    start, end = span.get("token_start", 0), span.get("token_end", 0)
    if effective_ranges is None:
        return max(end - start, 0)
    total = 0
    for range_start, range_end in effective_ranges:
        low, high = max(start, range_start), min(end, range_end)
        if high > low:
            total += high - low
    return total


def compute_sample_stats(
    spans: list[dict[str, Any]],
    consumed_weight_version: str | None,
    effective_ranges: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Per-sample staleness stats over that sample's weight version spans.

    ``spans`` are the ``weight_version_spans`` dicts of one trajectory
    (``version``/``token_start``/``token_end``) in the same coordinates as
    ``effective_ranges``. ``consumed_weight_version`` is the train-side stamp;
    when it is missing every lag is ``None`` rather than 0. Known spans are
    those whose version parses as an int; unknown spans (null or non-numeric
    version) are excluded from lags and counted in the missing rates.

    Token metrics count only tokens inside ``effective_ranges`` — the train
    loss mask — so ``effective_token_count == 0`` marks a sample that trained
    on none of its generation, and ``token_weighted_lag`` is ``None`` rather
    than 0 for it. Span-level counts stay structural: a masked-out span is
    still reported by :func:`_span_rows`.
    """

    known: list[tuple[int, str, int]] = []  # (version int, original string, effective tokens)
    known_tokens = 0
    unknown_tokens = 0
    unknown_spans = 0
    for span in spans:
        tokens = _effective_token_count(span, effective_ranges)
        version_int = _version_or_none(span.get("version"))
        if version_int is None:
            unknown_tokens += tokens
            unknown_spans += 1
        else:
            known_tokens += tokens
            known.append((version_int, span.get("version"), tokens))
    total_spans = len(spans)
    effective_tokens = known_tokens + unknown_tokens

    stats: dict[str, Any] = {
        "known_token_count": known_tokens,
        "unknown_token_count": unknown_tokens,
        "effective_token_count": effective_tokens,
        "spans_count": total_spans,
        "unknown_spans": unknown_spans,
    }
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
    stats["span_missing_rate"] = unknown_spans / total_spans if total_spans else None
    stats["token_missing_rate"] = unknown_tokens / effective_tokens if effective_tokens else None

    current = _version_or_none(consumed_weight_version)
    if known and current is not None:
        stats["oldest_lag"] = current - min_version
        stats["newest_lag"] = current - max_version
        stats["post_generation_lag"] = current - max_version
        # Zero-effective known spans (fully masked out, or token_start ==
        # token_end) leave nothing to weight: undefined, reported as None
        # rather than a division crash.
        stats["token_weighted_lag"] = (
            sum((current - version) * tokens for version, _, tokens in known) / known_tokens if known_tokens else None
        )
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

    Each row carries :func:`compute_sample_stats` output — including the
    ``spans_count`` / ``unknown_spans`` span counts — so detail rows from
    :func:`_detail_row` feed this directly. ``None`` values never enter a
    mean, median, or max: lag distributions cover only samples with a known
    consumed version, the generation-span mean covers only samples with known
    spans, and missing data is quantified by the three missing rates instead.

    ``sample_count`` counts the rows handed in, which the CLI restricts to
    samples training actually consumed. Samples that carry spans but whose loss
    mask keeps no token are counted separately in
    ``zero_effective_sample_count`` and excluded from the token-weighted lag
    mean, whose denominator is ``effective_token_count``. A row with no spans
    at all (an export predating span capture) is not zero-effective — it is
    simply unmeasured.
    """

    def values(key: str) -> list[Any]:
        return [row[key] for row in rows if row.get(key) is not None]

    oldest_lags = values("oldest_lag")
    newest_lags = values("newest_lag")
    spans_total = sum(row.get("spans_count", 0) for row in rows)
    unknown_spans = sum(row.get("unknown_spans", 0) for row in rows)
    known_tokens = sum(row.get("known_token_count", 0) for row in rows)
    unknown_tokens = sum(row.get("unknown_token_count", 0) for row in rows)
    all_unknown = [
        row for row in rows if row.get("spans_count", 0) > 0 and row.get("unknown_spans", 0) == row.get("spans_count")
    ]
    zero_effective = [
        row for row in rows if row.get("spans_count", 0) > 0 and row.get("effective_token_count", 0) == 0
    ]
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
        "effective_token_count": tokens_total,
        "zero_effective_sample_count": len(zero_effective),
        "span_missing_rate": unknown_spans / spans_total if spans_total else None,
        "token_missing_rate": unknown_tokens / tokens_total if tokens_total else None,
        "sample_missing_rate": len(all_unknown) / len(rows) if rows else None,
    }


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one parsed JSON object per non-blank line of a JSONL file.

    The writer contract (see :mod:`relax.utils.agentic_obs.writers`) allows a
    crashed process to leave a partial final line; this reader must tolerate
    that torn tail — a staleness post-mortem on a crashed run is exactly the
    audit's use case — so malformed lines are skipped with a warning instead
    of aborting the report.

    Lines are read as raw bytes and decoded inside the guard: writers emit
    ``ensure_ascii=False`` and trajectories are dominated by CJK text, so a
    random byte cut lands inside a multibyte character for many torn tails.
    Text-mode iteration would raise UnicodeDecodeError from the line iterator
    itself, before any json guard; ``json.loads`` on bytes performs the
    decode and raises the same error inside this try block.
    """

    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                yield json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("version_audit: skipping malformed line %s:%d", path, line_number)


def load_trajectories(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Index trajectory rows by integer ``sample_export_id``.

    Reads every ``trajectories/shard_*.jsonl`` file. ``sample_export_id`` is a
    decimal string since schema v2 and a JSON number before it; ``int()``
    accepts both, so legacy rows join fine — they simply predate span capture
    and show up as empty spans.
    """

    trajectories: dict[int, dict[str, Any]] = {}
    for path in sorted((run_dir / "trajectories").glob("shard_*.jsonl")):
        for row in _iter_jsonl(path):
            export_id = row.get("sample_export_id")
            if export_id is None:
                continue
            trajectories[int(export_id)] = row
    return trajectories


_TRAIN_SHARD_RE = re.compile(r"^(?:overflow_)?step_(\d+)_dp_(\d+)\.jsonl$")


def _train_shard_order(path: Path) -> tuple[int, int, int]:
    """Sort key ``(rollout_id, dp_rank, is_overflow)`` for one train shard.

    ``overflow_`` shards are separate backpressured updates for that
    ``(rollout_id, dp_rank)``, written to a differently-named file rather than
    a rewrite; ordering them last within a step is a tie-break heuristic.
    Lexicographic order is wrong here — ``overflow_...`` precedes ``step_...``
    and would let a stale stamp clobber a newer one — hence the numeric key
    parsed from the filename.
    """

    match = _TRAIN_SHARD_RE.match(path.name)
    assert match is not None, f"not a train shard name: {path.name}"
    return (int(match.group(1)), int(match.group(2)), 1 if path.name.startswith("overflow_") else 0)


@dataclass(frozen=True)
class ConsumedSample:
    """One train-stream sample: what training really consumed.

    ``weight_version`` is the actor's consumed stamp (``None`` = consumed, but
    the version was not captured — still a confirmed training sample).
    ``effective_ranges`` is the loss mask as half-open response-coordinate
    intervals, or ``None`` when the record carries no mask (older schema).
    """

    weight_version: str | None
    effective_ranges: list[tuple[int, int]] | None


def _mask_ranges(loss_mask: Any) -> list[tuple[int, int]] | None:
    """Decode a train record's loss mask into nonzero half-open intervals.

    The train stream encodes the mask as ``{"ranges": [[start, end, value]]}``
    over response coordinates; ``value`` is kept as-is (a rescaled mask) so the
    interval set only matters for whether a token trains at all. A missing or
    malformed mask is unknown evidence, not an all-zero mask.
    """

    if not isinstance(loss_mask, dict):
        return None
    ranges = loss_mask.get("ranges")
    if not isinstance(ranges, list):
        return None
    decoded = []
    for item in ranges:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return None
        start, end, value = item
        if value:
            decoded.append((int(start), int(end)))
    return decoded


def load_consumed_samples(run_dir: Path) -> dict[int, ConsumedSample]:
    """Latest train-stream evidence per sample across train DP shards.

    Presence in the returned mapping *is* the training evidence: a trajectory
    whose export id is absent was exported but never consumed (filtered,
    evaluation-only, or the update never reached disk), so it must not enter
    the training population.

    Covers both regular ``step_*_dp_*.jsonl`` shards and the
    ``overflow_step_*_dp_*.jsonl`` shards train_hooks writes under backpressure
    — the regime where staleness is most interesting. Shards are ordered by
    ``(rollout_id, dp_rank, is_overflow)``, so a sample consumed by a later
    step always overwrites the earlier stamp.
    """

    consumed: dict[int, ConsumedSample] = {}
    shards = [path for path in (run_dir / "train").glob("*step_*_dp_*.jsonl") if _TRAIN_SHARD_RE.match(path.name)]
    for path in sorted(shards, key=_train_shard_order):
        for row in _iter_jsonl(path):
            for sample in row.get("samples", []):
                export_id = sample.get("sample_export_id")
                if export_id is None:
                    continue
                consumed[int(export_id)] = ConsumedSample(
                    weight_version=sample.get("consumed_weight_version"),
                    effective_ranges=_mask_ranges(sample.get("loss_mask")),
                )
    return consumed


def _span_rows(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten lineage spans into one row per span, in response coordinates.

    Each resp node stores its spans relative to its own tokens, but the train
    loss mask indexes the whole response — resp tokens plus the interleaved
    observations ``build_sample`` keeps. Shift every span by the tokens that
    precede its node inside the response (nodes before the first resp node are
    prompt and outside it), so a span and the mask cover the same tokens.

    Only ``resp`` nodes generate tokens, so the turn counter increments on
    ``resp`` nodes; the row keeps the node's ``state_hash`` for traceback and
    its node-relative interval for reading against the lineage.
    """

    spans: list[dict[str, Any]] = []
    turn = 0
    response_offset = 0
    started = False
    for node in trajectory.get("lineage", []):
        if node.get("kind") != "resp":
            if started:
                response_offset += int(node.get("token_count", 0) or 0)
            continue
        started = True
        turn += 1
        for span in node.get("weight_version_spans", []):
            spans.append(
                {
                    "turn": turn,
                    "state_hash": node.get("state_hash"),
                    "token_start": response_offset + int(span.get("token_start", 0) or 0),
                    "token_end": response_offset + int(span.get("token_end", 0) or 0),
                    "node_token_start": span.get("token_start", 0),
                    "node_token_end": span.get("token_end", 0),
                    "version": span.get("version"),
                    "attempt_seq": span.get("attempt_seq"),
                }
            )
        response_offset += int(node.get("token_count", 0) or 0)
    return spans


def _detail_row(trajectory: dict[str, Any], consumed: ConsumedSample) -> dict[str, Any]:
    """Join one trajectory with its train record into an audit detail row.

    ``spans`` stays the full detail list (the ``--jsonl`` payload), while
    ``spans_count`` / ``unknown_spans`` come straight from
    :func:`compute_sample_stats` so downstream aggregation needs no re-parse.
    Token metrics use only the tokens ``consumed.effective_ranges`` keeps.
    """

    spans = _span_rows(trajectory)
    stats = compute_sample_stats(spans, consumed.weight_version, consumed.effective_ranges)
    return {
        "sample_export_id": trajectory.get("sample_export_id"),
        "session_id": trajectory.get("session_id"),
        "run_id": trajectory.get("run_id"),
        "consumed_weight_version": consumed.weight_version,
        "spans": spans,
        **stats,
    }


def _unconfirmed_row(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Summary of an exported trajectory with no train-stream evidence.

    Reported apart from the training population: these exports were filtered,
    evaluation-only, or never reached a written update, so their lag is not a
    training fact. Span counts still come from :func:`compute_sample_stats`
    (with no consumed stamp) so readers can see how much generation they carry.
    """

    spans = _span_rows(trajectory)
    stats = compute_sample_stats(spans, None)
    return {
        "sample_export_id": trajectory.get("sample_export_id"),
        "session_id": trajectory.get("session_id"),
        "run_id": trajectory.get("run_id"),
        "spans_count": stats["spans_count"],
        "unknown_spans": stats["unknown_spans"],
    }


def _age(span_version: str | None, consumed: str | None) -> int | None:
    """Version age at consumption; ``None`` when either side is unknown."""

    version = _version_or_none(span_version)
    current = _version_or_none(consumed)
    return current - version if version is not None and current is not None else None


def main(argv: list[str] | None = None) -> int:
    """CLI entry: audit one run directory and print the staleness report."""

    parser = argparse.ArgumentParser(
        prog="version_audit", description="Audit weight version staleness for one agentic observation run."
    )
    parser.add_argument("run_dir", help="Run directory (<dump_dir>/<run_id>) holding trajectories/ and train/")
    parser.add_argument("--session", help="Only audit samples of this session_id")
    parser.add_argument("--group", help="Only audit samples of this group_id")
    parser.add_argument(
        "--missing-only", action="store_true", help="Only report samples with unknown spans or unknown tokens"
    )
    parser.add_argument(
        "--jsonl", action="store_true", help="Emit one JSON detail row per sample instead of the report"
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        parser.error(f"run directory not found: {run_dir}")
    trajectories = load_trajectories(run_dir)
    consumed = load_consumed_samples(run_dir)

    rows: list[dict[str, Any]] = []
    unconfirmed: list[dict[str, Any]] = []
    for export_id, trajectory in sorted(trajectories.items()):
        if args.session and trajectory.get("session_id") != args.session:
            continue
        if args.group and trajectory.get("group_id") != args.group:
            continue
        train_record = consumed.get(export_id)
        if train_record is None:
            unconfirmed.append(_unconfirmed_row(trajectory))
            continue
        rows.append(_detail_row(trajectory, train_record))
    if args.missing_only:
        rows = [row for row in rows if row["unknown_spans"] > 0 or row["unknown_token_count"] > 0]
        unconfirmed = [row for row in unconfirmed if row["unknown_spans"] > 0]

    if args.jsonl:
        for row in unconfirmed:
            print(json.dumps({"confirmed": False, **row}, ensure_ascii=False))
        for row in rows:
            print(json.dumps({"confirmed": True, **row}, ensure_ascii=False))
        return 0

    summary = compute_run_summary(rows)
    summary["unconfirmed_sample_count"] = len(unconfirmed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for row in unconfirmed:
        print(
            f"sample={row['sample_export_id']} session={row['session_id']} unconfirmed=true "
            f"reason=no-train-record spans={row['spans_count']}"
        )
    for row in rows:
        for span in row["spans"]:
            age = _age(span.get("version"), row["consumed_weight_version"])
            print(
                f"sample={row['sample_export_id']} session={row['session_id']} turn={span['turn']} "
                f"attempt={span['attempt_seq']} tokens=[{span['token_start']},{span['token_end']}) "
                f"version={span['version']} age={age}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConsumedSample",
    "compute_run_summary",
    "compute_sample_stats",
    "load_consumed_samples",
    "load_trajectories",
    "main",
]
