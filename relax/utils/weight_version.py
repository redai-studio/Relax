# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Weight-version spans and policy-provenance audit for async / agentic rollout.

Token ranges are recorded only for generation that is successfully committed into
a sample. Cancelled or replaced prefixes that never reach ``append_resp`` /
``build_sample`` are excluded automatically.

Staleness is measured in published rollout weight-version sequence numbers for
the same policy (the integer string SGLang returns), not wall-clock time and not
necessarily one-to-one with optimizer steps when ``--update-weights-interval``
is greater than 1.

Unknown / non-numeric versions stay visible as opaque strings and are excluded
from lag averages; coverage / missing-rate metrics report how much provenance
was stamped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class WeightVersionSpan:
    """Half-open token interval ``[abs_start, abs_end)`` stamped with one version."""

    version: str
    abs_start: int
    abs_end: int

    def __post_init__(self) -> None:
        if self.abs_start < 0 or self.abs_end < self.abs_start:
            raise ValueError(f"invalid weight version span: {self}")

    @property
    def token_count(self) -> int:
        return self.abs_end - self.abs_start

    def shifted(self, delta: int) -> WeightVersionSpan:
        return WeightVersionSpan(self.version, self.abs_start + delta, self.abs_end + delta)

    def clipped(self, end: int) -> WeightVersionSpan | None:
        if self.abs_start >= end:
            return None
        if self.abs_end <= end:
            return self
        clipped_end = min(self.abs_end, end)
        if self.abs_start >= clipped_end:
            return None
        return WeightVersionSpan(self.version, self.abs_start, clipped_end)


@dataclass
class WeightVersionsPerCall:
    """Weight provenance for one successful generation call / agentic turn."""

    spans: list[WeightVersionSpan] = field(default_factory=list)

    def to_dicts(self) -> list[dict[str, str | int]]:
        return [asdict(span) for span in self.spans]

    @staticmethod
    def from_dicts(data: Sequence[dict[str, Any]]) -> WeightVersionsPerCall:
        return WeightVersionsPerCall(
            spans=[
                WeightVersionSpan(
                    version=str(span["version"]),
                    abs_start=int(span["abs_start"]),
                    abs_end=int(span["abs_end"]),
                )
                for span in data
            ]
        )

    def shifted(self, delta: int) -> WeightVersionsPerCall:
        return WeightVersionsPerCall(spans=[span.shifted(delta) for span in self.spans])

    def clipped(self, end: int) -> WeightVersionsPerCall:
        spans = [clipped for span in self.spans if (clipped := span.clipped(end)) is not None]
        return WeightVersionsPerCall(spans=spans)

    @staticmethod
    def from_meta_info(meta_info: dict[str, Any], *, output_end: int) -> WeightVersionsPerCall:
        """Build absolute spans for one generate attempt ending at ``output_end``.

        Prefers engine ``weight_versions`` (output-relative). Falls back to a
        single scalar ``weight_version`` covering the attempt's output tokens.
        Missing provenance yields an empty call so turn counts stay aligned.
        """
        num_output_tokens = _num_output_tokens(meta_info)
        output_relative = _output_relative_spans(meta_info, num_output_tokens=num_output_tokens)
        if output_relative is None:
            return WeightVersionsPerCall(spans=[])
        output_start = output_end - num_output_tokens
        return WeightVersionsPerCall(
            spans=[
                WeightVersionSpan(version=version, abs_start=output_start + start, abs_end=output_start + end)
                for version, start, end in output_relative
                if start < end
            ]
        )


def append_spans_from_meta_info(
    spans: list[WeightVersionSpan],
    *,
    meta_info: dict[str, Any],
    pending_len_before: int,
    num_new_tokens: int,
) -> None:
    """Append attempt-local spans onto an in-flight request's pending list.

    Coordinates are relative to the start of ``pending_token_delta`` (0-based).
    Empty token attempts (e.g. abort with no new tokens) leave spans unchanged
    so previously accepted prefixes keep their provenance.
    """
    if num_new_tokens <= 0:
        return
    relative = _output_relative_spans(meta_info, num_output_tokens=num_new_tokens)
    if relative is None:
        return
    for version, start, end in relative:
        if start >= end:
            continue
        spans.append(
            WeightVersionSpan(
                version=version,
                abs_start=pending_len_before + start,
                abs_end=pending_len_before + end,
            )
        )


def coalesce_spans(spans: Iterable[WeightVersionSpan]) -> list[WeightVersionSpan]:
    """Merge adjacent same-version spans; keep gaps for missing provenance."""
    merged: list[WeightVersionSpan] = []
    for span in spans:
        if span.token_count <= 0:
            continue
        if merged and merged[-1].version == span.version and merged[-1].abs_end == span.abs_start:
            prev = merged[-1]
            merged[-1] = WeightVersionSpan(prev.version, prev.abs_start, span.abs_end)
        else:
            merged.append(span)
    return merged


def parse_numeric_version(version: str | int | None) -> int | None:
    if version is None:
        return None
    text = str(version)
    return int(text) if text.isdigit() else None


@dataclass(frozen=True)
class WeightVersionStats:
    sample_count: int
    versioned_sample_count: int
    versioned_token_count: int
    token_version_sum: int
    oldest_version: int | None
    newest_version: int | None
    unknown_span_count: int
    unknown_token_count: int

    @property
    def generation_version_span(self) -> int | None:
        if self.oldest_version is None or self.newest_version is None:
            return None
        return self.newest_version - self.oldest_version

    def oldest_lag(self, current_version: int | None) -> int | None:
        if self.oldest_version is None or current_version is None:
            return None
        return current_version - self.oldest_version

    def newest_lag(self, current_version: int | None) -> int | None:
        if self.newest_version is None or current_version is None:
            return None
        return current_version - self.newest_version

    def token_weighted_lag(self, current_version: int | None) -> float | None:
        if current_version is None or self.versioned_token_count == 0:
            return None
        return current_version - self.token_version_sum / self.versioned_token_count

    @property
    def sample_coverage(self) -> float | None:
        if self.sample_count == 0:
            return None
        return self.versioned_sample_count / self.sample_count

    @property
    def sample_missing_rate(self) -> float | None:
        coverage = self.sample_coverage
        return None if coverage is None else 1.0 - coverage


def sample_weight_version_stats(sample: Any) -> WeightVersionStats:
    return aggregate_weight_version_stats([sample])


def aggregate_weight_version_stats(samples: Iterable[Any]) -> WeightVersionStats:
    sample_count = 0
    versioned_sample_count = 0
    versioned_token_count = 0
    token_version_sum = 0
    oldest_version: int | None = None
    newest_version: int | None = None
    unknown_span_count = 0
    unknown_token_count = 0

    for sample in samples:
        sample_count += 1
        sample_has_numeric = False
        for span in iter_sample_weight_spans(sample):
            numeric = parse_numeric_version(span.version)
            if numeric is None:
                unknown_span_count += 1
                unknown_token_count += span.token_count
                continue
            versioned_token_count += span.token_count
            token_version_sum += numeric * span.token_count
            oldest_version = numeric if oldest_version is None else min(oldest_version, numeric)
            newest_version = numeric if newest_version is None else max(newest_version, numeric)
            sample_has_numeric = True
        versioned_sample_count += int(sample_has_numeric)

    return WeightVersionStats(
        sample_count=sample_count,
        versioned_sample_count=versioned_sample_count,
        versioned_token_count=versioned_token_count,
        token_version_sum=token_version_sum,
        oldest_version=oldest_version,
        newest_version=newest_version,
        unknown_span_count=unknown_span_count,
        unknown_token_count=unknown_token_count,
    )


def compute_policy_provenance_metrics(
    samples: Sequence[Any],
    *,
    current_version: int | None = None,
    group_by_index: bool = True,
) -> dict[str, float]:
    """Side-channel provenance metrics for samples selected into training.

    Does not change sampling or GRPO group filtering. Missing numeric provenance
    is excluded from lag averages and reported via coverage / missing_rate.
    """
    if not samples:
        return {}

    if group_by_index:
        groups = _group_samples(samples)
        group_stats = [aggregate_weight_version_stats(group) for group in groups]
        flat_stats = aggregate_weight_version_stats(samples)
    else:
        group_stats = [sample_weight_version_stats(sample) for sample in samples]
        flat_stats = aggregate_weight_version_stats(samples)

    resolved_current = current_version
    if resolved_current is None:
        # Fallback: newest stamped numeric version among selected samples.
        # Documents a lower bound on consume-time lag when the logger has no
        # engine version handle; true train-time lag may be higher.
        resolved_current = flat_stats.newest_version

    metrics: dict[str, float] = {}
    oldest_lags = [lag for stats in group_stats if (lag := stats.oldest_lag(resolved_current)) is not None]
    newest_lags = [lag for stats in group_stats if (lag := stats.newest_lag(resolved_current)) is not None]
    version_spans = [span for stats in group_stats if (span := stats.generation_version_span) is not None]

    if oldest_lags:
        metrics["avg_staleness"] = sum(oldest_lags) / len(oldest_lags)
        metrics["max_staleness"] = float(max(oldest_lags))
    if newest_lags:
        metrics["avg_post_generation_staleness"] = sum(newest_lags) / len(newest_lags)
        metrics["max_post_generation_staleness"] = float(max(newest_lags))
    if version_spans:
        metrics["avg_generation_version_span"] = sum(version_spans) / len(version_spans)
        metrics["max_generation_version_span"] = float(max(version_spans))

    token_weighted = flat_stats.token_weighted_lag(resolved_current)
    if token_weighted is not None:
        metrics["token_weighted_staleness"] = float(token_weighted)
    if flat_stats.sample_count:
        metrics["weight_version_sample_coverage"] = float(flat_stats.sample_coverage or 0.0)
        metrics["weight_version_missing_rate"] = float(flat_stats.sample_missing_rate or 0.0)
    if flat_stats.unknown_span_count:
        metrics["unknown_version_span_count"] = float(flat_stats.unknown_span_count)
        metrics["unknown_version_token_count"] = float(flat_stats.unknown_token_count)
    return metrics


def build_weight_version_diagnostic(
    sample: Any,
    *,
    current_version: int | None = None,
) -> dict[str, Any]:
    """Per-sample report tracing Session / turn / generation-segment provenance."""
    metadata = getattr(sample, "metadata", None) or {}
    trace = metadata.get("agentic_trace") if isinstance(metadata, dict) else None
    trace = trace if isinstance(trace, dict) else {}
    calls = list(getattr(sample, "weight_versions", None) or [])
    turns = trace.get("turns") if isinstance(trace.get("turns"), list) else []

    segments: list[dict[str, Any]] = []
    for call_idx, call in enumerate(calls):
        turn = turns[call_idx] if call_idx < len(turns) and isinstance(turns[call_idx], dict) else {}
        call_spans = getattr(call, "spans", None)
        if call_spans is None and isinstance(call, dict):
            call_spans = WeightVersionsPerCall.from_dicts(call.get("spans", [])).spans
        for span in call_spans or []:
            numeric = parse_numeric_version(span.version)
            segments.append(
                {
                    "session_id": getattr(sample, "session_id", None) or trace.get("session_id"),
                    "sample_index": getattr(sample, "index", None),
                    "group_index": getattr(sample, "group_index", None),
                    "turn_idx": turn.get("turn_idx", call_idx),
                    "request_id": turn.get("request_id"),
                    "resp_state_hash": turn.get("resp_state_hash"),
                    "rollout_id": turn.get("rollout_id"),
                    "abort_count": turn.get("abort_count"),
                    "call_idx": call_idx,
                    "version": span.version,
                    "numeric_version": numeric,
                    "abs_start": span.abs_start,
                    "abs_end": span.abs_end,
                    "token_count": span.token_count,
                    "consume_lag": None if numeric is None or current_version is None else current_version - numeric,
                    "unknown_version": numeric is None,
                }
            )

    stats = sample_weight_version_stats(sample)
    resolved_current = current_version if current_version is not None else stats.newest_version
    return {
        "session_id": getattr(sample, "session_id", None) or trace.get("session_id"),
        "sample_index": getattr(sample, "index", None),
        "group_index": getattr(sample, "group_index", None),
        "leaf_state_hash": trace.get("leaf_state_hash"),
        "turn_count": trace.get("turn_count", len(calls)),
        "generation_call_count": len(calls),
        "current_version": resolved_current,
        "oldest_version": stats.oldest_version,
        "newest_version": stats.newest_version,
        "generation_version_span": stats.generation_version_span,
        "post_generation_staleness": stats.newest_lag(resolved_current),
        "staleness": stats.oldest_lag(resolved_current),
        "token_weighted_staleness": stats.token_weighted_lag(resolved_current),
        "weight_version_sample_coverage": 1.0 if stats.versioned_sample_count else 0.0,
        "unknown_version_span_count": stats.unknown_span_count,
        "segments": segments,
    }


def iter_sample_weight_spans(sample: Any) -> Iterable[WeightVersionSpan]:
    for call in getattr(sample, "weight_versions", None) or []:
        if isinstance(call, WeightVersionsPerCall):
            yield from call.spans
        elif isinstance(call, WeightVersionSpan):
            yield call
        elif isinstance(call, dict) and "abs_start" in call:
            yield WeightVersionSpan(str(call["version"]), int(call["abs_start"]), int(call["abs_end"]))
        elif isinstance(call, dict) and "spans" in call:
            yield from WeightVersionsPerCall.from_dicts(call["spans"]).spans
        elif isinstance(call, list):
            yield from WeightVersionsPerCall.from_dicts(call).spans
        # Legacy scalar strings are intentionally not inventing a token range.


def encode_weight_versions_for_train(sample: Any) -> list[list[dict[str, str | int]]]:
    encoded: list[list[dict[str, str | int]]] = []
    for call in getattr(sample, "weight_versions", None) or []:
        if isinstance(call, WeightVersionsPerCall):
            encoded.append(call.to_dicts())
        elif isinstance(call, list):
            encoded.append(list(call))
        else:
            encoded.append([])
    return encoded


def _num_output_tokens(meta_info: dict[str, Any]) -> int:
    logprobs = meta_info.get("output_token_logprobs")
    if logprobs is not None:
        return len(logprobs)
    completion = meta_info.get("completion_tokens")
    if completion is not None:
        return int(completion)
    return 0


def _output_relative_spans(
    meta_info: dict[str, Any],
    *,
    num_output_tokens: int,
) -> list[tuple[str, int, int]] | None:
    if (raw_spans := meta_info.get("weight_versions")) is not None:
        spans: list[tuple[str, int, int]] = []
        for span in raw_spans:
            version = str(span["version"])
            start = int(span["start"])
            end = int(span["end"])
            if end > num_output_tokens > 0:
                raise ValueError(
                    f"weight version spans {raw_spans} extend past the {num_output_tokens} output tokens"
                )
            spans.append((version, start, end))
        return spans
    if meta_info.get("weight_version") is not None:
        if num_output_tokens <= 0:
            return []
        return [(str(meta_info["weight_version"]), 0, num_output_tokens)]
    return None


def _group_samples(samples: Sequence[Any]) -> list[list[Any]]:
    grouped: dict[Any, list[Any]] = {}
    order: list[Any] = []
    for sample in samples:
        key = getattr(sample, "group_index", None)
        if key is None:
            key = id(sample)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(sample)
    return [grouped[key] for key in order]
