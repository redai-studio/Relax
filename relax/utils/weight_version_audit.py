# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Side-channel audit of which policy version produced which trained tokens.

This module never participates in sampling, advantage estimation or the loss.
It only reads :attr:`relax.utils.types.Sample.generation_segments`, which is
populated on the rollout side, and reports what is known about the versions.
Anything unknown stays unknown: a missing or unparsable version is reported as
missing, never coerced to version 0 or to the newest published version.
"""

from typing import Any

from relax.utils.types import Sample


def parse_weight_version(value: str | None) -> int | None:
    """Return the monotonic integer behind a published weight version.

    Returns ``None`` when the version is absent or not an integer, so callers
    can count it as unknown instead of guessing.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def trained_token_count(loss_mask: list[int] | None, token_start: int, token_end: int) -> int:
    """Count response tokens inside ``[token_start, token_end)`` that are trained."""
    if loss_mask is None:
        return max(token_end - token_start, 0)
    end = min(token_end, len(loss_mask))
    start = min(max(token_start, 0), end)
    return int(sum(loss_mask[start:end]))


def sample_version_report(sample: Sample) -> dict[str, Any]:
    """Build a per-sample diagnostic report that can be traced back to a turn.

    ``segments`` keeps one entry per committed generation span, listing the
    exact token range of every weight version involved, so a report can point at
    the session, turn and tokens instead of a summary number.
    """
    segments: list[dict[str, Any]] = []
    known_tokens = 0
    unknown_tokens = 0
    for segment in sample.generation_segments:
        spans: list[dict[str, Any]] = []
        segment_versions: list[int] = []
        segment_known = 0
        segment_unknown = 0
        for span in segment.version_spans:
            tokens = trained_token_count(sample.loss_mask, span.token_start, span.token_end)
            version = parse_weight_version(span.version)
            spans.append(
                {
                    "version": version,
                    "token_start": span.token_start,
                    "token_end": span.token_end,
                    "tokens": tokens,
                }
            )
            if tokens == 0:
                continue
            if version is None:
                segment_unknown += tokens
            else:
                segment_known += tokens
                segment_versions.append(version)
        known_tokens += segment_known
        unknown_tokens += segment_unknown
        segments.append(
            {
                "turn_index": segment.turn_index,
                "token_start": segment.token_start,
                "token_end": segment.token_end,
                "tokens": segment_known + segment_unknown,
                "versions": segment_versions,
                "spans": spans,
                "state_hash": segment.state_hash,
            }
        )

    response_tokens = sample.effective_response_length
    unknown_tokens += max(response_tokens - known_tokens - unknown_tokens, 0)
    return {
        "session_id": sample.session_id,
        "group_index": sample.group_index,
        "index": sample.index,
        "response_tokens": response_tokens,
        "known_tokens": known_tokens,
        "unknown_tokens": unknown_tokens,
        "segments": segments,
    }
