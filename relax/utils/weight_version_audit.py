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


def sample_version_report(sample: Sample) -> dict[str, Any]:
    """Build a per-sample diagnostic report that can be traced back to a turn.

    ``segments`` keeps one entry per committed generation span so a report can
    point at the exact token range, turn and weight version(s) involved.
    """
    segments: list[dict[str, Any]] = []
    known_tokens = 0
    unknown_tokens = 0
    for segment in sample.generation_segments:
        versions = [version for version in map(parse_weight_version, segment.weight_versions) if version is not None]
        if versions:
            known_tokens += segment.token_count
        else:
            unknown_tokens += segment.token_count
        segments.append(
            {
                "turn_index": segment.turn_index,
                "token_start": segment.token_start,
                "token_end": segment.token_end,
                "tokens": segment.token_count,
                "versions": versions,
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
