# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from dataclasses import dataclass
from numbers import Integral
from typing import Any


SPEC_TOKEN_COUNT_KEYS = (
    ("spec_num_correct_drafts", "spec_num_proposed_drafts"),
    ("spec_accepted_drafts", "spec_proposed_drafts"),
    ("spec_accept_token_num", "spec_draft_token_num"),
)
_COUNTER_NAMES = ("accepted", "proposed", "verify", "completion")


def _counter(value: Any) -> int | None:
    """Normalize a backend counter; ``None`` means the backend did not provide
    it."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        value = int(value)
        return value if value >= 0 else None
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    return None


@dataclass(frozen=True)
class SpeculativeCounts:
    """Counters for one backend generation.

    ``None`` preserves an unavailable field while ``0`` preserves an explicit
    zero returned by the backend.  This distinction is required for coverage
    metrics and prevents missing data from becoming a false 0% result.
    """

    accepted: int | None = None
    proposed: int | None = None
    verify: int | None = None
    completion: int | None = None

    @classmethod
    def from_meta_info(cls, meta_info: dict[str, Any] | None) -> "SpeculativeCounts":
        if not isinstance(meta_info, dict):
            return cls()

        accepted = proposed = None
        selected_pair = next(
            (
                (accepted_key, proposed_key)
                for accepted_key, proposed_key in SPEC_TOKEN_COUNT_KEYS
                if accepted_key in meta_info and proposed_key in meta_info
            ),
            None,
        )
        if selected_pair is None:
            selected_pair = next(
                (
                    (accepted_key, proposed_key)
                    for accepted_key, proposed_key in SPEC_TOKEN_COUNT_KEYS
                    if accepted_key in meta_info or proposed_key in meta_info
                ),
                None,
            )
        if selected_pair is not None:
            accepted_key, proposed_key = selected_pair
            accepted = _counter(meta_info.get(accepted_key))
            proposed = _counter(meta_info.get(proposed_key))
        return cls(
            accepted=accepted,
            proposed=proposed,
            verify=_counter(meta_info.get("spec_verify_ct")),
            completion=_counter(meta_info.get("completion_tokens")),
        )

    def plus(self, other: "SpeculativeCounts") -> "SpeculativeCounts":
        values: dict[str, int | None] = {}
        for name in _COUNTER_NAMES:
            left = getattr(self, name)
            right = getattr(other, name)
            values[name] = left + right if left is not None and right is not None else None
        return SpeculativeCounts(**values)

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, **{name: getattr(self, name) for name in _COUNTER_NAMES}}

    @classmethod
    def from_dict(cls, data: Any) -> "SpeculativeCounts":
        if not isinstance(data, dict) or data.get("version") != 1:
            return cls()
        return cls(**{name: _counter(data.get(name)) for name in _COUNTER_NAMES})


@dataclass(frozen=True)
class SpeculativeGeneration:
    """A committed generation identity and its backend counters."""

    session_id: str
    generation_id: str
    state_hash: str
    counts: SpeculativeCounts

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "session_id": self.session_id,
            "generation_id": self.generation_id,
            "state_hash": self.state_hash,
            "counts": self.counts.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "SpeculativeGeneration | None":
        if not isinstance(data, dict) or data.get("version") != 1:
            return None
        identity = tuple(data.get(name) for name in ("session_id", "generation_id", "state_hash"))
        if not all(isinstance(value, str) and value for value in identity):
            return None
        return cls(*identity, counts=SpeculativeCounts.from_dict(data.get("counts")))
