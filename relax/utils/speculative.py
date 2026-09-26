# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from dataclasses import dataclass
from typing import Any


SPEC_TOKEN_COUNT_KEYS = (
    ("spec_num_correct_drafts", "spec_num_proposed_drafts"),
    ("spec_accepted_drafts", "spec_proposed_drafts"),
    ("spec_accept_token_num", "spec_draft_token_num"),
)
_COUNTER_NAMES = ("accepted", "proposed", "verify", "completion")


def _counter(value: Any) -> int | None:
    """Invalid, null and absent counters are unknown, never observed zero."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    return None


@dataclass(frozen=True)
class SpeculativeCounts:
    """Complete counts for one generation; None means incomplete or unknown.

    A logical generation can span backend attempts. A field is complete only if
    every attempt supplies it. Completion is backend completion_tokens, not
    exported response length (which can also contain observations).
    """

    accepted: int | None = None
    proposed: int | None = None
    verify: int | None = None
    completion: int | None = None

    @classmethod
    def from_meta_info(cls, meta_info: dict[str, Any]) -> "SpeculativeCounts":
        accepted = proposed = None

        for accept_key, draft_key in SPEC_TOKEN_COUNT_KEYS:
            if accept_key in meta_info and draft_key in meta_info:
                accepted = _counter(meta_info[accept_key])
                proposed = _counter(meta_info[draft_key])
                break
        else:
            for accept_key, draft_key in SPEC_TOKEN_COUNT_KEYS:
                if accept_key in meta_info or draft_key in meta_info:
                    accepted = _counter(meta_info.get(accept_key))
                    proposed = _counter(meta_info.get(draft_key))
                    break

        return cls(
            accepted,
            proposed,
            _counter(meta_info.get("spec_verify_ct")),
            _counter(meta_info.get("completion_tokens")),
        )

    def plus(self, other: "SpeculativeCounts") -> "SpeculativeCounts":
        values = {}
        for name in _COUNTER_NAMES:
            left = getattr(self, name)
            right = getattr(other, name)
            values[name] = left + right if left is not None and right is not None else None
        return SpeculativeCounts(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            **{name: getattr(self, name) for name in _COUNTER_NAMES},
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "SpeculativeCounts":
        if not isinstance(data, dict) or data.get("version") != 1:
            return cls()

        return cls(**{name: _counter(data.get(name)) for name in _COUNTER_NAMES})


@dataclass(frozen=True)
class SpeculativeGeneration:
    """A committed request, distinct from its content-addressed Forest
    state."""

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
        identity = [data.get(name) for name in ("session_id", "generation_id", "state_hash")]
        if not all(isinstance(value, str) and value for value in identity):
            return None
        return cls(*identity, counts=SpeculativeCounts.from_dict(data.get("counts")))
