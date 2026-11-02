# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fixed, immutable event fields for the driver-owned fault history."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from uuid import uuid4


MAX_UINT = (1 << 64) - 1
MAX_EVENT_BYTES = 1024
ROLES = frozenset(("actor", "critic", "rollout", "advantages", "advantage", "reference", "actor_fwd", "sft", "genrm"))
REASONS = MappingProxyType(
    {
        "fault_detected": ("reported_error", "heartbeat_timeout", "fatal_error"),
        "fatal_escalated": ("fatal_error",),
        "handling_started": (None,),
        "recovery_handoff": (None,),
        "recovery_succeeded": (None,),
        "recovery_failed": ("restart_exception", "handoff_exception", "step_restore_failed", "operation_failed"),
        "recovery_interrupted": ("superseded", "source_retired"),
        "handling_skipped": ("role_unregistered",),
        "termination_requested": ("fatal_error", "restart_limit"),
    }
)
_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_REQUIRED_FIELDS = frozenset(("run_id", "fault_id", "event_id", "role", "source_generation", "event_type"))
_DRAFT_FIELDS = _REQUIRED_FIELDS | {
    "reason_code",
    "recovery_id",
    "recovery_mode",
    "expected_evidence",
    "observed_evidence",
    "failed_evidence",
}
_PLANNED_EVENTS = frozenset(
    ("handling_started", "recovery_handoff", "recovery_succeeded", "recovery_failed", "recovery_interrupted")
)


def new_id() -> str:
    """Generate an internal UUID without retaining caller-provided text."""
    return uuid4().hex


def valid_id(value: object) -> bool:
    return type(value) is str and len(value) == 32 and _ID_PATTERN.fullmatch(value) is not None


def valid_uint(value: object, maximum: int = MAX_UINT) -> bool:
    return type(value) is int and 0 <= value <= maximum


def encode(value: dict[str, object]) -> bytes:
    """Encode already validated fields without arbitrary object conversion."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


@dataclass(frozen=True, slots=True)
class EventDraft:
    """An event before the Store assigns its sequence and recording time."""

    run_id: str
    fault_id: str
    event_id: str
    role: str
    source_generation: int
    event_type: str
    reason_code: str | None = None
    recovery_id: str | None = None
    recovery_mode: str | None = None
    expected_evidence: int = 0
    observed_evidence: int = 0
    failed_evidence: int = 0

    @classmethod
    def from_dict(cls, payload: object) -> EventDraft:
        """Reject unknown fields with a fixed error, without echoing input."""
        if type(payload) is not dict or len(payload) > len(_DRAFT_FIELDS):
            raise ValueError("INVALID_EVENT")
        if any(type(key) is not str for key in payload):
            raise ValueError("INVALID_EVENT")
        if not _REQUIRED_FIELDS <= payload.keys() or payload.keys() - _DRAFT_FIELDS:
            raise ValueError("INVALID_EVENT")
        draft = cls(**payload)
        draft.validate()
        return draft

    def validate(self) -> None:
        """Validate types, combinations and the final uint64 encoding bound."""
        if type(self) is not EventDraft:
            raise ValueError("INVALID_EVENT")
        if type(self.role) is not str or len(self.role) > 10 or self.role not in ROLES:
            raise ValueError("INVALID_EVENT")
        if type(self.event_type) is not str or len(self.event_type) > 21 or self.event_type not in REASONS:
            raise ValueError("INVALID_EVENT")
        if self.reason_code is not None and type(self.reason_code) is not str:
            raise ValueError("INVALID_EVENT")
        if self.reason_code not in REASONS[self.event_type]:
            raise ValueError("INVALID_EVENT")
        if self.recovery_mode is not None and type(self.recovery_mode) is not str:
            raise ValueError("INVALID_EVENT")
        if not all(valid_id(value) for value in (self.run_id, self.fault_id, self.event_id)):
            raise ValueError("INVALID_EVENT")
        if not valid_uint(self.source_generation):
            raise ValueError("INVALID_EVENT")
        masks = (self.expected_evidence, self.observed_evidence, self.failed_evidence)
        if not all(valid_uint(mask, 63) for mask in masks):
            raise ValueError("INVALID_EVENT")
        if (self.observed_evidence | self.failed_evidence) & ~self.expected_evidence:
            raise ValueError("INVALID_EVENT")
        if self.observed_evidence & self.failed_evidence:
            raise ValueError("INVALID_EVENT")

        budget_exhausted = self.event_type == "termination_requested" and self.reason_code == "restart_limit"
        if self.event_type in _PLANNED_EVENTS or budget_exhausted:
            if (
                not valid_id(self.recovery_id)
                or self.recovery_mode not in ("local", "global")
                or not self.expected_evidence
            ):
                raise ValueError("INVALID_EVENT")
            if budget_exhausted and self.recovery_mode != "global":
                raise ValueError("INVALID_EVENT")
        elif self.event_type == "handling_skipped":
            if not valid_id(self.recovery_id) or self.recovery_mode is not None or self.expected_evidence:
                raise ValueError("INVALID_EVENT")
        elif self.recovery_id is not None or self.recovery_mode is not None or self.expected_evidence:
            raise ValueError("INVALID_EVENT")

        if self.event_type == "recovery_succeeded" and (
            self.observed_evidence != self.expected_evidence or self.failed_evidence
        ):
            raise ValueError("INVALID_EVENT")
        if len(encode(self.record(MAX_UINT, MAX_UINT))) > MAX_EVENT_BYTES:
            raise ValueError("EVENT_TOO_LARGE")

    def record(self, sequence: int, timestamp: int) -> dict[str, object]:
        """Project a validated draft into a fresh, complete query record."""
        if not valid_uint(sequence) or sequence == 0 or not valid_uint(timestamp):
            raise ValueError("INVALID_RECORD")
        scope = None
        if self.event_type == "recovery_handoff":
            scope = "control_flow_handoff"
        elif self.event_type in ("recovery_succeeded", "recovery_failed", "recovery_interrupted"):
            scope = "recovery_operations"
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "fault_id": self.fault_id,
            "event_id": self.event_id,
            "source_generation": self.source_generation,
            "role": self.role,
            "event_type": self.event_type,
            "reason_code": self.reason_code,
            "recovery_id": self.recovery_id,
            "recovery_mode": self.recovery_mode,
            "expected_evidence": self.expected_evidence,
            "observed_evidence": self.observed_evidence,
            "failed_evidence": self.failed_evidence,
            "sequence": sequence,
            "recorded_at_ms": timestamp,
            "result_scope": scope,
            "availability": "operations_completed" if self.event_type == "recovery_succeeded" else "unverified",
        }
