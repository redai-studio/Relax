# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded failure-event timeline used by Relax fault recovery."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class FailureEvent:
    """One immutable observation in a fault lifecycle."""

    fault_id: str
    role: str
    phase: str
    occurred_at_ms: int
    action: Optional[str] = None
    reason: Optional[str] = None
    attempt: Optional[int] = None
    step: Optional[int] = None

    @property
    def event_id(self) -> str:
        """Stable identity used to deduplicate repeated reports."""
        attempt = 0 if self.attempt is None else self.attempt
        return f"{self.fault_id}:{self.phase}:{attempt}"


@dataclass(frozen=True)
class RecordedFailureEvent:
    """Failure event after the store assigns its sequence."""

    seq: int
    event: FailureEvent

    def to_dict(self) -> dict:
        result = asdict(self.event)
        result["event_id"] = self.event.event_id
        result["seq"] = self.seq
        return result


class FailureEventStore:
    """Bounded append-only failure-event timeline."""

    def __init__(self, capacity: int = 4096):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")

        self._events: deque[RecordedFailureEvent] = deque(maxlen=capacity)

    def append(self, event: FailureEvent) -> dict:
        """Append an event unless it is already in the retained timeline."""
        for recorded in self._events:
            if recorded.event.event_id == event.event_id:
                return {"seq": recorded.seq, "appended": False}

        seq = self._events[-1].seq + 1 if self._events else 1
        self._events.append(
            RecordedFailureEvent(
                seq=seq,
                event=event,
            )
        )

        return {"seq": seq, "appended": True}

    def query(
        self,
        *,
        role: Optional[str] = None,
        fault_id: Optional[str] = None,
        after_seq: int = 0,
        limit: int = 100,
    ) -> dict:
        """Return filtered events in store order."""
        if after_seq < 0:
            raise ValueError(f"after_seq must be >= 0, got {after_seq}")
        if limit <= 0:
            raise ValueError(f"limit must be > 0, got {limit}")

        oldest_seq = self._events[0].seq if self._events else None
        latest_seq = self._events[-1].seq if self._events else None

        events = []
        last_scanned_seq = after_seq

        for recorded in self._events:
            if recorded.seq <= after_seq:
                continue

            last_scanned_seq = recorded.seq

            if role is not None and recorded.event.role != role:
                continue
            if fault_id is not None and recorded.event.fault_id != fault_id:
                continue

            events.append(recorded.to_dict())
            if len(events) >= limit:
                break

        next_after_seq = last_scanned_seq if latest_seq is not None and last_scanned_seq < latest_seq else None

        history_truncated = oldest_seq is not None and after_seq + 1 < oldest_seq

        return {
            "events": events,
            "next_after_seq": next_after_seq,
            "oldest_seq": oldest_seq,
            "latest_seq": latest_seq,
            "history_truncated": history_truncated,
        }
