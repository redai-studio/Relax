# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded append-only history and pagination over a fixed sequence
snapshot."""

import base64
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from .models import MAX_UINT, ROLES, EventDraft, valid_id, valid_uint


MAX_CAPACITY = 4096
MAX_PAGE_SIZE = 100
SCAN_BUDGET = 512
MAX_CURSOR_BYTES = 1024
STORE_LOCK_TIMEOUT = 0.05


@dataclass(frozen=True, slots=True)
class _Record:
    draft: EventDraft
    sequence: int
    timestamp: int


class FaultEventStore:
    """Keep history and its deduplication index within the same capacity."""

    def __init__(self, run_id: str, capacity: int = MAX_CAPACITY) -> None:
        if not valid_id(run_id):
            raise ValueError("INVALID_RUN_ID")
        if type(capacity) is not int or not 1 <= capacity <= MAX_CAPACITY:
            raise ValueError("INVALID_CAPACITY")
        self.run_id = run_id
        self.capacity = capacity
        self._ring: list[_Record | None] = [None] * capacity
        self._ids: dict[str, _Record] = {}
        self._latest = 0
        self._available = True
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        """Permanently disable this history after an uncertain commit."""
        self._available = False

    def commit_status(self, draft: EventDraft) -> str:
        """Inspect the retained ID before a consumer settles a delivery loss.

        This does not append or retain tombstones for evicted records. It is a
        consumer-side operation and must not run under the admission lock.
        """
        if type(draft) is not EventDraft:
            raise ValueError("INVALID_EVENT")
        draft.validate()
        if draft.run_id != self.run_id:
            raise ValueError("RUN_MISMATCH")
        with self._guard():
            previous = self._ids.get(draft.event_id)
            if previous is None:
                return "absent"
            if self._ring[(previous.sequence - 1) % self.capacity] is not previous:
                self.invalidate()
                raise RuntimeError("UNAVAILABLE")
            return "duplicate" if previous.draft == draft else "conflict"

    @contextmanager
    def _guard(self) -> Iterator[None]:
        if not self._lock.acquire(timeout=STORE_LOCK_TIMEOUT):
            raise TimeoutError("BUSY")
        try:
            if not self._available:
                raise RuntimeError("UNAVAILABLE")
            yield
        finally:
            self._lock.release()

    def append(self, draft: EventDraft) -> str:
        """Atomically append, or recognize a retained duplicate or conflict."""
        if type(draft) is not EventDraft:
            raise ValueError("INVALID_EVENT")
        draft.validate()
        if draft.run_id != self.run_id:
            raise ValueError("RUN_MISMATCH")
        timestamp = time.time_ns() // 1_000_000
        with self._guard():
            old = self._ids.get(draft.event_id)
            if old is not None:
                if self._ring[(old.sequence - 1) % self.capacity] is not old:
                    self.invalidate()
                    raise RuntimeError("UNAVAILABLE")
                if old.draft != draft:
                    raise ValueError("CONFLICT")
                return "duplicate"
            if not valid_uint(timestamp):
                raise ValueError("INVALID_TIME")
            if self._latest == MAX_UINT:
                raise RuntimeError("SEQUENCE_LIMIT")
            sequence = self._latest + 1
            item = _Record(draft, sequence, timestamp)
            slot = (sequence - 1) % self.capacity
            try:
                previous = self._ring[slot]
                if previous is not None:
                    if self._ids.get(previous.draft.event_id) is not previous:
                        raise RuntimeError("UNAVAILABLE")
                    del self._ids[previous.draft.event_id]
                self._ring[slot] = item
                self._ids[draft.event_id] = item
                self._latest = sequence
            except Exception:
                self.invalidate()
                raise RuntimeError("UNAVAILABLE") from None
            return "accepted"

    def _decode_cursor(self, cursor: str, role: str | None, fault_id: str | None) -> tuple[int, int]:
        try:
            if type(cursor) is not str or not 0 < len(cursor) <= MAX_CURSOR_BYTES:
                raise ValueError
            payload = json.loads(base64.b64decode(cursor.encode("ascii"), altchars=b"-_", validate=True))
            if type(payload) is not list or len(payload) != 6 or type(payload[0]) is not int or payload[0] != 1:
                raise ValueError
            run, saved_role, saved_fault, after, end = payload[1:]
            if not valid_id(run) or not valid_uint(after) or not valid_uint(end):
                raise ValueError
            if saved_role is not None and (type(saved_role) is not str or saved_role not in ROLES):
                raise ValueError
            if saved_fault is not None and not valid_id(saved_fault):
                raise ValueError
            if saved_role != role or saved_fault != fault_id or after > end:
                raise ValueError
        except Exception:
            raise ValueError("INVALID_QUERY") from None
        if run != self.run_id:
            raise RuntimeError("RUN_MISMATCH")
        return after, end

    def query(
        self, role: str | None = None, fault_id: str | None = None, cursor: str | None = None, limit: int = 50
    ) -> dict[str, object]:
        """Return at most one bounded page; empty pages may still continue."""
        if type(limit) is not int or not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValueError("INVALID_QUERY")
        if role is not None and (type(role) is not str or len(role) > 10 or role not in ROLES):
            raise ValueError("INVALID_QUERY")
        if fault_id is not None and not valid_id(fault_id):
            raise ValueError("INVALID_QUERY")
        bounds = None if cursor is None else self._decode_cursor(cursor, role, fault_id)
        with self._guard():
            after, end = (0, self._latest) if bounds is None else bounds
            if end > self._latest:
                raise ValueError("INVALID_QUERY")
            oldest = max(1, self._latest - self.capacity + 1)
            missing_end = min(end, oldest - 1)
            missing = [after + 1, missing_end] if missing_end > after else None
            position = min(end, max(after, oldest - 1))
            items: list[_Record] = []
            for sequence in range(position + 1, min(end, position + SCAN_BUDGET) + 1):
                item = self._ring[(sequence - 1) % self.capacity]
                if item is None or item.sequence != sequence or self._ids.get(item.draft.event_id) is not item:
                    self.invalidate()
                    raise RuntimeError("UNAVAILABLE")
                position = sequence
                if role is not None and item.draft.role != role:
                    continue
                if fault_id is not None and item.draft.fault_id != fault_id:
                    continue
                items.append(item)
                if len(items) == limit:
                    break
            done = position == end
            truncated = self._latest > self.capacity

        # Copy only immutable references while holding the Store lock; all JSON
        # projection and cursor serialization happen after releasing it.
        next_cursor = None
        if not done:
            payload = json.dumps([1, self.run_id, role, fault_id, position, end], separators=(",", ":"))
            next_cursor = base64.urlsafe_b64encode(payload.encode("ascii")).decode("ascii")
        return {
            "run_id": self.run_id,
            "events": [item.draft.record(item.sequence, item.timestamp) for item in items],
            "snapshot_end_seq": end,
            "oldest_available_seq": oldest,
            "next_cursor": next_cursor,
            "done": done,
            "retention_truncated": truncated,
            "missing_range": missing,
            "gap_scope": "global",
            "filter_match_unknown": missing is not None and (role is not None or fault_id is not None),
        }

    def status(self) -> dict[str, object]:
        with self._guard():
            return {
                "run_id": self.run_id,
                "capacity": self.capacity,
                "event_count": len(self._ids),
                "latest_sequence": self._latest,
                "retention_truncated": self._latest > self.capacity,
            }
