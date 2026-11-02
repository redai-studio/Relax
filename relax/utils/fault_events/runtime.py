# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Driver-owned, bounded fault event admission and consumption."""

import threading
from collections import deque
from typing import TYPE_CHECKING

from relax.utils.logging_utils import get_logger

from .models import MAX_UINT, EventDraft, new_id
from .store import FaultEventStore


if TYPE_CHECKING:
    from .query_server import FaultQueryServer


logger = get_logger(__name__)
_ADMISSION_TIMEOUT = 0.005
_QUERY_LOCK_TIMEOUT = 0.05
_UNKNOWN_REASONS = (
    "SOURCE_RETIRED",
    "STORE_COMMIT_UNCERTAIN",
    "OUTCOME_UNOBSERVED",
    "CONTEXT_CAPACITY",
    "OBSERVATION_ERROR",
    "RECEIPT_CONFLICT",
)
_UNKNOWN_BITS = {reason: 1 << index for index, reason in enumerate(_UNKNOWN_REASONS)}


class FaultEventRuntime:
    """A bounded FIFO with one consumer; business calls never access the
    Store."""

    def __init__(self, capacity: int = 4096, queue_capacity: int = 512, *, run_id: str | None = None):
        if type(queue_capacity) is not int or not 1 <= queue_capacity <= 512:
            raise ValueError("INVALID_CAPACITY")
        self.run_id = new_id() if run_id is None else run_id
        self.store = FaultEventStore(self.run_id, capacity)
        self.queue_capacity = queue_capacity
        self.query_server: FaultQueryServer | None = None
        self._queue: deque[EventDraft] = deque()
        self._pending: dict[str, EventDraft] = {}
        self._lock = threading.Lock()
        self._diagnostic_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._disabled = threading.Event()
        self._store_unavailable = threading.Event()
        self._query_closed = False
        self._rejections = 0
        self._lost = 0
        self._unknown = 0
        self._inflight = 0
        self._consumer = threading.Thread(target=self._consume, name="fault-events", daemon=True)
        try:
            self._consumer.start()
        except Exception:
            self._stop.set()
            self._wake.set()
            raise

    def _increment(self, name: str, amount: int = 1) -> None:
        # This lock only protects fixed integers, never admission, Store, user
        # code or I/O. A rejected admission must not reacquire the busy lock.
        with self._diagnostic_lock:
            if name == "_rejections":
                self._rejections = min(MAX_UINT, self._rejections + amount)
            elif name == "_lost":
                self._lost = min(MAX_UINT, self._lost + amount)

    def note_unknown(self, reason: str = "OBSERVATION_ERROR") -> None:
        """Retain only a fixed reason bit, never exception text or
        identities."""
        if type(reason) is not str or reason not in _UNKNOWN_BITS:
            reason = "OBSERVATION_ERROR"
        with self._diagnostic_lock:
            self._unknown |= _UNKNOWN_BITS[reason]

    def _enqueue(self, draft: EventDraft) -> bool:
        """Called with the admission lock held and a validated immutable
        draft."""
        if self._stop.is_set() or self._disabled.is_set():
            self._increment("_rejections")
            return False
        pending = self._pending.get(draft.event_id)
        if pending is not None:
            if pending == draft:
                return True
            self._increment("_rejections")
            self.note_unknown()
            return False
        if len(self._queue) >= self.queue_capacity:
            self._increment("_rejections")
            return False
        self._pending[draft.event_id] = draft
        try:
            self._queue.append(draft)
        except Exception:
            self._pending.pop(draft.event_id, None)
            raise
        self._wake.set()
        return True

    def record(self, draft: EventDraft) -> bool:
        """Return local acceptance, not a commitment or a recovery outcome."""
        try:
            if type(draft) is not EventDraft:
                self._increment("_rejections")
                return False
            draft.validate()
            if draft.run_id != self.run_id:
                self._increment("_rejections")
                return False
        except ValueError:
            self._increment("_rejections")
            return False
        except Exception:
            self._increment("_rejections")
            self.note_unknown()
            return False

        acquired = False
        try:
            acquired = self._lock.acquire(timeout=_ADMISSION_TIMEOUT)
            if not acquired:
                self._increment("_rejections")
                return False
            return self._enqueue(draft)
        except Exception:
            self._increment("_rejections")
            self.note_unknown()
            return False
        finally:
            if acquired:
                self._lock.release()

    def diagnostics(self) -> dict:
        """Copy fixed diagnostic state without waiting on a Store operation."""
        if not self._lock.acquire(timeout=_QUERY_LOCK_TIMEOUT):
            raise TimeoutError("BUSY")
        try:
            queued, inflight = len(self._queue), self._inflight
        finally:
            self._lock.release()
        if not self._diagnostic_lock.acquire(timeout=_QUERY_LOCK_TIMEOUT):
            raise TimeoutError("BUSY")
        try:
            rejections, lost, unknown = self._rejections, self._lost, self._unknown
        finally:
            self._diagnostic_lock.release()
        query_available = False
        if not self._stop.is_set() and not self._store_unavailable.is_set() and self.query_server is not None:
            try:
                query_available = self.query_server.available is True
            except Exception:
                pass
        return {
            "admission_rejections": rejections,
            "known_lost_events": lost,
            "coverage_unknown": bool(unknown),
            "unknown_reasons": sorted(reason for reason, bit in _UNKNOWN_BITS.items() if unknown & bit),
            "queued": queued,
            "inflight": inflight,
            "queue_capacity": self.queue_capacity,
            "query_available": query_available,
        }

    def query(self, **kwargs) -> dict:
        if self._stop.is_set() or self._store_unavailable.is_set():
            raise RuntimeError("UNAVAILABLE")
        try:
            page = self.store.query(**kwargs)
        except RuntimeError as exc:
            if type(exc) is RuntimeError and exc.args == ("UNAVAILABLE",):
                self._invalidate_store()
            raise
        return {**page, **self.diagnostics()}

    def status(self) -> dict:
        if self._stop.is_set() or self._store_unavailable.is_set():
            raise RuntimeError("UNAVAILABLE")
        try:
            status = self.store.status()
        except RuntimeError as exc:
            if type(exc) is RuntimeError and exc.args == ("UNAVAILABLE",):
                self._invalidate_store()
            raise
        return {**status, **self.diagnostics()}

    @staticmethod
    def _definitely_uncommitted(exc: Exception) -> bool:
        if len(exc.args) != 1 or type(exc.args[0]) is not str:
            return False
        if type(exc) is TimeoutError:
            return exc.args[0] == "BUSY"
        if type(exc) is ValueError:
            return exc.args[0] in ("INVALID_EVENT", "RUN_MISMATCH", "CONFLICT", "INVALID_TIME")
        return type(exc) is RuntimeError and exc.args[0] == "SEQUENCE_LIMIT"

    def _invalidate_store(self) -> None:
        self._disabled.set()
        self._store_unavailable.set()
        self._wake.set()
        self.note_unknown("STORE_COMMIT_UNCERTAIN")
        try:
            self.store.invalidate()
        except Exception:
            # Runtime query/admission remain disabled even if invalidation
            # itself fails. Keep the original Store; never replace its history.
            pass

    def _submit(self, draft: EventDraft) -> None:
        for attempt in range(2):
            if self._stop.is_set():
                self._settle_abandoned(draft)
                return
            try:
                result = self.store.append(draft)
                if type(result) is not str or result not in ("accepted", "duplicate"):
                    self._invalidate_store()
                return
            except Exception as exc:
                if not self._definitely_uncommitted(exc):
                    self._invalidate_store()
                    return
                if type(exc) is RuntimeError:
                    self._disabled.set()
                if type(exc) is TimeoutError and attempt == 0:
                    continue
                self._settle_abandoned(draft)
                return

    def _settle_abandoned(self, draft: EventDraft) -> None:
        # A failed append attempt may repeat an already retained fact. Verify
        # it on the consumer, outside admission, before claiming a known loss.
        try:
            outcome = self.store.commit_status(draft)
        except Exception as exc:
            self.note_unknown("STORE_COMMIT_UNCERTAIN")
            if type(exc) is RuntimeError and exc.args == ("UNAVAILABLE",):
                self._invalidate_store()
            return
        if type(outcome) is not str:
            self._invalidate_store()
        elif outcome == "absent":
            self._increment("_lost")
        elif outcome == "conflict":
            self.note_unknown()
        elif outcome != "duplicate":
            self._invalidate_store()

    def _consume(self) -> None:
        try:
            while True:
                self._wake.wait(0.05)
                if not self._lock.acquire(timeout=_ADMISSION_TIMEOUT):
                    continue
                try:
                    if self._stop.is_set() or self._disabled.is_set():
                        return
                    draft = self._queue.popleft() if self._queue else None
                    if not self._queue:
                        self._wake.clear()
                    self._inflight = int(draft is not None)
                finally:
                    self._lock.release()
                if draft is None:
                    continue
                try:
                    self._submit(draft)
                finally:
                    with self._lock:
                        self._inflight = 0
                        self._pending.pop(draft.event_id, None)
        except Exception:
            # An unexpected consumer failure cannot produce a successful empty
            # query. Disable its Store even when the last commit is uncertain.
            self._invalidate_store()
        finally:
            # Only the consumer settles accepted entries. close() cannot
            # double-count an append that is still in flight.
            with self._lock:
                remaining = tuple(self._queue)
                self._queue.clear()
                self._pending.clear()
                self._inflight = 0
            for draft in remaining:
                self._settle_abandoned(draft)

    def close(self) -> None:
        """Signal shutdown without joining threads or waiting for queue
        drain."""
        self._stop.set()
        self._wake.set()
        if not self._close_lock.acquire(blocking=False):
            return
        try:
            if not self._query_closed:
                self._query_closed = True
                if self.query_server is not None:
                    try:
                        self.query_server.close()
                    except Exception:
                        self.note_unknown()
        finally:
            self._close_lock.release()


def _warn(message: str) -> None:
    try:
        logger.warning(message)
    except Exception:
        pass


def start_fault_events(*, enabled: bool, health_enabled: bool, port: int = 0) -> FaultEventRuntime | None:
    """Start the optional driver runtime; observation failures stay
    isolated."""
    if not enabled:
        return None
    if not health_enabled:
        _warn("Fault events require --use-health-check; collection disabled")
        return None
    try:
        runtime = FaultEventRuntime()
    except Exception:
        _warn("Fault event runtime unavailable")
        return None
    try:
        from .query_server import FaultQueryServer

        runtime.query_server = FaultQueryServer(runtime, port)
    except Exception:
        _warn("Fault event query unavailable")
    return runtime
