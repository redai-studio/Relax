# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Driver-owned, bounded fault event admission and consumption."""

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from relax.utils.logging_utils import get_logger

from .context import FaultTraceContext, capture_trace_context, parse_trace_context
from .models import MAX_UINT, REASONS, EventDraft, FaultContext, RecoveryEvidence, new_id, service_role, valid_uint
from .store import FaultEventStore, _Record
from .telemetry import OTelConfigError


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
_MAX_PUBLICATIONS = 18
_MAX_MEMBERS = 9
_MAX_RECOVERIES = 2


@dataclass(slots=True)
class Recovery:
    recovery_id: str
    members: tuple[FaultContext, ...]
    mode: str
    receiver_generation: int
    expected: int
    observed: int = 0
    failed: int = 0
    handoff: bool = False
    outcome: str | None = None
    conflict: bool = False
    _local_complete: threading.Event = field(default_factory=threading.Event, init=False, repr=False, compare=False)
    _local_failed: threading.Event = field(default_factory=threading.Event, init=False, repr=False, compare=False)
    _local_event_ids: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    _local_context: FaultTraceContext | None = field(default=None, init=False, repr=False, compare=False)


@dataclass(slots=True)
class _Publication:
    fault: FaultContext
    contexts: dict[str, FaultTraceContext | None] = field(default_factory=dict)
    published: int = 0
    attempted: int = 0
    consumed: bool = False
    specials: int = 0


class FaultEventRuntime:
    """A bounded FIFO with one consumer; business calls never access the
    Store."""

    def __init__(
        self, capacity: int = 4096, queue_capacity: int = 512, *, run_id: str | None = None, log_exporter=None
    ):
        if type(queue_capacity) is not int or not 1 <= queue_capacity <= 512:
            raise ValueError("INVALID_CAPACITY")
        self.run_id = new_id() if run_id is None else run_id
        self.store = FaultEventStore(self.run_id, capacity)
        self.queue_capacity = queue_capacity
        self.query_server: FaultQueryServer | None = None
        self._queue: deque[tuple[EventDraft, FaultTraceContext | None]] = deque()
        self._pending: dict[str, EventDraft] = {}
        self._admission_context: FaultTraceContext | None = None
        self._log_exporter = log_exporter
        self._otel_startup_errors = 0
        self._otel_observer_errors = 0
        self._otel_context_errors = 0
        self._publications: dict[tuple[int, str], _Publication] = {}
        self._snapshot: tuple[FaultContext, ...] = ()
        self._recoveries: dict[str, Recovery] = {}
        self.generation = -1
        self._source_disabled = False
        self._source_retired = True
        self._source_epoch = (-1, threading.Event())
        self._snapshot_invalid = threading.Event()
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

    def _call(self, function, *args):
        context = capture_trace_context()
        if not self._lock.acquire(timeout=_ADMISSION_TIMEOUT):
            self._increment("_rejections")
            self.note_unknown()
            return None
        try:
            if self._stop.is_set() or self._disabled.is_set():
                return None
            self._admission_context = context
            self._settle_local_completions()
            return function(*args) if function is not None else None
        except Exception:
            self.note_unknown()
            return None
        finally:
            self._admission_context = None
            self._lock.release()

    def _otel_error(self, kind: str = "observer") -> None:
        with self._diagnostic_lock:
            if kind == "context":
                self._otel_context_errors = min(MAX_UINT, self._otel_context_errors + 1)
            elif kind == "startup":
                self._otel_startup_errors = min(MAX_UINT, self._otel_startup_errors + 1)
            else:
                self._otel_observer_errors = min(MAX_UINT, self._otel_observer_errors + 1)

    def _parse_contexts(
        self, payload: object, faults: tuple[FaultContext, ...]
    ) -> dict[str, FaultTraceContext | None]:
        if payload is None:
            return {}
        allowed = {event_id for fault in faults for event_id in (fault.event_id, fault.fatal_event_id) if event_id}
        if type(payload) is not dict or len(payload) > 2 * _MAX_MEMBERS:
            self._otel_error("context")
            return {}
        contexts = {}
        for event_id, value in payload.items():
            if type(event_id) is not str or len(event_id) != 32 or event_id not in allowed:
                self._otel_error("context")
                continue
            try:
                contexts[event_id] = parse_trace_context(value)
            except Exception:
                self._otel_error("context")
        return contexts

    def _draft(
        self,
        fault: FaultContext,
        kind: str,
        reason: str | None,
        event_id: str,
        recovery: Recovery | None = None,
        skip_id: str | None = None,
    ) -> EventDraft:
        # Fields come from validated immutable contexts and fixed event kinds;
        # no caller data or serialization runs in the admission transaction.
        return EventDraft(
            self.run_id,
            fault.fault_id,
            event_id,
            fault.role,
            fault.source_generation,
            kind,
            reason,
            recovery.recovery_id if recovery is not None else skip_id,
            recovery.mode if recovery is not None else None,
            recovery.expected if recovery is not None else 0,
            recovery.observed if recovery is not None else 0,
            recovery.failed if recovery is not None else 0,
        )

    def _retire_publication(self, publication: _Publication) -> None:
        # These stable identities were never admitted. Accepted identities are
        # settled only by the consumer, including across source replacement.
        missing = int(not publication.published & 1)
        if publication.fault.fatal_event_id is not None and not publication.published & 2:
            missing += 1
        self._increment("_lost", missing)

    def _publication(
        self, fault: FaultContext, contexts: dict[str, FaultTraceContext | None] | None = None
    ) -> _Publication | None:
        key = fault.source_generation, fault.role
        publication = self._publications.get(key)
        if publication is not None and publication.fault.fault_id == fault.fault_id:
            previous = publication.fault
            if (
                fault.event_id != previous.event_id
                or fault.reason != previous.reason
                or (previous.fatal and not fault.fatal)
                or (previous.fatal_event_id is not None and fault.fatal_event_id != previous.fatal_event_id)
            ):
                self.note_unknown()
                return None
            publication.fault = fault
            for event_id in (fault.event_id, fault.fatal_event_id):
                if event_id is not None and event_id not in publication.contexts:
                    publication.contexts[event_id] = (contexts or {}).get(event_id)
            return publication
        if publication is not None:
            self._retire_publication(publication)
        elif len(self._publications) >= _MAX_PUBLICATIONS:
            self._source_disabled = True
            self._snapshot = ()
            self._snapshot_invalid.set()
            self.note_unknown("CONTEXT_CAPACITY")
            return None
        publication = _Publication(fault)
        publication.contexts = {
            event_id: (contexts or {}).get(event_id)
            for event_id in (fault.event_id, fault.fatal_event_id)
            if event_id is not None
        }
        self._publications[key] = publication
        return publication

    def _publish(self, publication: _Publication) -> None:
        fault = publication.fault
        events = ((1, fault.event_id, "fault_detected", fault.reason),)
        if fault.fatal_event_id is not None:
            events += ((2, fault.fatal_event_id, "fatal_escalated", "fatal_error"),)
        for bit, event_id, kind, reason in events:
            if not (publication.published | publication.attempted) & bit:
                publication.attempted |= bit
                if self._enqueue(
                    self._draft(fault, kind, reason, event_id), publication.contexts.get(event_id), capture=False
                ):
                    publication.published |= bit

    def _reject_candidate(self, role: str | None) -> None:
        self.note_unknown()
        if self._call(self._remove_candidate, role) is None:
            self._snapshot_invalid.set()

    def _remove_candidate(self, role: str | None) -> bool:
        self._snapshot = () if role is None else tuple(fault for fault in self._snapshot if fault.role != role)
        return True

    def observe_fault(self, payload: dict | None, role: object = None, trace_contexts: object = None) -> None:
        """Merge a same-round health/stale result without repeating
        admission."""
        expected_role = None
        try:
            if self.generation >= 0 and (self._source_retired or self._source_epoch[1].is_set()):
                return
            if role is not None:
                expected_role = service_role(role)
            if payload is None and expected_role is None:
                return
            if type(payload) is dict:
                generation = payload.get("source_generation")
                if valid_uint(generation) and generation != self.generation:
                    return
            fault = FaultContext.from_dict(payload)
            if expected_role is not None and fault.role != expected_role:
                raise ValueError("INVALID_FAULT")
        except Exception:
            self._reject_candidate(expected_role)
            return
        contexts = self._parse_contexts(trace_contexts, (fault,))
        result = self._call(self._observe_fault, fault, contexts)
        if result is None and fault.source_generation == self.generation:
            self._snapshot_invalid.set()

    def _observe_fault(self, fault: FaultContext, contexts: dict | None = None) -> bool:
        if (
            fault.source_generation != self.generation
            or self._source_disabled
            or self._source_retired
            or self._source_epoch[1].is_set()
        ):
            return False
        publication = self._publication(fault, contexts)
        if publication is None:
            self._remove_candidate(fault.role)
            return False
        self._publish(publication)
        if any(candidate.role == fault.role for candidate in self._snapshot):
            self._snapshot = tuple(
                publication.fault if candidate.role == fault.role else candidate for candidate in self._snapshot
            )
        elif len(self._snapshot) < _MAX_MEMBERS:
            self._snapshot += (publication.fault,)
        else:
            self._source_disabled = True
            self._snapshot = ()
            self._snapshot_invalid.set()
            self.note_unknown("CONTEXT_CAPACITY")
            return False
        return True

    def observe_snapshot(self, payload: dict) -> None:
        """Start a health-query round, retaining only its valid candidates."""
        try:
            if type(payload) is not dict:
                raise ValueError("INVALID_SNAPSHOT")
            generation = payload.get("source_generation")
            if valid_uint(generation) and (
                generation != self.generation or self._source_retired or self._source_epoch[1].is_set()
            ):
                return
            if (
                not {"roles", "source_generation", "faults", "observation_error"} <= payload.keys()
                or payload.keys() - {"roles", "source_generation", "faults", "observation_error", "trace_contexts"}
                or not valid_uint(generation)
                or type(payload["observation_error"]) is not bool
                or type(payload["roles"]) not in (list, tuple)
                or type(payload["faults"]) not in (list, tuple)
                or len(payload["roles"]) > _MAX_MEMBERS
                or len(payload["faults"]) > _MAX_MEMBERS
            ):
                raise ValueError("INVALID_SNAPSHOT")
            roles = tuple(service_role(role) for role in payload["roles"])
            faults = tuple(FaultContext.from_dict(fault) for fault in payload["faults"])
            if (
                len(set(roles)) != len(roles)
                or len({fault.role for fault in faults}) != len(faults)
                or any(fault.role not in roles or fault.source_generation != generation for fault in faults)
            ):
                raise ValueError("INVALID_SNAPSHOT")
        except Exception:
            self._reject_candidate(None)
            return
        if payload["observation_error"]:
            self.note_unknown()
        contexts = self._parse_contexts(payload.get("trace_contexts"), faults)
        result = self._call(self._observe_snapshot, generation, faults, contexts)
        if result is None and generation == self.generation:
            self._snapshot_invalid.set()

    def _observe_snapshot(
        self, generation: int, faults: tuple[FaultContext, ...], contexts: dict | None = None
    ) -> bool:
        if (
            generation != self.generation
            or self._source_disabled
            or self._source_retired
            or self._source_epoch[1].is_set()
        ):
            return False
        active = {fault.fault_id for recovery in self._recoveries.values() for fault in recovery.members}
        for publication in self._publications.values():
            publication.attempted = 0
            publication.specials &= 2  # Fatal termination is once per fault; skip is once per round.
            if publication.fault.fault_id not in active:
                publication.consumed = False
        candidates = []
        self._snapshot = ()
        self._snapshot_invalid.clear()
        # Replace/retire every changed identity before retrying cached entries.
        # A newly observed F2 must not rescue an already retired F1 first.
        for fault in faults:
            publication = self._publication(fault, contexts)
            if self._source_disabled:
                return False
            if publication is not None:
                candidates.append(publication.fault)
        self._snapshot = tuple(candidates)
        for publication in self._publications.values():
            if publication.fault.source_generation == generation:
                self._publish(publication)
        return True

    def bind_source(self) -> tuple[int | None, None]:
        """Retire the old source and reserve the next receiver generation."""
        try:
            event_ids = tuple(new_id() for _ in range(_MAX_RECOVERIES * _MAX_MEMBERS))
            result = self._call(self._bind_source, event_ids, threading.Event())
            return result if result is not None else (None, None)
        except Exception:
            self.note_unknown()
            return None, None

    def retire_source(self, generation: int | None = None) -> None:
        """Retire an actually abandoned source once, without remote work."""
        try:
            if generation is not None and not valid_uint(generation):
                raise ValueError("INVALID_GENERATION")
            epoch = self._source_epoch
            generation = epoch[0] if generation is None else generation
            if generation != epoch[0] or generation < 0:
                return
            # This fence belongs to one source object. A late caller can close
            # only that epoch, even if bind_source has already installed a new
            # one. Counting and terminal selection still use admission below.
            epoch[1].set()
            self.note_unknown("SOURCE_RETIRED")
            event_ids = tuple(new_id() for _ in range(_MAX_RECOVERIES * _MAX_MEMBERS))
            self._call(self._retire_source, generation, event_ids)
        except Exception:
            self.note_unknown()

    def _retire_source(self, generation: int | None, event_ids: tuple[str, ...]) -> bool:
        old = self.generation if generation is None else generation
        if old != self.generation or old < 0 or self._source_retired:
            return False
        self._source_epoch[1].set()
        self._source_retired = True
        self._snapshot = ()
        self._snapshot_invalid.set()
        self.note_unknown("SOURCE_RETIRED")
        for publication in self._publications.values():
            self._retire_publication(publication)
        self._publications.clear()
        for index, recovery in enumerate(tuple(self._recoveries.values())):
            if recovery.receiver_generation == old:
                self._finish(
                    recovery,
                    "recovery_interrupted",
                    "source_retired",
                    event_ids[index * _MAX_MEMBERS : (index + 1) * _MAX_MEMBERS],
                )
        return True

    def _bind_source(self, event_ids: tuple[str, ...], source_fence: threading.Event) -> tuple[int | None, None]:
        old = self.generation
        self._retire_source(old, event_ids)
        if old >= MAX_UINT:
            self._source_disabled = True
            self._snapshot_invalid.set()
            self.note_unknown()
            return None, None
        self.generation = old + 1
        self._source_disabled = False
        self._source_retired = False
        self._source_epoch = (self.generation, source_fence)
        self._snapshot_invalid.clear()
        return self.generation, None

    def _emit(
        self,
        members: tuple[FaultContext, ...],
        kind: str,
        reason: str | None,
        recovery: Recovery,
        event_ids: tuple[str, ...],
    ) -> None:
        for fault, event_id in zip(members, event_ids, strict=False):
            if not self._enqueue(self._draft(fault, kind, reason, event_id, recovery)):
                self._increment("_lost")

    def start_recovery(self, role: str, mode: str, expected: int) -> Recovery | None:
        try:
            role = service_role(role)
            if isinstance(expected, RecoveryEvidence):
                expected = int(expected)
            if (
                type(mode) is not str
                or mode not in ("local", "global")
                or not valid_uint(expected, 63)
                or not expected
            ):
                raise ValueError("INVALID_RECOVERY")
            recovery_id = new_id()
            event_ids = tuple(new_id() for _ in range(_MAX_MEMBERS))
            local_event_ids = tuple(new_id() for _ in range(2 * _MAX_MEMBERS)) if mode == "local" else ()
            return self._call(self._start_recovery, role, mode, expected, recovery_id, event_ids, local_event_ids)
        except Exception:
            self.note_unknown()
            return None

    def _start_recovery(
        self,
        role: str,
        mode: str,
        expected: int,
        recovery_id: str,
        event_ids: tuple[str, ...],
        local_event_ids: tuple[str, ...],
    ) -> Recovery | None:
        if (
            self._snapshot_invalid.is_set()
            or self._source_disabled
            or self._source_retired
            or self._source_epoch[1].is_set()
        ):
            self.note_unknown("OUTCOME_UNOBSERVED")
            return None
        if len(self._recoveries) >= _MAX_RECOVERIES:
            self.note_unknown("CONTEXT_CAPACITY")
            return None
        if not valid_uint(self.generation) or (mode == "global" and self.generation == MAX_UINT):
            self.note_unknown()
            return None
        active = {fault.fault_id for recovery in self._recoveries.values() for fault in recovery.members}
        candidates = []
        for fault in self._snapshot:
            publication = self._publications.get((fault.source_generation, fault.role))
            if (
                publication is not None
                and publication.fault == fault
                and not publication.consumed
                and not fault.fatal
                and fault.fault_id not in active
                and (mode == "global" or fault.role == role)
            ):
                candidates.append(fault)
        if not candidates or not any(fault.role == role for fault in candidates):
            self.note_unknown("OUTCOME_UNOBSERVED")
            return None
        members = tuple(candidates)
        recovery = Recovery(recovery_id, members, mode, self.generation + int(mode == "global"), expected)
        recovery._local_event_ids = local_event_ids
        self._recoveries[recovery_id] = recovery
        for fault in members:
            self._publications[(fault.source_generation, fault.role)].consumed = True
        self._emit(members, "handling_started", None, recovery, event_ids)
        return recovery

    def complete_local(self, recovery: Recovery | None, failed: bool = False) -> None:
        """Remember the original local flow's return/exception without
        waiting."""
        if recovery is None:
            return
        try:
            if type(recovery) is not Recovery or recovery.mode != "local" or type(failed) is not bool:
                raise ValueError("INVALID_LOCAL_COMPLETION")
            # The identity check is also safe before taking admission: a late
            # call may mark only its own object, never a replacement attempt.
            if self._recoveries.get(recovery.recovery_id) is not recovery:
                return
            if failed:
                recovery._local_failed.set()
            if not recovery._local_complete.is_set():
                recovery._local_context = capture_trace_context()
            recovery._local_complete.set()
            self._call(None)
        except Exception:
            self.note_unknown()

    def _settle_local_completions(self) -> None:
        # Every lifecycle transaction drains these fixed fences first, before
        # source retirement, capacity checks or the next round's adopted set.
        for recovery in tuple(self._recoveries.values()):
            if recovery.mode != "local" or not recovery._local_complete.is_set() or recovery.outcome is not None:
                continue
            event_ids = recovery._local_event_ids
            previous_context = self._admission_context
            self._admission_context = recovery._local_context
            try:
                if recovery._local_failed.is_set():
                    self._finish(recovery, "recovery_failed", "restart_exception", event_ids[_MAX_MEMBERS:])
                    continue
                self._evidence(recovery, 0, 0, True, "restart_exception", event_ids)
            finally:
                self._admission_context = previous_context
            if recovery.outcome is None:
                # The real local flow has ended; no receipt can fill a missing
                # local observation later. Free its slot without inventing a
                # business failure or a successful recovery.
                self.note_unknown("OUTCOME_UNOBSERVED")
                self._recoveries.pop(recovery.recovery_id, None)

    def evidence(
        self,
        recovery: Recovery | None,
        observed: int = 0,
        failed: int = 0,
        handoff: bool = False,
        reason: str = "restart_exception",
    ) -> None:
        if recovery is None:
            return
        try:
            if isinstance(observed, RecoveryEvidence):
                observed = int(observed)
            if isinstance(failed, RecoveryEvidence):
                failed = int(failed)
            if (
                type(recovery) is not Recovery
                or not valid_uint(observed, 63)
                or not valid_uint(failed, 63)
                or type(handoff) is not bool
                or type(reason) is not str
                or reason not in REASONS["recovery_failed"]
            ):
                raise ValueError("INVALID_EVIDENCE")
            event_ids = tuple(new_id() for _ in range(2 * _MAX_MEMBERS))
            self._call(self._evidence, recovery, observed, failed, handoff, reason, event_ids)
        except Exception:
            self.note_unknown()

    def _evidence(
        self, recovery: Recovery, observed: int, failed: int, handoff: bool, reason: str, event_ids: tuple[str, ...]
    ) -> None:
        if recovery.outcome is not None:
            if (observed & recovery.failed) or (failed & recovery.observed):
                recovery.conflict = True
                self.note_unknown("RECEIPT_CONFLICT")
            return
        if self._recoveries.get(recovery.recovery_id) is not recovery:
            return
        if recovery.mode == "global" and (observed or handoff):
            self.note_unknown()
            return
        if (observed | failed) & ~recovery.expected or (recovery.observed | observed) & (recovery.failed | failed):
            recovery.conflict = True
            self.note_unknown("RECEIPT_CONFLICT")
            return
        recovery.observed |= observed
        recovery.failed |= failed
        if handoff and not recovery.handoff:
            recovery.handoff = True
            self._emit(recovery.members, "recovery_handoff", None, recovery, event_ids[:_MAX_MEMBERS])
        if recovery.failed:
            self._finish(recovery, "recovery_failed", reason, event_ids[_MAX_MEMBERS:])
        elif (
            recovery.mode == "local"
            and recovery.handoff
            and recovery.observed == recovery.expected
            and not recovery.conflict
        ):
            self._finish(recovery, "recovery_succeeded", None, event_ids[_MAX_MEMBERS:])

    def finish(self, recovery: Recovery | None, kind: str, reason: str | None) -> None:
        if recovery is None:
            return
        try:
            if (
                type(recovery) is not Recovery
                or type(kind) is not str
                or kind not in ("recovery_failed", "recovery_interrupted", "recovery_succeeded")
                or (reason is not None and type(reason) is not str)
                or reason not in REASONS[kind]
            ):
                raise ValueError("INVALID_OUTCOME")
            event_ids = tuple(new_id() for _ in range(_MAX_MEMBERS))
            self._call(self._finish, recovery, kind, reason, event_ids)
        except Exception:
            self.note_unknown()

    def _finish(self, recovery: Recovery, kind: str, reason: str | None, event_ids: tuple[str, ...]) -> None:
        if recovery.outcome is not None or self._recoveries.get(recovery.recovery_id) is not recovery:
            return
        if kind == "recovery_succeeded" and (
            recovery.mode != "local"
            or not recovery.handoff
            or recovery.observed != recovery.expected
            or recovery.failed
            or recovery.conflict
        ):
            self.note_unknown()
            return
        recovery.outcome = kind
        self._recoveries.pop(recovery.recovery_id)
        self._emit(recovery.members, kind, reason, recovery, event_ids)

    def special(self, role: str, kind: str, reason: str, recovery: Recovery | None = None) -> None:
        try:
            role = service_role(role)
            if (
                type(kind) is not str
                or kind not in ("handling_skipped", "termination_requested")
                or type(reason) is not str
                or reason not in REASONS[kind]
                or (recovery is not None and type(recovery) is not Recovery)
            ):
                raise ValueError("INVALID_SPECIAL")
            event_ids = tuple(new_id() for _ in range(_MAX_MEMBERS))
            self._call(self._special, role, kind, reason, recovery, event_ids, new_id())
        except Exception:
            self.note_unknown()

    def _special(
        self, role: str, kind: str, reason: str, recovery: Recovery | None, event_ids: tuple[str, ...], skip_id: str
    ) -> None:
        if reason == "restart_limit":
            if (
                recovery is None
                or recovery.mode != "global"
                or recovery.outcome is not None
                or self._recoveries.get(recovery.recovery_id) is not recovery
            ):
                return
            fault = next((fault for fault in recovery.members if fault.role == role), None)
            if fault is None:
                self.note_unknown("OUTCOME_UNOBSERVED")
                return
            recovery.outcome = kind
            self._recoveries.pop(recovery.recovery_id)
            self._emit(recovery.members, kind, reason, recovery, event_ids)
            return
        else:
            fault = next((fault for fault in self._snapshot if fault.role == role), None)
            if fault is None or self._snapshot_invalid.is_set() or recovery is not None:
                self.note_unknown("OUTCOME_UNOBSERVED")
                return
            publication = self._publications[(fault.source_generation, fault.role)]
            bit = 1 if kind == "handling_skipped" else 2
            if publication.specials & bit:
                return
            if reason == "fatal_error" and not fault.fatal:
                self.note_unknown()
                return
            publication.specials |= bit
            publication.consumed = True
            draft = self._draft(
                fault, kind, reason, event_ids[0], skip_id=skip_id if kind == "handling_skipped" else None
            )
        if not self._enqueue(draft):
            self._increment("_lost")

    def _enqueue(self, draft: EventDraft, context: FaultTraceContext | None = None, *, capture: bool = True) -> bool:
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
            self._queue.append((draft, self._admission_context if capture else context))
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
        context = capture_trace_context()
        try:
            acquired = self._lock.acquire(timeout=_ADMISSION_TIMEOUT)
            if not acquired:
                self._increment("_rejections")
                return False
            return self._enqueue(draft, context, capture=False)
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
            publications, snapshot_size, recoveries = (
                len(self._publications),
                len(self._snapshot),
                len(self._recoveries),
            )
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
            "publications": publications,
            "snapshot_size": snapshot_size,
            "active_recoveries": recoveries,
            **self._otel_diagnostics(),
        }

    def _otel_diagnostics(self) -> dict:
        result = {
            "otel_enabled": bool(self._otel_startup_errors),
            "otel_available": False,
            "otel_queue_capacity": 0,
            "otel_queued": 0,
            "otel_inflight": 0,
            "otel_admitted": 0,
            "otel_rejections": 0,
            "otel_exported": 0,
            "otel_export_failures": 0,
            "otel_shutdown_dropped": 0,
            "otel_context_errors": 0,
            "otel_closed": self._stop.is_set(),
        }
        if self._log_exporter is not None:
            try:
                reported = self._log_exporter.diagnostics()
                validated = {}
                for key in result:
                    value = reported[key]
                    if (type(result[key]) is bool and type(value) is not bool) or (
                        type(result[key]) is int and not valid_uint(value)
                    ):
                        raise ValueError("INVALID_OTEL_DIAGNOSTICS")
                    validated[key] = value
                result.update(validated)
            except Exception:
                self._otel_error()
        with self._diagnostic_lock:
            result["otel_context_errors"] = min(MAX_UINT, result["otel_context_errors"] + self._otel_context_errors)
            result["otel_startup_errors"] = self._otel_startup_errors
            result["otel_observer_errors"] = self._otel_observer_errors
        return result

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

    def _submit(self, draft: EventDraft, context: FaultTraceContext | None) -> None:
        for attempt in range(2):
            if self._stop.is_set():
                self._settle_abandoned(draft)
                return
            try:
                committed = self.store._append_committed(draft)
                if type(committed) is not tuple or len(committed) != 2:
                    self._invalidate_store()
                    return
                result, record = committed
                if (
                    type(result) is not str
                    or result not in ("accepted", "duplicate")
                    or type(record) is not _Record
                    or record.draft != draft
                ):
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
            # Export is a separate best-effort boundary after a definite local
            # commit. Its failures must never disable or invalidate the Store.
            if result == "accepted" and self._log_exporter is not None:
                try:
                    self._log_exporter.offer(record.draft.record(record.sequence, record.timestamp), context)
                except Exception:
                    self._otel_error()
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
                    entry = self._queue.popleft() if self._queue else None
                    if not self._queue:
                        self._wake.clear()
                    self._inflight = int(entry is not None)
                finally:
                    self._lock.release()
                if entry is None:
                    continue
                draft, context = entry
                try:
                    self._submit(draft, context)
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
                for publication in self._publications.values():
                    self._retire_publication(publication)
                self._publications.clear()
                self._snapshot = ()
                self._source_retired = True
                if self._recoveries:
                    self.note_unknown("OUTCOME_UNOBSERVED")
                    self._recoveries.clear()
            for draft, _ in remaining:
                self._settle_abandoned(draft)

    def close(self) -> None:
        """Signal local shutdown without draining the event queue.

        The optional exporter receives its own finite close budget; local
        history and query shutdown never wait for business queue drain.
        """
        self._stop.set()
        self._wake.set()
        if not self._close_lock.acquire(blocking=False):
            return
        try:
            if self._log_exporter is not None:
                try:
                    self._log_exporter.close(timeout=0.05)
                except Exception:
                    self._otel_error()
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
        from .telemetry import start_fault_log_exporter

        runtime._log_exporter = start_fault_log_exporter(enabled=True)
    except OTelConfigError as exc:
        runtime._otel_error("startup")
        # Fixed text naming the rejected setting; it never contains the value.
        _warn(f"Fault event log export disabled: {exc.args[0]}")
    except Exception:
        runtime._otel_error("startup")
        _warn("Fault event log export unavailable")
    try:
        from .query_server import FaultQueryServer

        runtime.query_server = FaultQueryServer(runtime, port)
    except Exception:
        _warn("Fault event query unavailable")
    return runtime
