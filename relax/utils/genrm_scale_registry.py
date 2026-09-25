# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GenRM scale-operation registry: state machine, idempotency, mutual
exclusion.

This module holds the Task 4 contract core for GenRM elastic scaling:

- Operation state machines aligned with Rollout scaling semantics (values match
  ``relax.distributed.ray.rollout.ScaleOutStatus`` / ``ScaleInStatus``):

    scale-out: PENDING -> CREATING -> HEALTH_CHECKING -> READY -> ACTIVE
               terminal: ACTIVE / PARTIAL / FAILED  (no CANCELLED; unfinished
               cleanup is reconciled instead -- see ``cleanup_required``)
    scale-in:  PENDING -> DRAINING -> REMOVING -> COMPLETED / FAILED

  The terminal sets are cross-checked against the autoscaler's
  ``SCALE_OUT_TERMINAL_STATUSES`` / ``SCALE_IN_TERMINAL_STATUSES`` in unit
  tests, so the existing terminal判定 keeps working for GenRM operations.

- ``num_replicas`` is an *absolute* target total. Idempotency keys replay by
  request-body fingerprint ``(model, target, timeout_secs)``: same key + same
  fingerprint returns the original operation; same key + different fingerprint
  is a 409; a keyed NOOP decision is recorded and replayed verbatim (a retry
  after a capacity change never executes a new operation).

- Mutual exclusion: one in-flight (non-terminal) operation per model, plus any
  terminal operation whose cleanup is still required, blocks new scale
  requests with 409.

The registry is deliberately dependency-free (stdlib + repo logger) so the
contract is unit-testable on CPU without Ray/SGLang. The GenRM Serve
deployment delegates to it; actual Ray lifecycle execution (placement-group
bring-up, drain fences, PG release) is delegated to manager hooks and is not
implemented here.
"""

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class GenRMScaleDirection(str, enum.Enum):
    """Direction of a GenRM scale operation."""

    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"


class GenRMScaleOutStatus(str, enum.Enum):
    """Scale-out operation states.

    Values deliberately match ``ScaleOutStatus`` in
    ``relax/distributed/ray/rollout.py`` (minus CANCELLED per the Task 4 RFC:
    unfinished cleanup is reconciled, not cancelled), so status strings are
    interchangeable with the Rollout / autoscaler machinery.
    """

    PENDING = "PENDING"
    CREATING = "CREATING"
    HEALTH_CHECKING = "HEALTH_CHECKING"
    READY = "READY"
    ACTIVE = "ACTIVE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class GenRMScaleInStatus(str, enum.Enum):
    """Scale-in operation states; values match ``ScaleInStatus`` in rollout."""

    PENDING = "PENDING"
    DRAINING = "DRAINING"
    REMOVING = "REMOVING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# Terminal status sets. Scale-out omits CANCELLED on purpose (RFC: reconcile
# instead); it remains a subset of the autoscaler's SCALE_OUT_TERMINAL_STATUSES.
GENRM_SCALE_OUT_TERMINAL_STATUSES = frozenset({"ACTIVE", "PARTIAL", "FAILED"})
GENRM_SCALE_IN_TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED"})

# Legal status transitions. Failure (and PARTIAL, once at least one replica is
# published) may be entered from any live state.
_SCALE_OUT_TRANSITIONS: Dict[str, frozenset] = {
    GenRMScaleOutStatus.PENDING.value: frozenset(
        {GenRMScaleOutStatus.CREATING.value, GenRMScaleOutStatus.FAILED.value}
    ),
    GenRMScaleOutStatus.CREATING.value: frozenset(
        {GenRMScaleOutStatus.HEALTH_CHECKING.value, GenRMScaleOutStatus.FAILED.value}
    ),
    GenRMScaleOutStatus.HEALTH_CHECKING.value: frozenset(
        {GenRMScaleOutStatus.READY.value, GenRMScaleOutStatus.PARTIAL.value, GenRMScaleOutStatus.FAILED.value}
    ),
    GenRMScaleOutStatus.READY.value: frozenset(
        {GenRMScaleOutStatus.ACTIVE.value, GenRMScaleOutStatus.PARTIAL.value, GenRMScaleOutStatus.FAILED.value}
    ),
    GenRMScaleOutStatus.ACTIVE.value: frozenset(),
    GenRMScaleOutStatus.PARTIAL.value: frozenset(),
    GenRMScaleOutStatus.FAILED.value: frozenset(),
}

_SCALE_IN_TRANSITIONS: Dict[str, frozenset] = {
    GenRMScaleInStatus.PENDING.value: frozenset({GenRMScaleInStatus.DRAINING.value, GenRMScaleInStatus.FAILED.value}),
    GenRMScaleInStatus.DRAINING.value: frozenset({GenRMScaleInStatus.REMOVING.value, GenRMScaleInStatus.FAILED.value}),
    GenRMScaleInStatus.REMOVING.value: frozenset(
        {GenRMScaleInStatus.COMPLETED.value, GenRMScaleInStatus.FAILED.value}
    ),
    GenRMScaleInStatus.COMPLETED.value: frozenset(),
    GenRMScaleInStatus.FAILED.value: frozenset(),
}


def _terminal_statuses(direction: str) -> frozenset:
    if direction == GenRMScaleDirection.SCALE_IN.value:
        return GENRM_SCALE_IN_TERMINAL_STATUSES
    return GENRM_SCALE_OUT_TERMINAL_STATUSES


def _transitions(direction: str) -> Dict[str, frozenset]:
    if direction == GenRMScaleDirection.SCALE_IN.value:
        return _SCALE_IN_TRANSITIONS
    return _SCALE_OUT_TRANSITIONS


def _is_status_terminal(direction: str, status: str) -> bool:
    return status in _terminal_statuses(direction)


def _new_request_id() -> str:
    return f"genrm-scale-{uuid.uuid4()}"


@dataclass
class GenRMScaleOperation:
    """One registered GenRM scale operation and its state.

    ``target`` is the absolute total requested by the client. The terminal
    snapshot fields (``current``/``ready``/``created``/``removed``/``failed``/
    ``cleanup_required``) are filled by ``finish`` and reported together with
    the request status.
    """

    request_id: str
    direction: str
    model_name: str
    target: int
    timeout_secs: Optional[float] = None
    idempotency_key: Optional[str] = None
    status: str = GenRMScaleOutStatus.PENDING.value
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # Capacity observed at submit time (for logging / NOOP bookkeeping).
    observed_current: int = 0
    observed_ready: int = 0
    # Terminal snapshot fields.
    current: Optional[int] = None
    ready: Optional[int] = None
    created: int = 0
    removed: int = 0
    failed: int = 0
    cleanup_required: bool = False
    error_message: Optional[str] = None
    detail: Optional[str] = None

    def fingerprint(self) -> tuple:
        """Request-body fingerprint: (model, target, timeout_secs)."""
        return (self.model_name, self.target, self.timeout_secs)

    def is_terminal(self) -> bool:
        return _is_status_terminal(self.direction, self.status)

    def update_status(self, status: str, error_message: Optional[str] = None) -> None:
        self.status = status
        self.updated_at = time.time()
        if error_message:
            self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "direction": self.direction,
            "status": self.status,
            "model_name": self.model_name,
            "target": self.target,
            "timeout_secs": self.timeout_secs,
            "idempotency_key": self.idempotency_key,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current": self.current,
            "ready": self.ready,
            "created": self.created,
            "removed": self.removed,
            "failed": self.failed,
            "cleanup_required": self.cleanup_required,
            "error_message": self.error_message,
            "detail": self.detail,
        }


@dataclass
class _IdempotencyRecord:
    """Outcome recorded for one (direction, idempotency_key)."""

    fingerprint: tuple
    # None outcome.request_id marks a recorded NOOP decision (replayed verbatim).
    request_id: Optional[str] = None
    noop_response: Optional[Dict[str, Any]] = None


class GenRMScaleRegistry:
    """Registry enforcing the GenRM scale contract for one Serve deployment.

    All mutating methods take a single internal lock (no awaits are held while
    locked), so a Serve replica's event loop plus any executor callbacks can
    share one registry safely.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # (direction, idempotency_key) -> _IdempotencyRecord
        self._idempotency: Dict[tuple, _IdempotencyRecord] = {}
        self._operations: Dict[str, GenRMScaleOperation] = {}
        # model_name -> protected lower bound (initial engine count).
        self._initial_capacity: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def register_initial(self, model_name: str, engine_count: int) -> None:
        """Record the protected lower bound (``initial``) for one model."""
        with self._lock:
            self._initial_capacity[model_name] = max(1, int(engine_count))

    def initial_for(self, model_name: str) -> int:
        with self._lock:
            return self._initial_capacity.get(model_name, 1)

    # ------------------------------------------------------------------
    # Submission (validation -> idempotent replay -> exclusion -> NOOP/accept)
    # ------------------------------------------------------------------

    def submit(
        self,
        direction: str,
        *,
        model_name: str,
        target: Any,
        timeout_secs: Optional[float] = None,
        idempotency_key: Optional[str] = None,
        current: int,
        ready: int,
    ) -> Dict[str, Any]:
        """Admit (or replay) a scale request.

        Returns a demo-shaped decision dict:
          - accepted: {"http": 200, "status": "PENDING", "request_id": ...,
             "dispatch": True}
          - replay of an existing operation: {"http": 200, "status": <current
             op status>, "request_id": ..., "dispatch": False}
          - NOOP: {"http": 200, "status": "NOOP", "current": ..., "ready": ...}
          - rejected: {"http": 400/409/422, "status": ..., "detail": ...}
        """
        if direction not in (GenRMScaleDirection.SCALE_OUT.value, GenRMScaleDirection.SCALE_IN.value):
            raise ValueError(f"unknown scale direction: {direction!r}")

        # 1. Validation: target must be a positive int (bool excluded).
        if isinstance(target, bool) or not isinstance(target, int):
            return {"http": 422, "status": "INVALID_TYPE", "detail": "num_replicas must be an integer"}
        initial = self.initial_for(model_name)
        if target < initial:
            return {
                "http": 400,
                "status": "INVALID_RANGE",
                "detail": f"target {target} is below the protected initial capacity {initial}",
            }

        with self._lock:
            # 2. Idempotent replay by (direction, key) + fingerprint.
            if idempotency_key is not None:
                record = self._idempotency.get((direction, idempotency_key))
                if record is not None:
                    fingerprint = (model_name, target, timeout_secs)
                    if record.fingerprint != fingerprint:
                        return {
                            "http": 409,
                            "status": "IDEMPOTENCY_CONFLICT",
                            "detail": "idempotency key was reused with a different request body",
                        }
                    if record.request_id is None:
                        # Keyed NOOP: replay the recorded response verbatim.
                        return dict(record.noop_response or {})
                    op = self._operations.get(record.request_id)
                    if op is not None:
                        return {"http": 200, "status": op.status, "request_id": op.request_id, "dispatch": False}
                    # Operation record lost (should not happen); fall through
                    # to normal admission so the retry still lands safely.

            # 3. Mutual exclusion: in-flight op or unresolved cleanup.
            blocking = self._blocking_operation_locked(model_name)
            if blocking is not None:
                return {
                    "http": 409,
                    "status": "CONFLICT",
                    "detail": f"another scale operation is active or unresolved for model '{model_name}'",
                }

            # 4. NOOP on satisfied absolute targets.
            if direction == GenRMScaleDirection.SCALE_OUT.value:
                satisfied = target <= current
            else:
                satisfied = target >= current
            if satisfied:
                response = {"http": 200, "status": "NOOP", "current": current, "ready": ready}
                if idempotency_key is not None:
                    self._idempotency[(direction, idempotency_key)] = _IdempotencyRecord(
                        fingerprint=(model_name, target, timeout_secs),
                        request_id=None,
                        noop_response=dict(response),
                    )
                return response

            # 5. Admit the operation.
            operation = GenRMScaleOperation(
                request_id=_new_request_id(),
                direction=direction,
                model_name=model_name,
                target=target,
                timeout_secs=timeout_secs,
                idempotency_key=idempotency_key,
                status=GenRMScaleInStatus.PENDING.value
                if direction == GenRMScaleDirection.SCALE_IN.value
                else GenRMScaleOutStatus.PENDING.value,
                observed_current=current,
                observed_ready=ready,
            )
            self._operations[operation.request_id] = operation
            if idempotency_key is not None:
                self._idempotency[(direction, idempotency_key)] = _IdempotencyRecord(
                    fingerprint=operation.fingerprint(),
                    request_id=operation.request_id,
                )
            logger.info(
                "GenRM scale-%s admitted: request_id=%s model=%s target=%d current=%d",
                "out" if direction == "scale_out" else "in",
                operation.request_id,
                model_name,
                target,
                current,
            )
            return {"http": 200, "status": "PENDING", "request_id": operation.request_id, "dispatch": True}

    def _blocking_operation_locked(self, model_name: str) -> Optional[GenRMScaleOperation]:
        """An in-flight operation, or a terminal one with unresolved
        cleanup."""
        for op in self._operations.values():
            if op.model_name != model_name:
                continue
            if not op.is_terminal():
                return op
            if op.cleanup_required:
                return op
        return None

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def advance(self, request_id: str, status: str, error_message: Optional[str] = None) -> Dict[str, Any]:
        """Advance an operation to ``status`` following the legal transitions.

        Raises:
            KeyError: unknown request_id.
            ValueError: the transition is not legal.
        """
        with self._lock:
            op = self._operations[request_id]
            allowed = _transitions(op.direction).get(op.status, frozenset())
            if status not in allowed:
                raise ValueError(f"illegal transition {op.status} -> {status} for {op.direction}")
            op.update_status(status, error_message)
            logger.info("GenRM scale op %s: %s -> %s", request_id, op.status, status)
            return op.to_dict()

    def finish(
        self,
        request_id: str,
        *,
        status: str,
        current: int,
        ready: int,
        created: int = 0,
        removed: int = 0,
        failed: int = 0,
        cleanup_required: bool = False,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Move an operation to a terminal status with the result snapshot."""
        with self._lock:
            op = self._operations[request_id]
            if status not in _terminal_statuses(op.direction):
                raise ValueError(f"{status} is not a terminal status for {op.direction}")
            op.update_status(status, error_message)
            op.current = current
            op.ready = ready
            op.created = created
            op.removed = removed
            op.failed = failed
            op.cleanup_required = cleanup_required
            logger.info(
                "GenRM scale op %s finished: status=%s current=%d ready=%d created=%d removed=%d failed=%d "
                "cleanup_required=%s",
                request_id,
                status,
                current,
                ready,
                created,
                removed,
                failed,
                cleanup_required,
            )
            return op.to_dict()

    def set_detail(self, request_id: str, detail: Optional[str]) -> None:
        """Attach an execution detail (e.g.
        'manager_scale_not_implemented')."""
        with self._lock:
            self._operations[request_id].detail = detail
            self._operations[request_id].updated_at = time.time()

    def clear_cleanup(self, request_id: str) -> bool:
        """Clear ``cleanup_required`` after a successful reconcile.

        The flag is what blocks new scale requests for the model, so it is only
        ever set on terminal operations. Returns ``True`` when the flag was set
        and is now cleared; ``False`` for an unknown request or one that was
        already clean (making the reconcile replay idempotent).
        """
        with self._lock:
            op = self._operations.get(request_id)
            if op is None or not op.cleanup_required:
                return False
            op.cleanup_required = False
            op.updated_at = time.time()
            logger.info("GenRM scale op %s: cleanup_required cleared after reconcile", request_id)
            return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_status(self, direction: str, request_id: str) -> Optional[Dict[str, Any]]:
        """Public status lookup; ``None`` for unknown IDs."""
        with self._lock:
            op = self._operations.get(request_id)
            if op is None or op.direction != direction:
                return None
            return op.to_dict()

    def list_operations(
        self,
        direction: str,
        model_name: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List operations with optional model/status filters."""
        with self._lock:
            ops = [
                op.to_dict()
                for op in self._operations.values()
                if op.direction == direction
                and (model_name is None or op.model_name == model_name)
                and (status is None or op.status == status)
            ]
        return ops

    def active_operation(self, model_name: str) -> Optional[Dict[str, Any]]:
        """The operation currently blocking new scale requests, if any."""
        with self._lock:
            op = self._blocking_operation_locked(model_name)
            return op.to_dict() if op is not None else None
