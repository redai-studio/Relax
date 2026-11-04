# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fixed completion slots and bounded, replica-owned receipt delivery."""

import queue
import threading
from typing import Any

from .models import MAX_UINT, RecoveryEvidence, encode, valid_id, valid_uint


MAX_RECEIPT_BYTES = 256
OPERATIONS = {
    "actor_send": RecoveryEvidence.ACTOR_WEIGHT_SENT,
    "actor_fwd_recv": RecoveryEvidence.ACTOR_FWD_WEIGHT_RECEIVED,
    "reference_recv": RecoveryEvidence.REFERENCE_WEIGHT_RECEIVED,
}
_CREATION_LOCK = threading.Lock()


def validate_token(token: object) -> None:
    if (
        type(token) is not dict
        or any(type(key) is not str for key in token)
        or token.keys() != {"run_id", "recovery_id", "recovery_generation"}
    ):
        raise ValueError("INVALID_TOKEN: token must contain exactly the recovery identity fields")
    if not valid_id(token["run_id"]) or not valid_id(token["recovery_id"]):
        raise ValueError("INVALID_TOKEN: run_id and recovery_id must be 32 lowercase hex digits")
    if not valid_uint(token["recovery_generation"]):
        raise ValueError("INVALID_TOKEN: recovery_generation must be an unsigned 64-bit int")


def validate_receipt(token: object, operation: object, outcome: object) -> None:
    validate_token(token)
    if type(operation) is not str or operation not in OPERATIONS:
        raise ValueError("INVALID_RECEIPT: operation must identify a weight operation")
    if type(outcome) is not str or outcome not in ("succeeded", "failed"):
        raise ValueError("INVALID_RECEIPT: outcome must be succeeded or failed")
    if len(encode({**token, "operation_id": operation, "outcome": outcome})) > MAX_RECEIPT_BYTES:
        raise ValueError("INVALID_RECEIPT: encoded receipt must not exceed 256 bytes")


def parse_snapshot(payload: object) -> dict | None:
    if payload is None:
        return None
    if type(payload) is not dict or payload.keys() != {"token", "observed", "failed", "conflict"}:
        raise ValueError("INVALID_RECEIPTS: snapshot must contain exactly token, observed, failed and conflict")
    validate_token(payload["token"])
    observed, failed = payload["observed"], payload["failed"]
    if (
        not valid_uint(observed, RecoveryEvidence.WEIGHT_OPERATIONS)
        or not valid_uint(failed, RecoveryEvidence.WEIGHT_OPERATIONS)
        or (observed | failed) & ~RecoveryEvidence.WEIGHT_OPERATIONS
        or observed & failed
    ):
        raise ValueError("INVALID_RECEIPTS: evidence must contain disjoint weight operation bits")
    if type(payload["conflict"]) is not bool:
        raise ValueError("INVALID_RECEIPTS: conflict must be a bool")
    return {**payload, "token": dict(payload["token"])}


class ReceiptInbox:
    """Keep the first result per operation and sticky ambiguous slots."""

    def __init__(self, seed: dict | None = None) -> None:
        if seed is not None:
            validate_token(seed)
        self._seed = seed.copy() if seed is not None else None
        # Receipt slots are wire masks. Business evidence remains IntFlag
        # until Runtime admission; both use the enum's single bit definition.
        self.observed = self.failed = 0
        self._conflicts = 0

    @property
    def conflict(self) -> bool:
        return bool(self._conflicts)

    def receive(self, token: dict, operation: str, outcome: str) -> str:
        validate_receipt(token, operation, outcome)
        if self._seed is None or token != self._seed:
            return "stale"
        bit = OPERATIONS[operation].value
        previous = self.observed if outcome == "succeeded" else self.failed
        opposite = self.failed if outcome == "succeeded" else self.observed
        if opposite & bit:
            self._conflicts |= bit
            return "conflict"
        if previous & bit:
            return "duplicate"
        if outcome == "succeeded":
            self.observed |= bit
        else:
            self.failed |= bit
        return "accepted"

    def snapshot(self) -> dict | None:
        if self._seed is None:
            return None
        return {
            "token": dict(self._seed),
            "observed": self.observed & ~self._conflicts,
            "failed": self.failed & ~self._conflicts,
            "conflict": self.conflict,
        }


def create_inbox(seed: dict | None, generation: int | None) -> ReceiptInbox | None:
    if seed is None:
        return None
    inbox = ReceiptInbox(seed)
    if seed["recovery_generation"] != generation:
        return None
    return inbox


class ReceiptSender:
    """Eight queued messages and one in flight, with no business-thread RPC."""

    def __init__(self, health_status: Any) -> None:
        self._status = health_status
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._counter_lock = threading.Lock()
        self.dropped = 0
        self._thread = threading.Thread(target=self._send, name="fault-receipts", daemon=True)
        self._thread.start()

    def _drop(self) -> None:
        with self._counter_lock:
            self.dropped = min(MAX_UINT, self.dropped + 1)

    def emit(self, token: dict, operation: str, outcome: str) -> None:
        try:
            validate_receipt(token, operation, outcome)
            self._queue.put_nowait((dict(token), operation, outcome))
        except (ValueError, TypeError, queue.Full):
            self._drop()

    def _send(self) -> None:
        import ray

        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                ref = self._status.record_recovery_receipt.remote(*item)
                while not self._stop.is_set():
                    ready, _ = ray.wait([ref], timeout=0.1)
                    if ready:
                        ray.get(ref)
                        break
            except Exception:
                self._drop()
            finally:
                self._queue.task_done()

    def close(self) -> None:
        self._stop.set()


def _sender(owner: Any) -> ReceiptSender | None:
    sender = getattr(owner, "_fault_receipt_sender", None)
    if sender is not None:
        return sender
    if not _CREATION_LOCK.acquire(timeout=0.005):
        return None
    try:
        if getattr(owner, "_fault_receipt_sender", None) is None:
            owner._fault_receipt_sender = ReceiptSender(owner.healthy)
        return owner._fault_receipt_sender
    finally:
        _CREATION_LOCK.release()


def emit_receipt(owner: Any, token: dict | None, operation: str, outcome: str) -> None:
    """Create at most one sender per replica and isolate optional reporting."""
    if token is None:
        return
    try:
        sender = _sender(owner)
        if sender is not None:
            sender.emit(token, operation, outcome)
    except Exception:
        pass
