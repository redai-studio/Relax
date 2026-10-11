# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Deterministic contract simulator for RFC #351; no Ray or GenRM runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Replica:
    replica_id: str
    owner: str
    pg: str
    workers: int = 2
    healthy: bool = True
    published: bool = True
    admission_open: bool = True
    inflight: set[str] = field(default_factory=set)
    backend_busy: bool = False
    released: bool = False


class ContractDemo:
    def __init__(self) -> None:
        self.replicas = {"initial-0": Replica("initial-0", "initial", "training-pg")}
        self.initial_capacity = 1
        self.pg_live = {"training-pg"}
        self.requests: dict[str, dict[str, Any]] = {}
        # (direction, idempotency_key) -> (fingerprint, outcome). The fingerprint
        # covers the full request body minus the key itself; the outcome is either
        # a recorded NOOP response (replayed verbatim) or the original request id.
        self.idempotency: dict[tuple[str, str], tuple[tuple[Any, ...], dict[str, Any]]] = {}
        self.active_request: str | None = None
        self.paused = False
        self.next_id = 1
        self.events: list[dict[str, Any]] = []
        self._record("BOOT", "Initial replica ready; training PG belongs to its original owner")

    @property
    def current(self) -> int:
        return sum(
            r.published or (r.owner == "manager" and r.healthy and not r.released and not r.admission_open)
            for r in self.replicas.values()
        )

    @property
    def ready(self) -> int:
        return sum(r.published and r.admission_open for r in self.replicas.values())

    def _record(self, kind: str, message: str, **extra: Any) -> None:
        self.events.append(
            {
                "kind": kind,
                "message": message,
                "current": self.current,
                "ready": self.ready,
                "published": sorted(r.replica_id for r in self.replicas.values() if r.published),
                "inflight": {r.replica_id: sorted(r.inflight) for r in self.replicas.values() if r.inflight},
                "live_pgs": sorted(self.pg_live),
                "paused": self.paused,
                **extra,
            }
        )

    def request(
        self,
        direction: str,
        target: Any,
        *,
        model: str = "default",
        timeout_secs: Any = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if direction not in ("out", "in"):
            raise ValueError("direction must be 'out' or 'in'")
        if isinstance(target, bool) or not isinstance(target, int):
            return {"http": 422, "status": "INVALID_TYPE"}
        if target < self.initial_capacity or (direction == "in" and target < self.initial_capacity):
            return {"http": 400, "status": "INVALID_RANGE"}
        # The fingerprint is the full request body minus the key itself: the
        # normalized model selection, the absolute target and timeout_secs.
        # Direction is part of the record key, so a key reused across the
        # scale_out/scale_in endpoints never collides.
        fingerprint = (model, target, timeout_secs)
        record_key = (direction, idempotency_key)
        if idempotency_key is not None:
            existing = self.idempotency.get(record_key)
            if existing is not None:
                existing_fingerprint, outcome = existing
                if existing_fingerprint != fingerprint:
                    self._record("REJECT_409", "Idempotency key was reused with a different request body")
                    return {"http": 409, "status": "IDEMPOTENCY_CONFLICT"}
                if outcome["kind"] == "noop":
                    # A NOOP decision replays verbatim, including the current
                    # observed at decision time; the retry never re-evaluates
                    # capacity, so it can never execute a new operation.
                    self._record("IDEMPOTENT_REPLAY", "NOOP: returned the original response without re-evaluating")
                    return dict(outcome["response"])
                request = self.requests[outcome["request_id"]]
                self._record("IDEMPOTENT_REPLAY", f"{outcome['request_id']}: returned the original operation")
                return {"http": 200, "status": request["status"], "request_id": outcome["request_id"]}
        if self.active_request or self.paused:
            self._record("REJECT_409", "Another lifecycle operation is active or cleanup is unresolved")
            return {"http": 409, "status": "CONFLICT"}
        if (direction == "out" and target <= self.current) or (direction == "in" and target >= self.current):
            # A keyed NOOP is a decision about this request, so it is recorded
            # like any other decision. Otherwise a retry after a capacity change
            # would execute a new operation the client never asked for.
            response = {"http": 200, "status": "NOOP", "current": self.current}
            if idempotency_key is not None:
                self.idempotency[record_key] = (fingerprint, {"kind": "noop", "response": response})
                self._record("NOOP", f"{direction} target {target} already satisfied; response recorded for replay")
            else:
                self._record("NOOP", f"{direction} target {target} already satisfied")
            return response
        request_id = f"req-{self.next_id:02d}"
        self.next_id += 1
        self.requests[request_id] = {
            "direction": direction,
            "target": target,
            "starting_current": self.current,
            "model": model,
            "timeout_secs": timeout_secs,
            "status": "PENDING",
            "idempotency_key": idempotency_key,
        }
        if idempotency_key is not None:
            # The record is written together with the admission decision, so a
            # concurrent duplicate of this first request can only replay or 409.
            self.idempotency[record_key] = (fingerprint, {"kind": "operation", "request_id": request_id})
        self.active_request = request_id
        self._record("PENDING", f"{direction} to absolute target {target}", request_id=request_id)
        return {"http": 200, "status": "PENDING", "request_id": request_id}

    def get_status(self, request_id: str) -> dict[str, Any]:
        """Expose the public status lookup shape, including the unknown-ID
        case."""
        request = self.requests.get(request_id)
        if request is None:
            self._record("NOT_FOUND", f"{request_id}: request ID is unknown")
            return {"http": 404, "status": "NOT_FOUND", "request_id": request_id}
        return {
            "http": 200,
            "request_id": request_id,
            "direction": request["direction"],
            "status": request["status"],
            "current": self.current,
            "ready": self.ready,
        }

    def _request(self, request_id: str, direction: str) -> dict[str, Any]:
        if self.active_request != request_id or self.requests[request_id]["direction"] != direction:
            raise ValueError("request is not active for this direction")
        return self.requests[request_id]

    def scale_out(self, request_id: str, *, health_ok: bool | tuple[bool, ...] = True, cleanup_ok: bool = True) -> str:
        request = self._request(request_id, "out")
        attempt = 0
        while self.current < request["target"]:
            number = len([r for r in self.replicas.values() if r.owner == "manager"]) + 1
            replica_id, pg = f"genrm-{number}", f"manager-pg-{number}"
            candidate = Replica(replica_id, "manager", pg, healthy=False, published=False, admission_open=False)
            self.replicas[replica_id] = candidate
            self.pg_live.add(pg)
            request["status"] = "CREATING"
            self._record("PG_ALLOCATED", f"{replica_id}: manager owns {pg}; two mock workers created")
            request["status"] = "HEALTH_CHECKING"
            self._record("HEALTH_CHECKING", f"{replica_id}: health result pending; route unchanged")
            candidate_healthy = health_ok[attempt] if isinstance(health_ok, tuple) else health_ok
            attempt += 1
            if not candidate_healthy:
                request["status"] = "PARTIAL" if self.current > request["starting_current"] else "FAILED"
                if cleanup_ok:
                    candidate.released = True
                    self.pg_live.remove(pg)
                    self._record("ROLLBACK", f"{replica_id}: candidate and manager PG released")
                else:
                    self.paused = True
                    self._record("CLEANUP_PENDING", f"{replica_id}: cleanup failed; PG retained")
                self.active_request = None
                return request["status"]
            candidate.healthy = True
            candidate.admission_open = True
            candidate.published = True
            request["status"] = "READY"
            self._record("PUBLISHED", f"{replica_id}: healthy head enters route; no weight sync")
        request["status"] = "ACTIVE"
        self.active_request = None
        self._record("ACTIVE", "Scale-out finished", request_id=request_id)
        return "ACTIVE"

    def admit(self, replica_id: str, call_id: str) -> bool:
        replica = self.replicas[replica_id]
        if not replica.admission_open or not replica.published:
            self._record("REJECT_STALE_ROUTE", f"{call_id}: {replica_id} did not accept request")
            return False
        replica.inflight.add(call_id)
        self._record("ADMITTED", f"{call_id}: accepted by {replica_id}")
        return True

    def complete(self, replica_id: str, call_id: str) -> None:
        self.replicas[replica_id].inflight.remove(call_id)
        self._record("COMPLETED_CALL", f"{call_id}: accepted call finished")

    def close_admission(self, request_id: str) -> str:
        request = self._request(request_id, "in")
        candidates = [r for r in self.replicas.values() if r.owner == "manager" and r.published]
        candidate = candidates[-1]
        candidate.admission_open = False
        candidate.published = False
        request["status"] = "DRAINING"
        request["victim"] = candidate.replica_id
        self._record("DRAINING", f"{candidate.replica_id}: route removed, admission closed; accepted calls retained")
        return candidate.replica_id

    def finish_scale_in(self, request_id: str, *, backend_idle: bool = True, cleanup_ok: bool = True) -> str:
        request = self._request(request_id, "in")
        candidate = self.replicas[request["victim"]]
        if candidate.inflight or candidate.backend_busy or not backend_idle:
            self._record("WAIT_DRAIN", f"{candidate.replica_id}: accepted calls or backend still busy")
            return "DRAINING"
        request["status"] = "REMOVING"
        self._record("REMOVING", f"{candidate.replica_id}: backend idle; stopping all mock workers")
        if not cleanup_ok:
            request["status"] = "FAILED"
            self.paused = True
            self.active_request = None
            self._record("CLEANUP_PENDING", f"{candidate.replica_id}: manager PG retained for retry")
            return "FAILED"
        candidate.released = True
        self.pg_live.remove(candidate.pg)
        if self.current > request["target"]:
            request["status"] = "PENDING"
            self._record("RELEASED", f"{candidate.replica_id}: manager PG released; continuing to target")
            return "PENDING"
        request["status"] = "COMPLETED"
        self.active_request = None
        self._record("RELEASED", f"{candidate.replica_id}: all workers stopped; manager PG released")
        return "COMPLETED"

    def fail_drain(self, request_id: str) -> None:
        request = self._request(request_id, "in")
        candidate = self.replicas[request["victim"]]
        if candidate.admission_open or candidate.released:
            raise ValueError("drain failure requires a retained closed replica")
        request["status"] = "FAILED"
        self.paused = True
        self.active_request = None
        self._record("DRAIN_FAILED", f"{candidate.replica_id}: retained workers and PG; route remains closed")

    def retry_scale_in(self, request_id: str, *, backend_idle: bool = True, cleanup_ok: bool = True) -> str:
        """Resume a failed drain only after the caller explicitly retries
        it."""
        request = self.requests.get(request_id)
        if request is None or request["direction"] != "in" or request["status"] != "FAILED":
            raise ValueError("request is not a failed scale-in operation")
        if not self.paused or self.active_request is not None:
            raise ValueError("scale-in retry is not waiting for reconciliation")
        self.paused = False
        self.active_request = request_id
        self._record("RETRY", f"{request_id}: retrying retained drain and cleanup")
        return self.finish_scale_in(request_id, backend_idle=backend_idle, cleanup_ok=cleanup_ok)

    def retry_cleanup(self, request_id: str) -> str:
        """Reconcile unpublished manager candidates left by a failed
        allocation."""
        request = self.requests.get(request_id)
        if request is None or request["direction"] != "out" or request["status"] not in {"FAILED", "PARTIAL"}:
            raise ValueError("request is not waiting for allocation cleanup")
        if not self.paused or self.active_request is not None:
            raise ValueError("allocation cleanup is not waiting for reconciliation")
        pending = [
            replica
            for replica in self.replicas.values()
            if replica.owner == "manager" and not replica.published and not replica.released
        ]
        for replica in pending:
            replica.released = True
            self.pg_live.discard(replica.pg)
        self.paused = False
        self._record("RECONCILED", f"{request_id}: released {len(pending)} retained candidate PG(s)")
        return "RECONCILED"

    def late_health_ack(self, replica_id: str) -> bool:
        candidate = self.replicas[replica_id]
        if candidate.released or self.paused or not self.active_request:
            self._record("LATE_ACK_IGNORED", f"{replica_id}: stale result cannot publish a route")
            return False
        raise ValueError("candidate still belongs to an active operation")


def scripted_scenarios() -> dict[str, Any]:
    normal = ContractDemo()
    out = normal.request("out", 2, idempotency_key="scale-to-two")
    replay = normal.request("out", 2, idempotency_key="scale-to-two")
    key_conflict = normal.request("out", 3, idempotency_key="scale-to-two")
    conflict = normal.request("out", 2)
    normal.scale_out(out["request_id"])
    normal.admit("genrm-1", "score-17")
    inside = normal.request("in", 1)
    normal.close_admission(inside["request_id"])
    stale = normal.admit("genrm-1", "stale-dispatch")
    waiting = normal.finish_scale_in(inside["request_id"])
    normal.complete("genrm-1", "score-17")
    normal.finish_scale_in(inside["request_id"])
    noop = normal.request("in", 1)
    protected = normal.request("in", 0)

    health_failure = ContractDemo()
    failed = health_failure.request("out", 2)
    health_failure.scale_out(failed["request_id"], health_ok=False)
    late_ack = health_failure.late_health_ack("genrm-1")

    busy_backend = ContractDemo()
    started = busy_backend.request("out", 2)
    busy_backend.scale_out(started["request_id"])
    inward = busy_backend.request("in", 1)
    busy_backend.close_admission(inward["request_id"])
    busy_backend.finish_scale_in(inward["request_id"], backend_idle=False)
    busy_backend.fail_drain(inward["request_id"])

    cleanup_failure = ContractDemo()
    failure = cleanup_failure.request("out", 2)
    cleanup_failure.scale_out(failure["request_id"], health_ok=False, cleanup_ok=False)
    after_failure = cleanup_failure.request("out", 2)
    cleanup_retry = cleanup_failure.retry_cleanup(failure["request_id"])
    retry_request = cleanup_failure.request("out", 2)
    cleanup_failure.scale_out(retry_request["request_id"])
    # A keyed NOOP is recorded; after capacity changes, the same key and body
    # must replay the original NOOP instead of executing a new scale-out.
    noop_at_two = cleanup_failure.request("out", 2, idempotency_key="recheck")
    back_in = cleanup_failure.request("in", 1)
    cleanup_failure.close_admission(back_in["request_id"])
    cleanup_failure.finish_scale_in(back_in["request_id"])
    noop_replay = cleanup_failure.request("out", 2, idempotency_key="recheck")
    unknown_status = health_failure.get_status("req-99")
    failed_drain_keeps_pg = busy_backend.paused and "manager-pg-1" in busy_backend.pg_live
    drain_retry = busy_backend.retry_scale_in(inward["request_id"])

    return {
        "scope": "Deterministic contract simulation; no Ray, SGLang, GPU, HTTP server, or Task 3 integration",
        "scenarios": {
            "1→2→1": normal.events,
            "Health failure": health_failure.events,
            "Backend busy": busy_backend.events,
            "Cleanup failure": cleanup_failure.events,
        },
        "checks": {
            "inflight_conflict": conflict["http"] == 409,
            "idempotent_replay": replay.get("request_id") == out["request_id"],
            "idempotency_key_conflict": key_conflict["http"] == 409,
            "late_dispatch_rejected": stale is False,
            "late_health_ack_ignored": late_ack is False,
            "accepted_request_drains": waiting == "DRAINING" and not normal.replicas["genrm-1"].inflight,
            "absolute_noop": noop["status"] == "NOOP",
            "initial_protected": protected["http"] == 400 and "training-pg" in normal.pg_live,
            "cleanup_failure_blocks_new_scale": after_failure["http"] == 409,
            "failed_drain_keeps_pg": failed_drain_keeps_pg,
            "cleanup_retry_reconciles": cleanup_retry == "RECONCILED"
            and "manager-pg-1" not in cleanup_failure.pg_live,
            "drain_retry_completes": drain_retry == "COMPLETED" and not busy_backend.paused,
            "keyed_noop_replayed_verbatim": noop_replay["status"] == "NOOP"
            and noop_replay.get("current") == 2
            and "request_id" not in noop_replay,
            "keyed_noop_blocks_new_execution": noop_at_two["status"] == "NOOP" and cleanup_failure.current == 1,
            "unknown_request_404": unknown_status["http"] == 404,
            "final_capacity": [normal.current, normal.ready],
            "manager_pg_released": "manager-pg-1" not in normal.pg_live,
            "no_weight_sync": all(event["kind"] != "WEIGHT_SYNCING" for event in normal.events),
        },
    }


if __name__ == "__main__":
    result = scripted_scenarios()
    path = Path(__file__).with_name("results") / "contract-demo.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["checks"], ensure_ascii=False, indent=2))
    print(path)
