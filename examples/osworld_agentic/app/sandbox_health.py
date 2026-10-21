# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""OSWorld-only, best-effort lifecycle journal and run-cumulative reduction.

No provider queries, threads or files are created on import. Query adapters
must return released/exists/unknown and must only query the supplied owned ID.
"""

from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def enabled() -> bool:
    return os.environ.get("OSWORLD_SANDBOX_HEALTH") == "1"


class Journal:
    """One writer/file per process; producers never wait for shared storage."""

    def __init__(self, root: str, run_id: str, session_id: str) -> None:
        self.root, self.run_id, self.session_id = Path(root), run_id, session_id
        self.producer = uuid.uuid4().hex
        self.pending: queue.Queue = queue.Queue(maxsize=4096)
        self.attempt = 0
        self.thread = threading.Thread(target=self._write, daemon=True, name="osworld-health-writer")
        self.thread.start()

    def emit(self, kind: str, **fields: Any) -> None:
        try:
            self.pending.put_nowait(
                dict(
                    run_id=self.run_id,
                    session_id=self.session_id,
                    event_id=uuid.uuid4().hex,
                    time=time.time(),
                    attempt=self.attempt,
                    kind=kind,
                    **fields,
                )
            )
        except Exception:
            logger.warning("Sandbox health event lost: %s", kind, exc_info=True)

    def _write(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / f"{self.producer}.jsonl").open("a", encoding="utf-8") as handle:
                while True:
                    event = self.pending.get()
                    try:
                        handle.write(json.dumps(event) + "\n")
                        handle.flush()
                    except Exception:
                        logger.warning("Sandbox health journal write failed", exc_info=True)
                    finally:
                        self.pending.task_done()
        except Exception:
            logger.warning("Sandbox health journal unavailable", exc_info=True)

    def flush(self, timeout: float = 0.2) -> None:
        deadline = time.monotonic() + timeout
        while self.pending.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)


_journal: Journal | None = None


def journal() -> Journal | None:
    global _journal
    if not enabled():
        return None
    if _journal is None:
        try:
            _journal = Journal(
                os.environ["OSWORLD_HEALTH_DIR"], os.environ["OSWORLD_RUN_TAG"], os.environ["RELAX_SESSION_ID"]
            )
        except Exception:
            logger.warning("Sandbox health disabled: missing journal configuration", exc_info=True)
    return _journal


def emit(kind: str, **fields: Any) -> None:
    writer = journal()
    if writer is not None:
        writer.emit(kind, **fields)


def observe_nex(writer: Journal, client: Any) -> None:
    """Observe this dedicated Agent process's Nex calls; preserve all behavior.

    Intercept create's successful return before DesktopEnv startup can fail.
    Internal reset recreations are captured too. No workspace enumeration.
    """
    if getattr(client, "_relax_health_observed", False):
        return
    create, delete = client.create_sandbox, client.delete_sandbox

    def create_observed(*args: Any, **kwargs: Any) -> Any:
        token = uuid.uuid4().hex
        writer.emit("creating", creation_id=token)
        try:
            sid = create(*args, **kwargs)
        except BaseException:
            writer.emit("creation_unknown", creation_id=token, error_category="unknown")
            raise
        writer.emit("created", creation_id=token, sandbox_id=str(sid))
        return sid

    def delete_observed(sid: str, *args: Any, **kwargs: Any) -> Any:
        writer.emit("closing", sandbox_id=str(sid))
        try:
            result = delete(sid, *args, **kwargs)
        except BaseException:
            writer.emit("close_error", sandbox_id=str(sid), error_category="cleanup_error")
            raise
        writer.emit("close_returned", sandbox_id=str(sid))
        return result

    client.create_sandbox, client.delete_sandbox = create_observed, delete_observed
    client._relax_health_observed = True


class Aggregate:
    """Deduplicate events, retaining attempts and unresolved creations across
    restarts."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.events: dict[str, dict] = {}
        self.offsets: dict[Path, int] = {}
        self.observations: list[dict] = []

    def read(self, root: Path) -> None:
        for path in root.glob("*.jsonl"):
            if path.stat().st_size == self.offsets.get(path, -1):
                continue
            with path.open(encoding="utf-8") as handle:
                handle.seek(self.offsets.get(path, 0))
                while line := handle.readline():
                    if not line.endswith("\n"):
                        break  # A live writer or a process killed during its last write.
                    try:
                        self.add(json.loads(line))
                    except (ValueError, KeyError, TypeError):
                        logger.warning("Invalid sandbox health event in %s", path.name)
                    self.offsets[path] = handle.tell()

    def add(self, event: dict) -> None:
        if event["run_id"] == self.run_id:
            if event["kind"] == "heartbeat":
                key = f"heartbeat:{event['session_id']}"
                if event["time"] > self.events.get(key, {}).get("time", -1):
                    self.events[key] = event
            else:
                self.events.setdefault(event["event_id"], event)

    def snapshot(self, now: float, grace: float, stale: float, query: Callable[[str], str] | None = None) -> dict:
        sessions: dict[str, list] = defaultdict(list)
        for event in self.events.values():
            sessions[event["session_id"]].append(event)
        counts = defaultdict(int)
        durations, anomalies = [], []
        cleanup = defaultdict(int)
        for session_id, events in sessions.items():
            events.sort(key=lambda e: (e["time"], e["event_id"]))
            kinds = {e["kind"] for e in events}
            ready = next((e for e in events if e["kind"] == "ready"), None)
            done = next((e for e in events if e["kind"] == "init_done"), None)
            end = next((e for e in events if e["kind"] == "session_end"), None)
            counts["sessions"] += 1
            if done:
                counts["init_completed"] += 1
                counts["first_ready"] += int(ready is not None and ready["attempt"] == 1)
                counts["init_failed"] += int(done.get("outcome") == "exhausted")
                counts["unhealthy_reset"] += int(done.get("outcome") == "invalid_screenshot")
                if any(e["kind"] == "attempt_failed" and e["attempt"] == 1 for e in events):
                    counts["retry_completed"] += 1
                    counts["retry_recovered"] += int(ready is not None)
            else:
                counts["init_incomplete"] += 1
            if ready:
                durations.append(ready["ready_seconds"])
                if end:
                    counts["ready_ended"] += 1
                    confirmed = [
                        e
                        for e in events
                        if e["kind"] == "runtime_failure"
                        and not any(
                            c["kind"] == "closing"
                            and c.get("sandbox_id") == e.get("sandbox_id")
                            and c["time"] <= e["time"]
                            for c in events
                        )
                    ]
                    counts["runtime_failed"] += int(bool(confirmed))
                    counts["runtime_unknown_ended"] += int(any(e.get("error_category") == "unknown" for e in events))
            if end:
                counts[end.get("outcome", "unknown")] += 1
            counts["init_timeout"] += int(any(e.get("error_category") == "init_timeout" for e in events))
            counts["unknown_errors"] += int(any(e.get("error_category") == "unknown" for e in events))
            if "observer_unavailable" in kinds:
                counts["identity_observer_unavailable"] += 1
            instances: dict[str, list] = defaultdict(list)
            creations = {}
            for event in events:
                if event["kind"] == "creating":
                    creations[event["creation_id"]] = event
                if event["kind"] == "created":
                    creations.pop(event.get("creation_id"), None)
                if event.get("sandbox_id"):
                    instances[event["sandbox_id"]].append(event)
            for event in creations.values():
                counts["unknown_identity"] += 1
                anomalies.append(self._row(session_id, None, event, now, "unknown", "identity_unavailable"))
            for sid, history in instances.items():
                last = history[-1]
                closing = next((e for e in history if e["kind"] == "closing"), None)
                status = "not_requested"
                if query and ready and not end and not closing and ready.get("sandbox_id") == sid:
                    try:
                        if query(sid) == "released":
                            proof = dict(
                                run_id=self.run_id,
                                session_id=session_id,
                                event_id=f"runtime_failure:{session_id}",
                                time=now,
                                attempt=ready["attempt"],
                                sandbox_id=sid,
                                kind="runtime_failure",
                                error_category="confirmed_missing",
                            )
                            if proof["event_id"] not in self.events:
                                self.add(proof)
                                self.observations.append(proof)
                    except Exception:
                        logger.warning("Sandbox runtime query failed for owned instance %s", sid, exc_info=True)
                if closing:
                    status = "unavailable" if query is None else "unknown"
                    if query:
                        try:
                            answer = query(sid)
                            status = answer if answer in {"released", "exists", "unknown"} else "unknown"
                        except Exception:
                            logger.warning("Sandbox health query failed for owned instance %s", sid, exc_info=True)
                    if status == "released":
                        cleanup["released"] += 1
                    elif now - closing["time"] >= grace:
                        cleanup["unconfirmed"] += 1
                        cleanup[status] += 1
                    else:
                        cleanup["grace_pending"] += 1
                stage = "closed" if status == "released" else ("closing" if closing else last["kind"])
                if stage != "closed":
                    if not closing and (end or now - events[-1]["time"] > stale):
                        stage = "unknown" if not end else end.get("outcome", "unknown")
                    row = self._row(session_id, sid, closing or last, now, stage, status)
                    row["error_category"] = next(
                        (e["error_category"] for e in reversed(history) if e.get("error_category")),
                        next((e["error_category"] for e in reversed(events) if e.get("error_category")), ""),
                    )
                    anomalies.append(row)
            if not instances and not creations or (not end and now - events[-1]["time"] > stale):
                last = next((e for e in reversed(events) if e["kind"] != "heartbeat"), events[-1])
                stage = (
                    end.get("outcome", "unknown")
                    if end
                    else "unknown"
                    if now - events[-1]["time"] > stale
                    else last["kind"]
                )
                anomalies.append(self._row(session_id, None, last, now, stage, "unknown"))
        summary = []
        scalars = {}
        for name, numerator, denominator in (
            ("first_ready_rate", "first_ready", "init_completed"),
            ("retry_recovery_rate", "retry_recovered", "retry_completed"),
            ("init_failure_rate", "init_failed", "init_completed"),
            ("runtime_failure_rate", "runtime_failed", "ready_ended"),
        ):
            n, d = counts[numerator], counts[denominator]
            value = n / d if d else None
            summary.append(
                dict(
                    metric=f"sandbox/{name}",
                    value=value,
                    numerator=n,
                    denominator=d,
                    status=(
                        "partial_unknown"
                        if name == "runtime_failure_rate" and counts["runtime_unknown_ended"]
                        else "ok"
                    )
                    if d
                    else "no_data",
                )
            )
            if value is not None:
                scalars[f"sandbox/{name}"] = value
        p95 = sorted(durations)[math.ceil(len(durations) * 0.95) - 1] if len(durations) >= 2 else None
        summary.append(
            dict(
                metric="sandbox/ready_seconds_p95",
                value=p95,
                denominator=len(durations),
                status="ok" if p95 is not None else "insufficient_samples",
            )
        )
        if p95 is not None:
            scalars["sandbox/ready_seconds_p95"] = p95
        # Unknown allocation identities make the total unknowable; preserve a lower bound in the table.
        value = (
            cleanup["unconfirmed"]
            if not counts["unknown_identity"] and not counts["identity_observer_unavailable"]
            else None
        )
        if query is None and not value:
            value = None  # Unavailable confirmation must never look like a healthy zero.
        summary.append(
            dict(
                metric="sandbox/cleanup_unconfirmed",
                value=value,
                status="confirmation_unavailable" if query is None else "unknown_identity" if value is None else "ok",
                **dict(cleanup),
                unknown_identity=counts["unknown_identity"],
            )
        )
        if value is not None:
            scalars["sandbox/cleanup_unconfirmed"] = value
        summary.append(dict(metric="sandbox/session_accounting", **dict(counts)))
        return dict(scalars=scalars, summary=summary, instances=anomalies)

    def _row(self, session_id: str, sid: str | None, event: dict, now: float, stage: str, cleanup: str) -> dict:
        return dict(
            run_id=self.run_id,
            session_id=session_id,
            sandbox_id=sid,
            attempt=event["attempt"],
            stage=stage,
            stage_age_seconds=max(0, now - event["time"]),
            error_category=event.get("error_category", ""),
            cleanup_status=cleanup,
        )
