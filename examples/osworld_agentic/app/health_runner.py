# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in OSWorld entrypoints; no new ClearML Task or provider mutation."""

from __future__ import annotations

import fcntl
import json
import os
import runpy
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from .sandbox_health import Aggregate, enabled, journal, logger


def report(snapshot: dict, step: int) -> None:
    # Use the existing training Task through its MetricsService. Never instantiate
    # a direct ClearML adapter here (that would create a second Task).
    from relax.utils.metrics.client import MetricsClient
    from relax.utils.metrics.metrics_service_adapter import get_metrics_service_adapter

    adapter = get_metrics_service_adapter()
    if adapter is None:
        raise RuntimeError("Training metrics service is not ready")
    client = MetricsClient(adapter.client.service_url)
    payload = dict(snapshot["scalars"])
    rows = sorted(snapshot["instances"], key=lambda row: row["stage_age_seconds"], reverse=True)
    payload["sandbox/summary"] = {
        "__relax_table__": [
            *snapshot["summary"],
            {"metric": "sandbox/instance_table", "total_rows": len(rows), "displayed_rows": min(len(rows), 200)},
        ]
    }
    payload["sandbox/instances"] = {"__relax_table__": rows[:200]}
    result = client.report_backend("clearml", step, payload)
    if result.get("status") != "success":
        raise RuntimeError("Sandbox health metrics report failed")


def report_rollout(step: int) -> None:
    """Publish the latest local snapshot once at the rollout reporting
    boundary."""
    if not enabled():
        return
    try:
        path = Path(os.environ["OSWORLD_HEALTH_DIR"]) / "snapshot.json"
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        report(snapshot, step)
    except Exception:
        logger.warning("Sandbox rollout reporting failed; local snapshot retained", exc_info=True)


def monitor(stop: threading.Event, query: Callable[[str], str] | None = None) -> None:
    try:
        root = Path(os.environ["OSWORLD_HEALTH_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        # A run has exactly one reporter, including accidentally duplicated entrypoints.
        with (root / "reporter.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            interval = max(1.0, float(os.environ.get("OSWORLD_HEALTH_INTERVAL", "30")))
            grace = max(0.0, float(os.environ.get("OSWORLD_HEALTH_CLEANUP_GRACE", "120")))
            aggregate = Aggregate(os.environ["OSWORLD_RUN_TAG"])
            if query is None:
                logger.warning("Sandbox cleanup confirmation unavailable; close return is not release evidence")
            started = time.monotonic()
            while True:
                final = stop.is_set()
                try:
                    aggregate.read(root)
                    snapshot = aggregate.snapshot(time.time(), grace, max(90.0, 3 * interval), query)
                    snapshot["scalars"]["sandbox/elapsed_seconds"] = time.monotonic() - started
                    if aggregate.observations:
                        with (root / "reporter-events.jsonl").open("a", encoding="utf-8") as handle:
                            for proof in aggregate.observations:
                                handle.write(json.dumps(proof) + "\n")
                        aggregate.observations.clear()
                    temporary = root / "snapshot.tmp"
                    temporary.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
                    temporary.replace(root / "snapshot.json")
                except Exception:
                    logger.warning("Sandbox health reporting failed; journal retained", exc_info=True)
                if final:
                    break
                stop.wait(interval)
    except Exception:
        logger.warning("Sandbox health monitor unavailable", exc_info=True)


def run_agent(command: list[str]) -> int:
    writer = journal()
    if writer is None:
        os.execvp(command[0], command)
    writer.emit("session_start")
    try:
        child = subprocess.Popen(command)
    except Exception:
        logger.warning("Health supervisor unavailable; running Agent directly", exc_info=True)
        writer.emit("supervisor_unavailable", error_category="unknown")
        writer.flush()
        os.execvp(command[0], command)
    cancelled = False

    def forward(signum: int, _frame: object) -> None:
        nonlocal cancelled
        cancelled = True
        writer.emit("cancel_requested", error_category="cancelled")
        # Relax terminates the entire process group; the child already receives
        # this signal. Forwarding again could interrupt its in-progress cleanup.

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, forward)
    while True:
        try:
            code = child.wait(timeout=15)
            break
        except subprocess.TimeoutExpired:
            writer.emit("heartbeat")
    outcome = (
        "cancelled"
        if cancelled or code in (-signal.SIGTERM, -signal.SIGINT)
        else "completed"
        if code == 0
        else "process_exit"
    )
    writer.emit("session_end", outcome=outcome, error_category="unknown" if code else "", exit_code=code)
    writer.flush()
    return code if code >= 0 else 128 - code


def finish_monitor(stop: threading.Event, worker: threading.Thread | None) -> None:
    stop.set()
    if worker is not None:
        worker.join(timeout=6)
        if worker.is_alive():
            logger.warning("Sandbox final flush timed out; consult retained journal/snapshot")


def main() -> None:
    if "--agent" in sys.argv[1:2]:
        raise SystemExit(run_agent(sys.argv[2:]))
    stop = threading.Event()
    worker = None
    if enabled():
        from relax.utils.tracking_utils import register_observer_finalizer

        worker = threading.Thread(target=monitor, args=(stop,), daemon=True, name="osworld-health-reporter")
        worker.start()
        register_observer_finalizer(lambda: finish_monitor(stop, worker))
    try:
        runpy.run_module("relax.entrypoints.train", run_name="__main__")
    finally:
        finish_monitor(stop, worker)


if __name__ == "__main__":
    main()
