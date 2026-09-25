# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Training-continuity E2E monitor for the GenRM elastic-scaling acceptance.

A sidecar to a running training job (launched via
``scripts/training/genrm/run-qwen3-0.6B-4xgpu-genrm-continuity.sh``): it waits
for the first training step, drives ``scale_out`` -> ``scale_in`` against the
live GenRM service, and asserts that training steps, rollouts and reward
scoring keep advancing through both scaling windows.

Everything the driver asserts comes from two independent sources: the job
driver log (step/rollout lines with timestamps) and the service itself
(``/genrm/engines`` capacity and per-engine ``served`` counters).
"""

import argparse
import json
import os
import re
import threading
import time
from datetime import datetime

import requests


GENRM_BASE = None
EVENTS = []
LOCK = threading.Lock()
TRAIN_EVENTS = []  # (epoch_ts, kind, detail)
LOG_PATH = None
TAIL_STOP = threading.Event()

STEP_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\b(step \d+:|rollout \d+:|All rollouts finished|All training steps finished)"
)


def log_event(event: str, **kw):
    with LOCK:
        EVENTS.append({"t": round(time.time(), 3), "event": event, **kw})


def http_get(path: str, timeout: float = 60):
    r = requests.get(f"{GENRM_BASE}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def http_post(path: str, body: dict, timeout: float = 600):
    r = requests.post(f"{GENRM_BASE}{path}", json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"{path} -> {r.status_code}: {r.text[:200]}")
    return r.json()


def engines() -> dict:
    snap = http_get("/engines")
    return {"current": snap.get("current"), "engines": snap.get("engines") or []}


def tail_log():
    """Tail the job driver log, recording train events with epoch
    timestamps."""
    inode = None
    fh = None
    pos = 0
    while not TAIL_STOP.is_set():
        try:
            if fh is None or inode != os.stat(LOG_PATH).st_ino:
                if fh is not None:
                    fh.close()
                inode = os.stat(LOG_PATH).st_ino
                fh = open(LOG_PATH, errors="replace")
                pos = 0
        except OSError:
            time.sleep(2.0)
            continue
        fh.seek(pos)
        for line in fh:
            m = STEP_RE.search(line)
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
                with LOCK:
                    TRAIN_EVENTS.append((ts, m.group(2).strip().rstrip(":")))
        pos = fh.tell()
        time.sleep(1.0)


def wait_for_train_event(pattern: str, timeout_s: float, from_ts: float = 0.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with LOCK:
            hit = any(kind.startswith(pattern) and ts >= from_ts for ts, kind in TRAIN_EVENTS)
        if hit:
            return True
        time.sleep(2.0)
    return False


def poll_status(direction: str, rid: str, terminals: set, timeout_s: float) -> dict:
    deadline = time.time() + timeout_s
    status = None
    while time.time() < deadline:
        status = http_get(f"/{direction}/{rid}")
        if status["status"] in terminals:
            return status
        time.sleep(1.0)
    raise RuntimeError(
        f"{direction} {rid} did not reach {terminals} in {timeout_s}s (last={status and status['status']})"
    )


def engines_sampler():
    """Record capacity/served/inflight snapshots while the run is live."""
    while not TAIL_STOP.is_set():
        try:
            snap = engines()
            log_event(
                "engines_snapshot",
                current=snap["current"],
                served={f"{e['host']}:{e['port']}": e.get("served", 0) for e in snap["engines"]},
                inflight={f"{e['host']}:{e['port']}": e.get("inflight", 0) for e in snap["engines"]},
            )
        except Exception as exc:  # noqa: BLE001
            log_event("engines_snapshot_failed", error=f"{type(exc).__name__}: {exc}"[:120])
        time.sleep(2.0)


def periodic_dump(out_dir: str):
    """Dump event evidence every 10 s so a monitor crash never loses it.

    Lesson from run 3: the controller tears the serve apps down when training
    finishes, and any unwrapped late poll crashes the monitor before its final
    dump. Incremental dumps make the timeline survive; verdicts.json stays
    untouched until the final verdict dump.
    """
    while not TAIL_STOP.is_set():
        try:
            with LOCK:
                events = list(EVENTS)
                train_events = [{"ts": ts, "kind": kind} for ts, kind in sorted(TRAIN_EVENTS)]
            with open(os.path.join(out_dir, "events.json"), "w") as f:
                json.dump(events, f, indent=2, ensure_ascii=False)
            with open(os.path.join(out_dir, "train_events.json"), "w") as f:
                json.dump(train_events, f, indent=2, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(10.0)


def main() -> int:
    global GENRM_BASE, LOG_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-log", required=True, help="path to the ray job driver log")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-rollout", type=int, default=8)
    args = parser.parse_args()

    LOG_PATH = args.job_log
    os.makedirs(args.out_dir, exist_ok=True)
    GENRM_BASE = "http://127.0.0.1:8000/genrm"
    log_event("monitor_start", job_log=LOG_PATH, genrm=GENRM_BASE)

    threading.Thread(target=tail_log, daemon=True).start()
    threading.Thread(target=engines_sampler, daemon=True).start()
    threading.Thread(target=lambda: periodic_dump(args.out_dir), daemon=True).start()

    verdicts: dict = {}
    # Boot transient: the app may 404 until the route is registered and the
    # initial engine may take minutes; poll /engines until capacity 1.
    deadline = time.time() + 900
    base = None
    while time.time() < deadline:
        try:
            snap = engines()
            if snap["current"] == 1:
                base = snap
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3.0)
    if base is None:
        log_event("monitor_error", error="initial genrm engine never reached capacity 1")
        dump(args.out_dir, verdicts, functional_error="initial engine never up")
        return 1
    initial_ids = {(e["host"], e["port"]) for e in base["engines"]}
    log_event("baseline_engines", ids=sorted(map(list, initial_ids)))

    # Wait for the first training step: scaling must land inside live training.
    if not wait_for_train_event("step", 1800):
        log_event("monitor_error", error="no training step observed within 1800s")
        dump(args.out_dir, verdicts, functional_error="no training step observed")
        return 1
    log_event("first_step_seen")

    # ---- scale-out window ---- #
    so = http_post("/scale_out", {"num_replicas": 2, "timeout_secs": 900.0})
    rid = so["request_id"]
    log_event("scale_out_submitted", request_id=rid)
    so_final = poll_status("scale_out", rid, {"ACTIVE", "PARTIAL", "FAILED"}, 900)
    so_active_ts = time.time()
    log_event("scale_out_final", status=so_final["status"])
    if so_final["status"] != "ACTIVE":
        dump(args.out_dir, verdicts, functional_error=f"scale-out ended {so_final['status']}")
        return 1

    # Two more training events must land while the elastic engine is serving.
    wait_for_train_event("rollout", 600, from_ts=so_active_ts - 1)
    wait_for_train_event("step", 600, from_ts=so_active_ts - 1)

    # Hold scale-in until the elastic engine has actually scored something
    # (the reward path through the elastic replica is part of the claim).
    elastic = None
    deadline = time.time() + 600
    while time.time() < deadline:
        snap = engines()
        served = {
            f"{e['host']}:{e['port']}": e.get("served", 0)
            for e in snap["engines"]
            if (e["host"], e["port"]) not in initial_ids
        }
        if served and sum(served.values()) > 0:
            elastic = served
            break
        time.sleep(3.0)
    log_event("elastic_served_check", served=elastic)

    # ---- scale-in window ---- #
    si = http_post("/scale_in", {"num_replicas": 1, "timeout_secs": 600.0})
    rid_si = si["request_id"]
    log_event("scale_in_submitted", request_id=rid_si)
    si_final = poll_status("scale_in", rid_si, {"COMPLETED", "FAILED"}, 600)
    si_done_ts = time.time()
    log_event("scale_in_final", status=si_final["status"])

    # Training must run to completion after the scale-in.
    finished = wait_for_train_event("All training steps finished", 1800, from_ts=si_done_ts - 1)
    time.sleep(5)
    # The controller tears the serve apps down once training finishes; late
    # polls must degrade to the last sampled snapshot instead of crashing
    # (run-3 lesson: an unwrapped final poll lost the whole verdict).
    try:
        final = engines()
    except Exception as exc:  # noqa: BLE001
        log_event("final_engines_unavailable", error=f"{type(exc).__name__}: {exc}"[:120])
        with LOCK:
            last = next((e for e in reversed(EVENTS) if e.get("event") == "engines_snapshot"), None)
        # Rebuild the engine list from the last good snapshot's served keys so
        # the final-capacity assertion still sees engine identities (run-5
        # lesson: an empty fallback list made it fail unconditionally).
        engines_from_snap = []
        for key in (last or {}).get("served") or {}:
            host, _, port = key.rpartition(":")
            engines_from_snap.append({"host": host, "port": int(port)})
        final = {"current": (last or {}).get("current"), "engines": engines_from_snap}
        verdicts["final_engines_from_last_snapshot"] = True
    final_ids = {(e["host"], e["port"]) for e in final["engines"]}

    with LOCK:
        events = list(TRAIN_EVENTS)
    steps = sorted(ts for ts, kind in events if kind.startswith("step"))
    # Each rollout index logs two lines in the job driver log (batch start
    # and result); count unique indices, not raw lines, and require the exact
    # expected set (run-6 lesson: the raw line count double-counted and a
    # `>=` assertion passed on a doubled numerator).
    rollout_idx = {int(kind.split()[1]) for _, kind in events if kind.startswith("rollout ")}
    rollouts = sorted(ts for ts, kind in events if kind.startswith("rollout "))
    max_gap = max((b - a for a, b in zip(steps + rollouts, (steps + rollouts)[1:])), default=0.0)
    in_window = [ts for ts in steps + rollouts if so_active_ts <= ts <= si_done_ts]

    verdicts["scale_out_active"] = so_final["status"] == "ACTIVE"
    verdicts["scale_in_completed"] = si_final["status"] == "COMPLETED"
    verdicts["elastic_engine_served_rewards"] = bool(elastic) and sum(elastic.values()) > 0
    verdicts["train_events_during_scaling_window"] = len(in_window) > 0
    verdicts["no_train_stall_over_120s"] = max_gap <= 120.0
    verdicts["training_finished_after_scale_in"] = finished
    verdicts["final_capacity_is_initial"] = final["current"] == 1 and initial_ids <= final_ids
    verdicts["rollout_count"] = len(rollout_idx)
    verdicts["num_rollout_expected"] = args.num_rollout
    verdicts["all_rollouts_landed"] = rollout_idx == set(range(args.num_rollout))
    verdicts["E2E_PASS"] = all(v for k, v in verdicts.items() if isinstance(v, bool))
    log_event("verdicts", **{k: v for k, v in verdicts.items() if isinstance(v, (bool, int))})

    dump(args.out_dir, verdicts, functional_error=None)
    return 0 if verdicts["E2E_PASS"] else 1


def dump(out_dir: str, verdicts: dict, functional_error):
    with LOCK:
        events = list(EVENTS)
        train_events = [{"ts": ts, "kind": kind} for ts, kind in sorted(TRAIN_EVENTS)]
    with open(os.path.join(out_dir, "events.json"), "w") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "train_events.json"), "w") as f:
        json.dump(train_events, f, indent=2, ensure_ascii=False)
    verdicts = dict(verdicts)
    verdicts["functional_error"] = functional_error
    with open(os.path.join(out_dir, "verdicts.json"), "w") as f:
        json.dump(verdicts, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
