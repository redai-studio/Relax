# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Task 4 preregistered autoscaler rounds (frozen thresholds, 2026-09-25).

Round A: a with-traffic LOW->HIGH->STEADY->LOW' cycle whose assertions
were fixed before the run (autoscaler_preregistration_20260925.md):
decisions are attributed to load phases via /scale_history
``triggered_at`` (unix) against phase boundaries recorded on the same
clock -- never via poll-discovery times.

Round B: a true-idle scale-in -- manual scale-out to 2 with zero
traffic, then the autoscaler must decide scale-in on valid idle evidence
(queue/running observed 0) within the preregistered window.

Both rounds assert full resource return: Ray free GPUs, per-GPU memory
vs the pre-scale-out snapshot, and the placement-group table count. TUI
screenshots (``--service genrm`` and default) are captured live during
STEADY via the monitor's headless ``--screenshot`` mode.

Usage (repo root, GPUs free):

    python demos/task4_genrm/e2e_autoscaler_preregistered.py
        --model-path /path/to/Qwen3-0.6B
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

GENRM_BASE = None
AUTOSCALER_BASE = None


def http_get(base, path, timeout=30):
    import requests

    r = requests.get(f"{base}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def http_post(base, path, body, timeout=300):
    import requests

    r = requests.post(f"{base}{path}", json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"{path} -> {r.status_code}: {r.text[:200]}")
    return r.json()


PROMPTS = [
    "Question: What is 17 * 23? Candidate answer: 391. Is the candidate answer correct? Reply with exactly YES or NO.",
    "Question: Is 97 a prime number? Candidate answer: yes. Is the candidate answer correct? Reply with exactly YES or NO.",
    "Question: What is 15% of 200? Candidate answer: 30. Is the candidate answer correct? Reply with exactly YES or NO.",
]

FILLER = (
    "You are a meticulous mathematics reviewer. Before answering, reproduce "
    "the full derivation, check each algebraic step, and list every assumption "
    "you make. Take your time and be exhaustive. "
) * 60
HIGH_PROMPT = (
    FILLER + "Now the question: What is 17 * 23? Candidate answer: 391. "
    "Is the candidate answer correct? Reply with exactly YES or NO."
)


class PhaseLoadGenerator:
    def __init__(self, ev):
        self.ev = ev
        self.phase = "LOW"
        self._stop = threading.Event()
        self._results = []
        self._lock = threading.Lock()
        self._pool = None

    def _worker(self, wid):
        i = 0
        while not self._stop.is_set():
            phase = self.phase
            if phase == "HIGH":
                prompt = f"[request {wid}]\n{HIGH_PROMPT}"
                max_new, timeout = 512, 900
            elif phase == "STEADY":
                prompt, max_new, timeout = PROMPTS[i % len(PROMPTS)], 256, 300
            else:
                prompt, max_new, timeout = PROMPTS[i % len(PROMPTS)], 16, 120
            t = time.time()
            try:
                http_post(
                    GENRM_BASE,
                    "/generate",
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "sampling_params": {"temperature": 0, "max_new_tokens": max_new},
                    },
                    timeout=timeout,
                )
                ok = True
            except Exception:  # noqa: BLE001
                ok = False
            with self._lock:
                self._results.append({"t": round(t - self.ev.t0, 3), "phase": phase, "ok": ok})
            i += 1

    def start(self, workers=48):
        self._pool = ThreadPoolExecutor(max_workers=workers)
        for wid in range(workers):
            self._pool.submit(self._worker, wid)

    def stop(self):
        self._stop.set()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        with self._lock:
            return list(self._results)


class Timeline:
    def __init__(self, ev):
        self.ev = ev
        self._stop = threading.Event()
        self.rows = []
        self._thread = None

    def _sample(self):
        row = {"t": round(time.time() - self.ev.t0, 3)}
        try:
            eng = http_get(GENRM_BASE, "/engines", timeout=10)
            row["current"] = eng.get("current")
            row["engines"] = [(e["host"], e["port"], e.get("served", 0)) for e in eng.get("engines", [])]
        except Exception as exc:  # noqa: BLE001
            row["engines_err"] = str(exc)[:80]
        self.rows.append(row)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def gpu_memory_vector():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    return [int(line.split(",")[1]) for line in out.splitlines()]


def pg_count():
    """Count placement groups in a non-terminal state.

    Ray keeps tombstone entries for removed-and-confirmed ``REMOVED`` placement
    groups in ``placement_group_table()`` forever, so the raw table length
    grows by one per completed scale-in and can never return to its pre-run
    baseline (settled in the r2 post-mortem: commit 46ff87a; reproduced CPU-
    only). Resource return is asserted by counting only placement groups whose
    state is not ``REMOVED``.
    """
    import ray

    table = ray.util.placement_group_table()
    return sum(1 for info in table.values() if info.get("state") != "REMOVED")


def wait_for(cond, timeout_s, desc, poll=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if cond():
            return True
        time.sleep(poll)
    print(f"    TIMEOUT waiting for {desc} after {timeout_s}s", flush=True)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--genrm-num-gpus", type=int, default=1)
    args_cli = parser.parse_args()

    out_dir = args_cli.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"autoscaler_preregistered_{time.strftime('%Y%m%d_%H%M%S')}",
    )

    class Ev:
        t0 = time.time()

        @staticmethod
        def log(event, **fields):
            entry = {"t": round(time.time() - Ev.t0, 3), "event": event, **fields}
            print(f"[{entry['t']:8.3f}s] {event} {json.dumps(fields, ensure_ascii=False)[:200]}", flush=True)
            events.append(entry)

    events: list = []
    os.makedirs(out_dir, exist_ok=True)
    ev = Ev()
    ev.log("e2e_start", model_path=args_cli.model_path, out_dir=out_dir)

    import ray
    from ray import serve
    from ray.util.placement_group import placement_group_table, remove_placement_group

    from relax.components.genrm import GenRM
    from relax.core.service import create_placement_group
    from relax.utils.autoscaler.autoscaler_service import AutoscalerService
    from relax.utils.autoscaler.config import AutoscalerConfig
    from relax.utils.utils import get_serve_url

    verdicts: dict = {}
    functional_error = None
    pg = None
    genrm_app = False
    autoscaler_app = False
    load = None
    timeline = None
    load_stopped = False
    timeline_stopped = False
    cleanup_pass = True
    cleanup_errors: list = []
    leftover_free_gpus = None

    try:
        ray.init(ignore_reinit_error=True)
        ev.log("ray_init", gpus=ray.cluster_resources().get("GPU", 0))

        cfg = Namespace(
            genrm_model_path=os.path.abspath(args_cli.model_path),
            genrm_num_gpus=args_cli.genrm_num_gpus,
            genrm_num_gpus_per_engine=1,
            genrm_engine_config={"mem_fraction_static": 0.85, "enable_metrics": True},
            genrm_sampling_config={},
            num_gpus_per_node=4,
            rollout_num_gpus=0,
            sglang_dp_size=1,
            seed=42,
            fully_async=True,
            rollout_external=False,
            rollout_num_gpus_per_engine=1,
            use_slime_router=False,
            offload_rollout=False,
            debug_train_only=False,
            fp16=False,
            use_rollout_routing_replay=False,
        )
        cfg._genrm_instances_resolved = {
            "__default__": {
                "model_path": cfg.genrm_model_path,
                "num_gpus": cfg.genrm_num_gpus,
                "num_gpus_per_engine": 1,
                "engine_config": cfg.genrm_engine_config,
                "sampling_config": cfg.genrm_sampling_config,
            }
        }

        pg = create_placement_group(num_gpus=cfg.genrm_num_gpus, node_group_affinity=False)
        ev.log("pg_created", bundles=len(pg[1]), gpu_ids=pg[2])
        serve.run(GenRM.bind(None, pg, cfg.genrm_num_gpus, cfg, "genrm"), name="genrm", route_prefix="/genrm")
        genrm_app = True
        global GENRM_BASE, AUTOSCALER_BASE
        GENRM_BASE = get_serve_url("/genrm")

        from urllib.parse import urlsplit, urlunsplit

        def _loopback(url):
            u = urlsplit(url)
            return urlunsplit((u.scheme, f"127.0.0.1:{u.port}", u.path, "", ""))

        GENRM_BASE = _loopback(GENRM_BASE)

        # ---- frozen configuration (autoscaler_preregistration_20260925.md)
        import yaml as _yaml

        autoscaler_yaml = {
            "enabled": True,
            "min_engines": 1,
            "max_engines": 2,
            "metrics_interval_secs": 3.0,
            "evaluation_interval_secs": 5.0,
            "condition_window_secs": 30.0,
            "scale_out_cooldown_secs": 20.0,
            "scale_in_cooldown_secs": 60.0,
            "min_coverage_scale_out": 0.5,
            "min_coverage_scale_in": 1.0,
            "service_targets": {"genrm": GENRM_BASE},
            "service_policies": {
                "genrm": {
                    "min_engines": cfg.genrm_num_gpus,
                    "max_engines": cfg.genrm_num_gpus + 1,
                    "scale_out_policy": {
                        "token_usage_threshold": 0.3,
                        "queue_depth_per_engine": 4,
                        "queue_time_p95_threshold": 3.0,
                        "ttft_p95_threshold": 3.0,
                        "condition_duration_secs": 15.0,
                        "max_delta": 1,
                    },
                    "scale_in_policy": {
                        "token_usage_threshold": 0.05,
                        "queue_depth_threshold": 0,
                        "throughput_variance_threshold": 1.0,
                        "condition_duration_secs": 45.0,
                        "max_delta": 1,
                        "projected_usage_max": 0.9,
                    },
                }
            },
        }
        autoscaler_cfg_path = os.path.join(out_dir, "autoscaler.yaml")
        with open(autoscaler_cfg_path, "w") as f:
            _yaml.safe_dump(autoscaler_yaml, f)
        autoscaler_cfg = AutoscalerConfig.from_yaml(autoscaler_cfg_path)
        autoscaler_handle = serve.run(
            AutoscalerService.bind(None, None, autoscaler_cfg, "autoscaler_genrm"),
            name="autoscaler_genrm",
            route_prefix="/autoscaler_genrm",
        )
        autoscaler_app = True
        AUTOSCALER_BASE = _loopback(get_serve_url("/autoscaler_genrm"))
        ev.log("deployed", genrm=GENRM_BASE, autoscaler=AUTOSCALER_BASE)

        def _initial_engines():
            # The serve replica may 404 (route not yet registered) or block
            # past the HTTP timeout while the manager boots the engine —
            # both are boot transients, not functional failures.
            try:
                return http_get(GENRM_BASE, "/engines", timeout=10).get("current") == cfg.genrm_num_gpus
            except Exception:  # noqa: BLE001
                return False

        if not wait_for(_initial_engines, 900, "initial engine"):
            raise RuntimeError("initial engine never came up")

        import requests as _rq

        _p = _rq.patch(
            f"{AUTOSCALER_BASE}/config",
            json={"scale_in_cooldown_secs": 60.0, "scale_out_cooldown_secs": 20.0},
            timeout=10,
        )
        _p.raise_for_status()
        # AutoscalerService._main_loop only runs after an explicit start()
        # (Ray Serve has no auto-start hook); without this call no scaling
        # decision is ever made. Found in review before the first run.
        autoscaler_handle.start.remote().result()
        ev.log("autoscaler_started", started=True)

        timeline = Timeline(ev)
        load = PhaseLoadGenerator(ev)
        timeline.start()
        load.start(workers=48)

        # Resource baselines before any scaling.
        baseline_free_gpus = ray.available_resources().get("GPU", 0)
        baseline_mem = gpu_memory_vector()
        baseline_pgs = pg_count()
        ev.log("resource_baseline", free_gpus=baseline_free_gpus, mem_mib=baseline_mem, pgs=baseline_pgs)

        # ------------------------------------------------------------------ #
        # Round A: with-traffic cycle
        # ------------------------------------------------------------------ #
        phases = {}

        load.phase = "LOW"
        phases["low_start"] = time.time()
        time.sleep(20)
        phases["low_end"] = time.time()
        low_snap = http_get(GENRM_BASE, "/engines")
        initial_ids = {(e["host"], e["port"]) for e in low_snap["engines"]}
        ev.log("phase_low_done", current=low_snap["current"])

        load.phase = "HIGH"
        phases["high_start"] = time.time()
        scaled_out = wait_for(
            lambda: http_get(GENRM_BASE, "/engines", timeout=10).get("current") == cfg.genrm_num_gpus + 1,
            300,
            "automatic scale-out",
        )
        phases["high_end"] = time.time()
        ev.log("phase_high_done", scaled_out=scaled_out, t=round(time.time() - ev.t0, 1))

        load.phase = "STEADY"
        phases["steady_start"] = time.time()
        # A3 sampling starts immediately: under the frozen thresholds the
        # elastic engine is removed ~55-60 s into STEADY, so sampling must
        # not sit behind the (potentially slow) screenshot subprocesses.
        elastic_served = 0
        t_steady = time.time()
        while time.time() - t_steady < 60:
            snap = http_get(GENRM_BASE, "/engines", timeout=10)
            for e in snap["engines"]:
                if (e["host"], e["port"]) not in initial_ids:
                    elastic_served = max(elastic_served, e.get("served", 0))
            time.sleep(2)
        # TUI screenshots (A8), still during STEADY load.
        screenshots_ok = True
        for svc, name in (("genrm", "tui-genrm.svg"), ("rollout", "tui-rollout.svg")):
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "relax.utils.autoscaler.monitor",
                        "--url",
                        AUTOSCALER_BASE,
                        "--service",
                        svc,
                        "--screenshot",
                        os.path.join(out_dir, name),
                    ],
                    cwd=REPO_ROOT,
                    timeout=180,
                    check=True,
                )
                ev.log("tui_screenshot_captured", service=svc, file=name)
            except Exception as exc:  # noqa: BLE001
                screenshots_ok = False
                ev.log("tui_screenshot_failed", service=svc, error=str(exc)[:160])
        phases["steady_end"] = time.time()
        ev.log("phase_steady_done", elastic_served=elastic_served, screenshots_ok=screenshots_ok)

        load.phase = "LOW"
        phases["low_prime_start"] = time.time()
        scaled_in = wait_for(
            lambda: http_get(GENRM_BASE, "/engines", timeout=10).get("current") == cfg.genrm_num_gpus,
            500,
            "automatic scale-in",
        )
        phases["run_end"] = time.time()
        ev.log("phase_low_prime_done", scaled_in=scaled_in)
        time.sleep(10)

        results = load.stop()
        load_stopped = True
        timeline.stop()
        timeline_stopped = True

        hist = http_get(AUTOSCALER_BASE, "/scale_history?limit=50&service=genrm").get("history", [])
        so = next((h for h in hist if h.get("action") == "scale_out"), None)
        # The scale_in history record only lands after the autoscaler's next
        # evaluation observes the terminal state (and the manager only marks
        # COMPLETED after the PG release wait) — poll for it instead of
        # racing the record's appearance.
        si = None
        si_deadline = time.time() + 90
        while time.time() < si_deadline:
            hist = http_get(AUTOSCALER_BASE, "/scale_history?limit=50&service=genrm").get("history", [])
            si = next((h for h in hist if h.get("action") == "scale_in"), None)
            if si is not None and si.get("status") in ("COMPLETED", "FAILED"):
                break
            time.sleep(3)
        ev.log("round_a_history", so=bool(so), si=bool(si), si_status=(si or {}).get("status"))

        with open(os.path.join(out_dir, "scale_history.json"), "w") as f:
            json.dump(hist, f, indent=2)
        with open(os.path.join(out_dir, "timeline.json"), "w") as f:
            json.dump(timeline.rows, f, indent=2)
        with open(os.path.join(out_dir, "load_requests.json"), "w") as f:
            json.dump(results, f, indent=2)
        with open(os.path.join(out_dir, "phase_boundaries.json"), "w") as f:
            json.dump(
                {
                    "phases": phases,
                    "history": [
                        {
                            "action": h.get("action"),
                            "triggered_at": h.get("triggered_at"),
                            "completed_at": h.get("completed_at"),
                        }
                        for h in hist
                    ],
                },
                f,
                indent=2,
            )

        so_dec = so["triggered_at"] if so else None
        si_dec = si["triggered_at"] if si else None
        final_snap = http_get(GENRM_BASE, "/engines")
        final_ids = {(e["host"], e["port"]) for e in final_snap["engines"]}

        verdicts.update(
            {
                "A1_no_scale_out_in_low": so_dec is None or so_dec >= phases["low_end"],
                "A2_scale_out_decided_in_high": so_dec is not None
                and phases["high_start"] <= so_dec <= phases["high_end"],
                "A3_elastic_served_in_steady": elastic_served > 0,
                "A4_scale_in_decided_outside_high": si_dec is not None
                and phases["steady_start"] <= si_dec <= phases["run_end"],
                "A5_scale_in_completed_final_is_initial": (si or {}).get("status") == "COMPLETED"
                and final_snap.get("current") == cfg.genrm_num_gpus
                and initial_ids <= final_ids,
                "A6_zero_load_failures": all(r["ok"] for r in results),
                "A6_load_total": len(results),
                "A8_tui_screenshots": screenshots_ok
                and all(
                    os.path.exists(os.path.join(out_dir, n)) and os.path.getsize(os.path.join(out_dir, n)) > 0
                    for n in ("tui-genrm.svg", "tui-rollout.svg")
                ),
            }
        )

        def _mem_stable(samples: int = 3, gap: float = 2.0):
            # Engine teardown memory release can lag a few seconds; take the
            # max across samples so residual usage is not missed by a lucky
            # single read.
            vecs = []
            for _ in range(samples):
                vecs.append(gpu_memory_vector())
                time.sleep(gap)
            return [max(v[i] for v in vecs) for i in range(len(vecs[0]))]

        after_mem = _mem_stable()
        after_pgs = pg_count()
        after_free = ray.available_resources().get("GPU", 0)
        mem_ok = all(abs(a - b) <= 500 for a, b in zip(after_mem, baseline_mem))
        verdicts["A7_resources_returned"] = after_free >= baseline_free_gpus and mem_ok and after_pgs == baseline_pgs
        ev.log(
            "round_a_resources",
            free=f"{after_free}/{baseline_free_gpus}",
            pgs=f"{after_pgs}/{baseline_pgs}",
            mem_ok=mem_ok,
        )

        # ------------------------------------------------------------------ #
        # Round B: true-idle scale-in
        # ------------------------------------------------------------------ #
        out_b = http_post(GENRM_BASE, "/scale_out", {"num_replicas": cfg.genrm_num_gpus + 1, "timeout_secs": 900.0})
        rid_b = out_b["request_id"]

        def _b_active():
            st = http_get(GENRM_BASE, f"/scale_out/{rid_b}", timeout=10)
            if st["status"] in ("FAILED", "PARTIAL"):
                raise RuntimeError(f"round B manual scale-out reached {st['status']}")
            return st["status"] == "ACTIVE"

        b_out = wait_for(_b_active, 900, "round B manual scale-out")
        ev.log("round_b_scaled_out", active=b_out)
        verdicts["B1_manual_scale_out_active"] = b_out
        # Round B's scale-in must be a NEW history record: filter by
        # triggered_at after this moment (Round A's own scale_in is already
        # in history and would otherwise match any later-than-scale_out cut).
        b_active_t = time.time()

        b_deadline = time.time() + 300
        si_b = None
        while time.time() < b_deadline:
            hist_b = http_get(AUTOSCALER_BASE, "/scale_history?limit=10&service=genrm").get("history", [])
            cand = [h for h in hist_b if h.get("action") == "scale_in" and h.get("triggered_at", 0) > b_active_t]
            if cand:
                si_b = cand[0]
                if si_b.get("status") in ("COMPLETED", "FAILED"):
                    break
            time.sleep(3)
        snap_b = (si_b or {}).get("metrics_snapshot") or {}
        verdicts["B2_idle_scale_in_within_window"] = bool(si_b) and si_b.get("status") == "COMPLETED"
        verdicts["B2_idle_evidence_zero_running"] = (
            bool(si_b) and snap_b.get("total_running_reqs") == 0 and snap_b.get("total_queue_reqs") == 0
        )
        verdicts["B2_idle_conditions_all_three"] = bool(si_b) and set(si_b.get("triggered_conditions") or []) >= {
            "token_usage_low",
            "no_queue",
            "throughput_stable",
        }

        final_b = http_get(GENRM_BASE, "/engines")
        verdicts["B3_final_one_initial_survived"] = final_b.get("current") == cfg.genrm_num_gpus and initial_ids <= {
            (e["host"], e["port"]) for e in final_b["engines"]
        }
        mem_b = _mem_stable()
        verdicts["B3_resources_returned"] = (
            ray.available_resources().get("GPU", 0) >= baseline_free_gpus
            and all(abs(a - b) <= 500 for a, b in zip(mem_b, baseline_mem))
            and pg_count() == baseline_pgs
        )
        ev.log("round_b_done", scale_in=bool(si_b), final=final_b.get("current"))

        with open(os.path.join(out_dir, "scale_history_round_b.json"), "w") as f:
            json.dump(
                http_get(AUTOSCALER_BASE, "/scale_history?limit=50&service=genrm").get("history", []), f, indent=2
            )

        verdicts["E2E_PASS"] = all(v for k, v in verdicts.items() if isinstance(v, bool))
        ev.log("verdicts", **{k: v for k, v in verdicts.items() if not isinstance(v, (dict, list))})
    except Exception as exc:  # noqa: BLE001
        functional_error = exc
        ev.log("functional_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        # Independent steps; evidence writes never gate cleanup.
        try:
            if load is not None and not load_stopped and load._pool is not None:
                _res = load.stop()
                load_stopped = True
                with open(os.path.join(out_dir, "load_requests.json"), "w") as f:
                    json.dump(_res, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            ev.log("load_stop_failed", error=str(exc))
        try:
            if timeline is not None and not timeline_stopped:
                timeline.stop()
                timeline_stopped = True
        except Exception as exc:  # noqa: BLE001
            ev.log("timeline_stop_failed", error=str(exc))
        for fname, payload in (("events.json", events),):
            try:
                with open(os.path.join(out_dir, fname), "w") as f:
                    json.dump(payload, f, indent=2)
            except Exception as exc:  # noqa: BLE001
                ev.log("evidence_write_failed", file=fname, error=str(exc))
        try:
            if autoscaler_app:
                serve.delete("autoscaler_genrm")
        except Exception as exc:  # noqa: BLE001
            cleanup_pass = False
            cleanup_errors.append(f"serve_delete_autoscaler: {exc}")
        try:
            if genrm_app:
                serve.delete("genrm")
                time.sleep(5)
        except Exception as exc:  # noqa: BLE001
            cleanup_pass = False
            cleanup_errors.append(f"serve_delete_genrm: {exc}")
        try:
            if pg is not None:
                remove_placement_group(pg[0])
                deadline = time.time() + 60
                while time.time() < deadline:
                    if placement_group_table(pg[0]).get("state") == "REMOVED":
                        break
                    time.sleep(1.0)
                else:
                    cleanup_pass = False
                    cleanup_errors.append("pg_remove: driver PG not REMOVED within 60s")
        except Exception as exc:  # noqa: BLE001
            cleanup_pass = False
            cleanup_errors.append(f"pg_remove: {exc}")
        try:
            leftover_free_gpus = ray.available_resources().get("GPU", 0)
        except Exception:  # noqa: BLE001
            leftover_free_gpus = None
        try:
            ray.shutdown()
        except Exception as exc:  # noqa: BLE001
            cleanup_pass = False
            cleanup_errors.append(f"ray_shutdown: {exc}")
        ev.log(
            "cleanup_result",
            cleanup_pass=cleanup_pass,
            cleanup_errors=cleanup_errors,
            leftover_ray_free_gpus=leftover_free_gpus,
        )
        ev.log(
            "e2e_done",
            functional_pass=functional_error is None and bool(verdicts.get("E2E_PASS")),
            cleanup_pass=cleanup_pass,
        )

    functional_pass = functional_error is None and bool(verdicts.get("E2E_PASS"))
    verdicts.update(
        {
            "functional_pass": functional_pass,
            "cleanup_pass": cleanup_pass,
            "cleanup_errors": cleanup_errors,
            "leftover_ray_free_gpus": leftover_free_gpus,
            "functional_error": (
                None if functional_error is None else f"{type(functional_error).__name__}: {functional_error}"
            ),
        }
    )
    verdicts["PASS"] = functional_pass and cleanup_pass
    try:
        with open(os.path.join(out_dir, "verdicts.json"), "w") as f:
            json.dump(verdicts, f, indent=2, ensure_ascii=False)
    finally:
        if functional_error is not None:
            raise functional_error
    return 0 if verdicts["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
