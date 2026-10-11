# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Task 4 autoscaler E2E: deterministic load curve drives automatic GenRM
scaling.

Deploys the real GenRM service plus one AutoscalerService (genrm service
target, demo-tuned thresholds), then runs LOW -> HIGH -> STEADY -> LOW'
scoring load and records the acceptance evidence:

  1. sustained saturation triggers automatic scale-out (1 -> 2);
  2. the elastic engine actually serves traffic (per-engine served counter);
  3. sustained low load triggers automatic scale-in (2 -> 1) after cooldown;
  4. the initial engine is never removed; no request fails at any transition.

Every second the timeline records replica count, per-engine served counters,
autoscaler decisions and scale history, producing a reproducible record.

Usage (repo root, python with ray/sglang/torch):

    python demos/task4_genrm/e2e_autoscaler_load.py \
        --model-path /path/to/Qwen3-0.6B --out-dir demos/task4_genrm/results/autoscaler_run
"""

import argparse
import json
import os
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
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)


PROMPTS = [
    "Question: What is 17 * 23? Candidate answer: 391. Is the candidate answer correct? Reply with exactly YES or NO.",
    "Question: Is 97 a prime number? Candidate answer: yes. Is the candidate answer correct? Reply with exactly YES or NO.",
    "Question: What is 15% of 200? Candidate answer: 30. Is the candidate answer correct? Reply with exactly YES or NO.",
]

# HIGH-phase prompt: ~1,800 tokens of context so a batch of concurrent
# requests actually fills the 0.6B engine's KV pool (~170k tokens) and drives
# token_usage over the scale-out threshold. Short 30-token prompts leave a
# 4090-sized 0.6B engine under 5% utilization no matter the concurrency.
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
    """Continuous scoring load whose concurrency follows a phase variable.

    HIGH uses long generations (max_new_tokens 768) so token usage and queue
    depth actually saturate a 0.6B engine; LOW uses short generations.
    """

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
                # Long context + long generation: fills the KV pool and keeps
                # decode running -- the two signals the scale-out thresholds
                # watch (token_usage / queue depth). Short prompts leave a
                # 4090-sized 0.6B engine under 5% utilization at any concurrency.
                # The worker id prefix defeats the radix prefix cache: identical
                # prompts would be KV-shared and leave the pool empty.
                prompt = f"[request {wid}]\n{HIGH_PROMPT}"
                max_new, timeout = 512, 900
            elif phase == "STEADY":
                prompt, max_new, timeout = PROMPTS[i % len(PROMPTS)], 256, 300
            else:
                prompt, max_new, timeout = PROMPTS[i % len(PROMPTS)], 16, 120
            t = time.time()
            try:
                code, body = http_post(
                    GENRM_BASE,
                    "/generate",
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "sampling_params": {"temperature": 0, "max_new_tokens": max_new},
                    },
                    timeout=timeout,
                )
                ok = code == 200
            except Exception:  # noqa: BLE001
                ok = False
            with self._lock:
                self._results.append(
                    {"t": round(t - self.ev.t0, 3), "phase": phase, "ok": ok, "latency": round(time.time() - t, 3)}
                )
            i += 1

    def start(self, workers=32):
        self._pool = ThreadPoolExecutor(max_workers=workers)
        for wid in range(workers):
            self._pool.submit(self._worker, wid)
        self.ev.log("load_started", workers=workers)

    def stop(self):
        self._stop.set()
        self._pool.shutdown(wait=True)
        with self._lock:
            results = list(self._results)
        failed = [r for r in results if not r["ok"]]
        self.ev.log("load_stopped", total=len(results), failed=len(failed), failures=failed[:10])
        return results


class Timeline:
    """1 Hz sampler of engines, autoscaler status and scale history."""

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
            row["ready"] = eng.get("ready")
            row["engines"] = [(e["host"], e["port"], e.get("served", 0)) for e in eng.get("engines", [])]
        except Exception as exc:  # noqa: BLE001
            row["engines_err"] = str(exc)[:80]
        try:
            st = http_get(AUTOSCALER_BASE, "/status", timeout=10)
            svc = (st.get("services") or {}).get("genrm") or st
            row["autoscaler_engines"] = svc.get("current_engines")
            dec = svc.get("last_decision")
            if dec:
                row["decision"] = dec.get("action")
            row["pending"] = len(svc.get("pending_requests") or [])
        except Exception as exc:  # noqa: BLE001
            row["status_err"] = str(exc)[:80]
        try:
            # All autoscaler state lives under the "genrm" service runtime;
            # the endpoint defaults to the (unused) "rollout" service.
            hist = http_get(AUTOSCALER_BASE, "/scale_history?limit=50&service=genrm", timeout=10)
            row["history"] = [
                (h.get("action"), h.get("status"), round(h.get("triggered_at", 0), 1))
                for h in hist.get("history", [])[:8]
            ]
        except Exception as exc:  # noqa: BLE001
            row["history_err"] = str(exc)[:80]
        self.rows.append(row)
        return row

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
        self._thread.join(timeout=5)

    def engine_counts(self):
        return [(r["t"], r.get("current")) for r in self.rows if "current" in r]


def wait_for(condition, timeout_s, desc, poll=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if condition():
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
        os.path.dirname(os.path.abspath(__file__)), "results", f"autoscaler_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)
    events = []

    class Ev:
        t0 = time.time()

        @staticmethod
        def log(event, **fields):
            entry = {"t": round(time.time() - Ev.t0, 3), "event": event, **fields}
            events.append(entry)
            print(f"[{entry['t']:8.3f}s] {event} {json.dumps(fields, ensure_ascii=False)[:200]}", flush=True)

    ev = Ev()

    import ray
    from ray import serve

    from relax.components.genrm import GenRM
    from relax.core.service import create_placement_group
    from relax.utils.autoscaler.autoscaler_service import AutoscalerService
    from relax.utils.autoscaler.config import AutoscalerConfig
    from relax.utils.utils import get_serve_url

    # Cleanup contract: the outer finally covers every failure exit
    # (deployment prep, load, sampling, assertions, file writes); each
    # cleanup step handles its own exception; evidence dumping is never a
    # precondition for cleanup; and only resources this run created are
    # released. Functional and cleanup outcomes are reported separately.
    timeline = None
    timeline_stopped = False
    load = None
    load_stopped = False
    results = None
    verdicts: dict = {}
    functional_error = None
    pg = None
    genrm_app = False
    autoscaler_app = False
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
        serve.run(GenRM.bind(None, pg, cfg.genrm_num_gpus, cfg, "genrm"), name="genrm", route_prefix="/genrm")
        genrm_app = True
        global GENRM_BASE, AUTOSCALER_BASE
        GENRM_BASE = get_serve_url("/genrm")
        # Single-node E2E: the Serve HTTP proxy binds to localhost inside this
        # container (container IP refused, 127.0.0.1 reachable); rewrite host.
        from urllib.parse import urlsplit, urlunsplit

        def _loopback(url: str) -> str:
            u = urlsplit(url)
            return urlunsplit((u.scheme, f"127.0.0.1:{u.port}", u.path, "", ""))

        GENRM_BASE = _loopback(GENRM_BASE)

        # Demo-tuned autoscaler config: fast scale-out on sustained saturation,
        # conservative scale-in on sustained low load, cooldowns against flapping.
        import yaml as _yaml

        autoscaler_yaml = {
            "enabled": True,
            "min_engines": 1,
            "max_engines": 2,
            "metrics_interval_secs": 3.0,
            "evaluation_interval_secs": 5.0,
            "condition_window_secs": 30.0,
            "scale_out_cooldown_secs": 20.0,
            "scale_in_cooldown_secs": 45.0,
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
                        # 48 concurrent short greedy requests complete in bursts,
                        # so a 0.6B engine's token throughput legitimately swings
                        # (measured variance 0.77 under LOW load); the default 0.1
                        # would never consider it stable. The low-load signal is
                        # carried by token_usage/queue; this guard only rules out
                        # scaling in while throughput is still ramping.
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

        # serve.run returns a DeploymentHandle to the app's ingress deployment;
        # serve.get_deployment_handle() cannot be used from a driver outside a
        # Serve application, and ray.get() does not accept DeploymentResponse.
        autoscaler_handle = serve.run(
            AutoscalerService.bind(None, None, autoscaler_cfg, "autoscaler_genrm"),
            name="autoscaler_genrm",
            route_prefix="/autoscaler_genrm",
        )
        autoscaler_app = True
        AUTOSCALER_BASE = _loopback(get_serve_url("/autoscaler_genrm"))
        ev.log("deployed", genrm=GENRM_BASE, autoscaler=AUTOSCALER_BASE)

        # Wait for the initial engine, then start autoscaler + load + timeline.
        def one_engine():
            try:
                return http_get(GENRM_BASE, "/engines", timeout=10).get("current") == cfg.genrm_num_gpus
            except Exception:  # noqa: BLE001
                return False

        if not wait_for(one_engine, 900, "initial engine"):
            return 2
        ev.log("initial_engine_up")

        import requests  # noqa: F401  (ensure import errors surface here, not in threads)

        autoscaler_handle.start.remote().result()
        ev.log("autoscaler_started")

        # Demo-tune the global cooldowns for the load experiment (per-service
        # cooldown overrides are not part of ServiceScalingPolicy yet; the PATCH
        # endpoint is the supported way to set them at runtime).
        import requests as _rq

        _p = _rq.patch(
            f"{AUTOSCALER_BASE}/config",
            json={"scale_in_cooldown_secs": 60.0, "scale_out_cooldown_secs": 20.0},
            timeout=10,
        )
        _p.raise_for_status()
        ev.log("autoscaler_cooldowns_patched", scale_in=60.0, scale_out=20.0)

        timeline = Timeline(ev)
        load = PhaseLoadGenerator(ev)
        timeline.start()
        load.start(workers=48)

        initial_ids = None
        try:
            # LOW: one engine must be sufficient; no scale-out allowed.
            load.phase = "LOW"
            time.sleep(20)
            low_snap = http_get(GENRM_BASE, "/engines")
            initial_ids = {(e["host"], e["port"]) for e in low_snap["engines"]}
            ev.log("phase_low_done", current=low_snap["current"], ids=sorted(map(list, initial_ids)))

            # HIGH: sustained saturation must trigger automatic scale-out.
            load.phase = "HIGH"
            scaled_out = wait_for(
                lambda: http_get(GENRM_BASE, "/engines", timeout=10).get("current") == cfg.genrm_num_gpus + 1,
                300,
                "automatic scale-out",
            )
            high_wait = round(time.time() - ev.t0, 1)
            ev.log("phase_high_result", scaled_out=scaled_out, t=high_wait)

            # STEADY: keep traffic on both engines; the elastic one must serve.
            load.phase = "STEADY"
            t_steady = time.time()
            elastic_served = 0
            while time.time() - t_steady < 60:
                snap = http_get(GENRM_BASE, "/engines", timeout=10)
                for e in snap["engines"]:
                    if (e["host"], e["port"]) not in initial_ids:
                        elastic_served = max(elastic_served, e.get("served", 0))
                time.sleep(2)
            ev.log("phase_steady_done", elastic_served=elastic_served)

            # LOW': sustained low load must trigger scale-in back to initial.
            load.phase = "LOW"
            scaled_in = wait_for(
                lambda: http_get(GENRM_BASE, "/engines", timeout=10).get("current") == cfg.genrm_num_gpus,
                500,
                "automatic scale-in",
            )
            ev.log("phase_low_prime_result", scaled_in=scaled_in)
            time.sleep(10)
        finally:
            # Load/timeline teardown and evidence writes: each step
            # independent; a failure here must not block the remaining
            # steps or the outer cleanup.
            try:
                if load is not None and not load_stopped and load._pool is not None:
                    results = load.stop()
                    load_stopped = True
            except Exception as exc:  # noqa: BLE001
                ev.log("load_stop_failed", error=str(exc))
            try:
                if timeline is not None and not timeline_stopped:
                    timeline.stop()
                    timeline_stopped = True
            except Exception as exc:  # noqa: BLE001
                ev.log("timeline_stop_failed", error=str(exc))
            for fname, payload in (
                ("timeline.json", timeline.rows if timeline is not None else []),
                ("events.json", events),
                ("load_requests.json", results if results is not None else []),
            ):
                try:
                    with open(os.path.join(out_dir, fname), "w") as f:
                        json.dump(payload, f, indent=2)
                except Exception as exc:  # noqa: BLE001
                    ev.log("evidence_write_failed", file=fname, error=str(exc))

        final_snap = http_get(GENRM_BASE, "/engines")
        final_ids = {(e["host"], e["port"]) for e in final_snap["engines"]}
        hist = http_get(AUTOSCALER_BASE, "/scale_history?limit=50&service=genrm").get("history", [])
        conditions = http_get(AUTOSCALER_BASE, "/conditions?service=genrm")

        verdicts = {
            "scale_out_triggered": any(h.get("action") == "scale_out" and h.get("status") == "ACTIVE" for h in hist),
            "elastic_engine_served_traffic": elastic_served > 0,
            "scale_in_triggered": any(h.get("action") == "scale_in" and h.get("status") == "COMPLETED" for h in hist),
            "final_count_is_initial": final_snap.get("current") == cfg.genrm_num_gpus,
            "initial_engine_survived": bool(initial_ids) and initial_ids <= final_ids,
            "zero_load_failures": all(r["ok"] for r in results),
            "load_total_requests": len(results),
            "scale_history_len": len(hist),
            "conditions_recorded": "conditions" in conditions,
            "timeline_rows": len(timeline.rows),
        }
        verdicts["E2E_PASS"] = all(v for k, v in verdicts.items() if isinstance(v, bool))
        ev.log("verdicts", **verdicts)
    except Exception as exc:  # noqa: BLE001
        functional_error = exc
        ev.log("functional_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        # Every step handles its own exception: a failure here never masks
        # the original error and never blocks the remaining steps.
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
                from ray.util.placement_group import placement_group_table, remove_placement_group

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

    # Final record: functional and cleanup outcomes are separate; total PASS
    # requires both. The original exception, if any, is re-raised after the
    # verdict file is written so it is never masked.
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
