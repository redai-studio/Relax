# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Task 4 end-to-end evidence: real GenRM 1->2->1 elastic scaling.

Runs the production code path (Ray Serve deployment + GenRMManager +
GenRMEngine/SGLang) on a live Ray cluster and records the acceptance
evidence defined in the RFC:

  1. scale-out 1->2 reaches ACTIVE with the new engine published only
     after its health check, on its own placement group;
  2. scoring traffic never stops failing-free across both operations
     (drain proof: no request error, no dropped request);
  3. scale-in 2->1 reaches COMPLETED with the victim drained, retired,
     and its GPU/placement group released;
  4. fixed-input, temperature=0 scores are identical before scale-out,
     after scale-out, and after scale-in.

Usage (from the repo root, with a python that has ray/sglang/torch):

    python demos/task4_genrm/e2e_genrm_scale.py \
        --model-path /path/to/Qwen2.5-0.5B-Instruct \
        --out-dir demos/task4_genrm/results/e2e_run

The script is read-only with respect to the repo except for --out-dir.
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

GENRM_BASE = None  # set after serve.run, via get_serve_url


# ---------------------------------------------------------------------------
# Evidence recorder
# ---------------------------------------------------------------------------
class Evidence:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.t0 = time.time()
        self.events: list = []
        self._lock = threading.Lock()

    def log(self, event: str, **fields) -> None:
        entry = {"t": round(time.time() - self.t0, 3), "event": event, **fields}
        with self._lock:
            self.events.append(entry)
        print(f"[{entry['t']:8.3f}s] {event} {json.dumps(fields, ensure_ascii=False)}", flush=True)

    def snapshot_gpu(self, tag: str) -> dict:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        import ray

        avail = {k: v for k, v in ray.available_resources().items() if k.startswith("GPU")}
        snap = {"tag": tag, "compute_apps": apps, "gpu_mem": gpus, "ray_free_gpus": avail}
        self.log("gpu_snapshot", **snap)
        return snap

    def dump(self) -> None:
        with open(os.path.join(self.out_dir, "events.json"), "w") as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# HTTP helpers (plain requests; the Serve proxy is on the head node)
# ---------------------------------------------------------------------------
import requests  # noqa: E402


def http_get(path: str, timeout: float = 60):
    r = requests.get(f"{GENRM_BASE}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def http_post(path: str, body: dict, timeout: float = 120):
    r = requests.post(f"{GENRM_BASE}{path}", json=body, timeout=timeout)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)


# ---------------------------------------------------------------------------
# Fixed scoring inputs (temperature=0 -> deterministic; identical text across
# phases proves the added/removed replicas serve the same model & weights)
# ---------------------------------------------------------------------------
JUDGE_PROMPTS = [
    (
        "math-1",
        "Question: What is 17 * 23? Candidate answer: 391. Is the candidate answer correct? Reply with exactly YES or NO.",
    ),
    (
        "math-2",
        "Question: What is the derivative of x^3? Candidate answer: 3x^2. Is the candidate answer correct? Reply with exactly YES or NO.",
    ),
    (
        "math-3",
        "Question: Is 97 a prime number? Candidate answer: yes. Is the candidate answer correct? Reply with exactly YES or NO.",
    ),
    (
        "math-4",
        "Question: What is the capital of France? Candidate answer: Paris. Is the candidate answer correct? Reply with exactly YES or NO.",
    ),
    (
        "math-5",
        "Question: What is 2 + 2? Candidate answer: 5. Is the candidate answer correct? Reply with exactly YES or NO.",
    ),
    (
        "math-6",
        "Question: How many sides does a hexagon have? Candidate answer: 6. Is the candidate answer correct? Reply with exactly YES or NO.",
    ),
    (
        "math-7",
        "Question: What is 15% of 200? Candidate answer: 30. Is the candidate answer correct? Reply with exactly YES or NO.",
    ),
    (
        "math-8",
        "Question: Solve 3x = 12. Candidate answer: x = 4. Is the candidate answer correct? Reply with exactly YES or NO.",
    ),
]


def score_once(tag: str) -> dict:
    """Score all fixed inputs sequentially; return {id: response_text}."""
    out = {}
    for qid, prompt in JUDGE_PROMPTS:
        code, body = http_post(
            "/generate",
            {
                "messages": [{"role": "user", "content": prompt}],
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
        )
        if code != 200:
            raise RuntimeError(f"generate failed for {qid}: {code} {body}")
        out[qid] = body["response"]
    return out


def compare_scores(a: dict, b: dict, label: str) -> list:
    diffs = [(qid, a[qid], b[qid]) for qid in a if a[qid] != b.get(qid)]
    return diffs


# ---------------------------------------------------------------------------
# Background load generator: continuous scoring through both scale ops.
# The drain handshake is only proven if live traffic exists while the victim
# is being drained, so this runs from before scale-out until after scale-in.
# ---------------------------------------------------------------------------
class LoadGenerator:
    def __init__(self, ev: "Evidence", workers: int = 8):
        self.ev = ev
        self.workers = workers
        self._stop = threading.Event()
        self._results: list = []
        self._lock = threading.Lock()
        self._pool = None

    def _worker(self, wid: int) -> None:
        i = 0
        while not self._stop.is_set():
            qid, prompt = JUDGE_PROMPTS[i % len(JUDGE_PROMPTS)]
            t = time.time()
            try:
                code, body = http_post(
                    "/generate",
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                    },
                    timeout=300,
                )
                ok = code == 200
                text = body["response"] if ok else str(body)
            except Exception as exc:  # noqa: BLE001
                ok, text = False, f"{type(exc).__name__}: {exc}"
            dt = time.time() - t
            with self._lock:
                self._results.append(
                    {"t": round(t - self.ev.t0, 3), "ok": ok, "latency": round(dt, 3), "text": text[:40]}
                )
            i += 1

    def start(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=self.workers)
        for wid in range(self.workers):
            self._pool.submit(self._worker, wid)
        self.ev.log("load_started", workers=self.workers)

    def stop(self) -> dict:
        self._stop.set()
        self._pool.shutdown(wait=True)
        total = len(self._results)
        failed = [r for r in self._results if not r["ok"]]
        lats = sorted(r["latency"] for r in self._results)

        def p(q):
            return lats[min(len(lats) - 1, int(q * len(lats)))] if lats else None

        summary = {
            "total": total,
            "failed": len(failed),
            "p50": p(0.50),
            "p95": p(0.95),
            "max": lats[-1] if lats else None,
            "failures": failed[:20],
        }
        self.ev.log("load_stopped", **{k: summary[k] for k in ("total", "failed", "p50", "p95", "max")})
        with open(os.path.join(self.ev.out_dir, "load_requests.json"), "w") as f:
            json.dump(self._results, f, indent=2)
        return summary


# ---------------------------------------------------------------------------
# Scale orchestration with full transition recording
# ---------------------------------------------------------------------------
def run_scale_op(ev: "Evidence", direction: str, target: int, expect_terminal: str, timeout_s: float) -> dict:
    t_start = time.time() - ev.t0  # relative to the evidence clock (same as load records)
    code, body = http_post(f"/{direction}", {"num_replicas": target, "timeout_secs": timeout_s})
    if code != 200:
        raise RuntimeError(f"{direction} submit failed: {code} {body}")
    ev.log(
        f"{direction}_submitted",
        request_id=body.get("request_id"),
        status=body.get("status"),
        current=body.get("current"),
    )
    if body.get("status") == "NOOP":
        raise RuntimeError(f"{direction} unexpectedly NOOP: {body}")
    rid = body["request_id"]

    seen: list = []
    deadline = time.time() + timeout_s + 300
    last = None
    final = None
    while time.time() < deadline:
        st = http_get(f"/{direction}/{rid}")
        if st["status"] != last:
            seen.append(st["status"])
            ev.log(f"{direction}_status", status=st["status"], current=st.get("current"), ready=st.get("ready"))
            last = st["status"]
        if st["status"] in ("ACTIVE", "PARTIAL", "FAILED", "COMPLETED"):
            final = st
            break
        time.sleep(1.0)
    if final is None:
        raise RuntimeError(f"{direction} {rid} did not reach a terminal state in time (last={last})")
    if final["status"] != expect_terminal:
        raise RuntimeError(f"{direction} {rid} ended as {final['status']}, expected {expect_terminal}: {final}")
    ev.log(
        f"{direction}_final",
        status=final["status"],
        current=final.get("current"),
        ready=final.get("ready"),
        created=final.get("created"),
        removed=final.get("removed"),
        transitions=seen,
    )
    return {
        "request_id": rid,
        "transitions": seen,
        "window": (t_start, time.time() - ev.t0),
        "final": {k: final.get(k) for k in ("status", "current", "ready", "created", "removed", "failed")},
    }


def load_requests(load: "LoadGenerator") -> list:
    """Load-generator results, safe to call before/after stop()."""
    with load._lock:
        return list(load._results)


def engines() -> dict:
    snap = http_get("/engines")
    return {
        "current": snap.get("current"),
        "ready": snap.get("ready"),
        "engines": snap.get("engines"),
    }


def engine_ids(snap: dict) -> set:
    """Stable engine identity = (host, port) of each live engine."""
    return {(e["host"], e["port"]) for e in (snap.get("engines") or [])}


def served_by(snap: dict, eid: tuple) -> int:
    for e in snap.get("engines") or []:
        if (e["host"], e["port"]) == eid:
            return e.get("served", 0)
    return -1


def wait_for_engines(expected: int, timeout_s: float = 900) -> dict:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            snap = engines()
            if snap["current"] == expected:
                return snap
            if snap != last:
                print(f"    engines: {snap}", flush=True)
                last = snap
        except Exception as exc:  # Serve replica still booting
            print(f"    waiting for serve replica: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(5.0)
    raise RuntimeError(f"engines did not reach {expected} within {timeout_s}s: {engines()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--genrm-num-gpus", type=int, default=1, help="initial engine count")
    parser.add_argument("--scale-out-timeout", type=float, default=900.0)
    parser.add_argument("--scale-in-timeout", type=float, default=900.0)
    args_cli = parser.parse_args()

    out_dir = args_cli.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", f"e2e_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    ev = Evidence(out_dir)
    ev.log("e2e_start", model_path=args_cli.model_path, repo_root=REPO_ROOT, out_dir=out_dir)

    import ray
    from ray import serve

    from relax.components.genrm import GenRM
    from relax.core.service import create_placement_group
    from relax.utils.utils import get_serve_url

    # Cleanup contract: the outer finally covers every failure exit
    # (deployment prep, load, sampling, assertions, file writes); each
    # cleanup step handles its own exception; evidence dumping is never a
    # precondition for cleanup; and only resources this run created are
    # released. Functional and cleanup outcomes are reported separately.
    load = None
    load_stopped = False
    verdicts: dict = {}
    functional_error = None
    pg = None
    app_started = False
    cleanup_pass = True
    cleanup_errors: list = []
    leftover_free_gpus = None

    try:
        ray.init(ignore_reinit_error=True)
        ev.log(
            "ray_init",
            node_ips=[n["NodeManagerAddress"] for n in ray.nodes() if n.get("Alive")],
            resources={k: v for k, v in ray.cluster_resources().items() if k.startswith("GPU") or k == "CPU"},
        )

        # Minimal production-shaped config namespace (the same fields the real
        # entrypoint resolves for the single-instance --genrm-model-path path).
        cfg = Namespace(
            genrm_model_path=os.path.abspath(args_cli.model_path),
            genrm_num_gpus=args_cli.genrm_num_gpus,
            genrm_num_gpus_per_engine=1,
            genrm_engine_config={"mem_fraction_static": 0.85},
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

        deployment = GenRM.bind(None, pg, cfg.genrm_num_gpus, cfg, "genrm")
        serve.run(deployment, name="genrm", route_prefix="/genrm")
        app_started = True
        global GENRM_BASE
        GENRM_BASE = get_serve_url("/genrm")
        # Single-node E2E: inside this container the Serve HTTP proxy binds to
        # localhost only (verified: 127.0.0.1 reachable, container IP refused),
        # so rewrite the host while keeping the port and path.
        from urllib.parse import urlsplit, urlunsplit

        _u = urlsplit(GENRM_BASE)
        GENRM_BASE = urlunsplit((_u.scheme, f"127.0.0.1:{_u.port}", _u.path, "", ""))
        ev.log("serve_run", url=GENRM_BASE)

        # ------------------------------------------------------------------ #
        # Phase 0: baseline (1 engine)
        # ------------------------------------------------------------------ #
        base_engines = wait_for_engines(cfg.genrm_num_gpus)
        ev.log("phase0_engines", **base_engines)
        ev.snapshot_gpu("baseline-1-engine")
        baseline_free_gpus = ray.available_resources().get("GPU", 0)
        ev.log("baseline_ray_free_gpus", free=baseline_free_gpus)
        scores_0 = score_once("baseline")
        ev.log("phase0_scores", scores=scores_0)
        initial_ids = engine_ids(base_engines)
        if len(initial_ids) != cfg.genrm_num_gpus:
            raise RuntimeError(f"expected {cfg.genrm_num_gpus} initial engine ids, got {initial_ids}")
        ev.log("phase0_initial_engine_ids", ids=sorted(map(list, initial_ids)))

        load = LoadGenerator(ev)
        verdicts: dict = {}
        passed = False
        try:
            load.start()
            time.sleep(5.0)  # let the load ramp before scaling

            # -------------------------------------------------------------- #
            # Phase 1: scale out 1 -> 2
            # -------------------------------------------------------------- #
            out_op = run_scale_op(ev, "scale_out", cfg.genrm_num_gpus + 1, "ACTIVE", args_cli.scale_out_timeout)
            after_out = wait_for_engines(cfg.genrm_num_gpus + 1, timeout_s=60)
            ev.log("phase1_engines", **after_out)
            ev.snapshot_gpu("after-scale-out-2-engines")
            ids_after_out = engine_ids(after_out)
            added_ids = ids_after_out - initial_ids
            if len(added_ids) != 1:
                raise RuntimeError(f"scale-out added {sorted(map(list, added_ids))}, expected exactly 1 new engine id")
            elastic_id = next(iter(added_ids))
            ev.log(
                "phase1_elastic_engine_id",
                id=list(elastic_id),
                initial_survived=sorted(map(list, initial_ids & ids_after_out)),
            )
            served_elastic_t1 = served_by(after_out, elastic_id)
            ev.log("phase1_served_counts", elastic_served=served_elastic_t1)

            scores_1 = score_once("after-scale-out")
            ev.log("phase1_scores", scores=scores_1)

            # Let the load generator hit the elastic engine for a while.
            time.sleep(15.0)
            mid_out = engines()
            served_elastic_t2 = served_by(mid_out, elastic_id)
            ev.log(
                "phase1_served_counts_mid",
                elastic_served=served_elastic_t2,
                **{"current": mid_out["current"], "ready": mid_out["ready"]},
            )

            time.sleep(5.0)

            # -------------------------------------------------------------- #
            # Phase 2: scale in 2 -> 1 (victim drained while load is running)
            # -------------------------------------------------------------- #
            in_op = run_scale_op(ev, "scale_in", cfg.genrm_num_gpus, "COMPLETED", args_cli.scale_in_timeout)
            after_in = wait_for_engines(cfg.genrm_num_gpus, timeout_s=60)
            ev.log("phase2_engines", **after_in)
            ev.snapshot_gpu("after-scale-in-1-engine")
            ids_after_in = engine_ids(after_in)
            removed_ids = ids_after_out - ids_after_in
            scores_2 = score_once("after-scale-in")
            ev.log("phase2_scores", scores=scores_2)

            load_summary = load.stop()
            load_stopped = True

            # Requests served during each scale-op window (drain proof needs live
            # traffic overlapping the operation, not merely zero failures).
            out_w = out_op["window"]
            in_w = in_op["window"]
            reqs_during_out = sum(1 for r in load_requests(load) if out_w[0] <= r["t"] <= out_w[1] and r["ok"])
            reqs_during_in = sum(1 for r in load_requests(load) if in_w[0] <= r["t"] <= in_w[1] and r["ok"])

            # Resource release: Ray free GPUs must return to the baseline level
            # (the elastic engine's dedicated PG returned its GPU).
            free_after_in = ray.available_resources().get("GPU", 0)

            # ------------------------------------------------------------------ #
            # Verdicts
            # ------------------------------------------------------------------ #
            verdicts = {
                "scale_out_active": out_op["final"]["status"] == "ACTIVE",
                "scale_out_transition_chain": out_op["transitions"],
                "scale_out_added_exactly_one_engine": len(added_ids) == 1,
                "scale_out_initial_survived": initial_ids <= ids_after_out,
                "elastic_engine_served_traffic": served_elastic_t2 > served_elastic_t1,
                "scale_in_completed": in_op["final"]["status"] == "COMPLETED",
                "scale_in_transition_chain": in_op["transitions"],
                "scale_in_removed_exactly_the_elastic_engine": removed_ids == {elastic_id},
                "scale_in_initial_survived": initial_ids <= ids_after_in,
                "scoring_identical_out": compare_scores(scores_0, scores_1, "out") == [],
                "scoring_identical_in": compare_scores(scores_0, scores_2, "in") == [],
                "load_zero_failures": load_summary["failed"] == 0,
                "load_total_requests": load_summary["total"],
                "load_requests_during_scale_out": reqs_during_out,
                "load_requests_during_scale_in": reqs_during_in,
                "load_served_during_both_ops": reqs_during_out > 0 and reqs_during_in > 0,
                "gpu_resources_returned": free_after_in >= baseline_free_gpus,
                "baseline_free_gpus": baseline_free_gpus,
                "free_gpus_after_scale_in": free_after_in,
                "scores_diff_out": compare_scores(scores_0, scores_1, "out"),
                "scores_diff_in": compare_scores(scores_0, scores_2, "in"),
            }
            passed = all(v for k, v in verdicts.items() if isinstance(v, bool))
            verdicts["E2E_PASS"] = passed
            ev.log("verdicts", **verdicts)
        finally:
            # Load generator must stop even when a phase raises.
            if load is not None and not load_stopped and load._pool is not None:
                load.stop()
            ev.dump()

    except Exception as exc:  # noqa: BLE001
        functional_error = exc
        ev.log("functional_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        # Every step handles its own exception: a failure here never masks
        # the original error and never blocks the remaining steps.
        try:
            ev.dump()
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: evidence dump failed: {exc}", file=sys.stderr, flush=True)
        try:
            if app_started:
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
        try:
            ev.dump()
        except Exception:  # noqa: BLE001
            pass

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
