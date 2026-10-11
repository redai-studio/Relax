# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Task 4 failure-injection E2E: real drain/abort/crash paths on GPU.

Three externally injectable failure scenarios against the live service
(4x4090 + Qwen3-0.6B, single Gateway):

  S1  long request across drain (happy path): two long generations are
      in flight, one on each engine; scale-in drains the victim while its
      request keeps decoding. The request must finish successfully on the
      victim and scale-in must reach COMPLETED only after it.

  S2  watcher deadline abort + fixed-victim reconcile: a long request on
      the victim outlives the operation's timeout_secs. The watcher aborts
      at the deadline and fails the registry dirty (model mutex held); the
      manager lifecycle is parked in its uninterruptible drain wait until
      the drain timeout -- the fence trades latency for safety, which this
      scenario measures. Reconcile then completes the original operation
      with the original victim once physical work is done.

  S3  engine crash during drain: the victim process is SIGKILLed while
      draining a long request. The in-flight request must be retried onto
      the surviving engine (per-request re-route), the drain must confirm
      once the victim's accounting clears, and scale-in must complete with
      the initial engine intact.

Late-init, health-check failure and PG-cleanup-failure exits are covered by
CPU fault-injection tests (they need internal seams, not external ones).

Usage (repo root, GPUs free):

    python demos/task4_genrm/e2e_failure_injection.py \
        --model-path /path/to/Qwen3-0.6B
"""

import argparse
import json
import os
import sys
import threading
import time
from argparse import Namespace


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

GENRM_BASE = None


class Evidence:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.t0 = time.time()
        self.events: list = []

    def log(self, event: str, **fields) -> None:
        entry = {"t": round(time.time() - self.t0, 3), "event": event, **fields}
        self.events.append(entry)
        print(f"[{entry['t']:8.3f}s] {event} {json.dumps(fields, ensure_ascii=False)[:220]}", flush=True)

    def dump(self) -> None:
        with open(os.path.join(self.out_dir, "events.json"), "w") as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)


import requests  # noqa: E402


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
    return {"current": snap.get("current"), "engines": snap.get("engines")}


def engine_ids(snap: dict) -> set:
    return {(e["host"], e["port"]) for e in (snap.get("engines") or [])}


def wait_for_engines(expected: int, timeout_s: float = 900) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            snap = engines()
            if snap["current"] == expected:
                return snap
        except Exception as exc:  # Serve replica still booting
            print(f"    waiting: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(3.0)
    raise RuntimeError(f"engines did not reach {expected} within {timeout_s}s")


def submit_scale(direction: str, target: int, timeout_secs: float) -> str:
    body = http_post(f"/{direction}", {"num_replicas": target, "timeout_secs": timeout_secs})
    if body.get("status") != "PENDING" or not body.get("request_id"):
        raise RuntimeError(f"{direction} submit: {body}")
    return body["request_id"]


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


def scale_out_to_2(ev: "Evidence", num_gpus: int) -> tuple:
    rid = submit_scale("scale_out", num_gpus + 1, 900.0)
    ev.log("scale_out_submitted", request_id=rid)
    final = poll_status("scale_out", rid, {"ACTIVE", "PARTIAL", "FAILED"}, 900)
    ev.log("scale_out_final", status=final["status"])
    if final["status"] != "ACTIVE":
        raise RuntimeError(f"scale-out failed: {final}")
    after = wait_for_engines(num_gpus + 1, timeout_s=60)
    return engine_ids(after)


def victim_inflight(victim: tuple) -> int:
    for e in engines().get("engines") or []:
        if (e["host"], e["port"]) == victim:
            return int(e.get("inflight") or 0)
    return 0


def await_victim_inflight(ev, victim: tuple, timeout_s: float = 30) -> int:
    """Block until the victim engine reports >=1 in-flight request; every
    drain/deadline proof below depends on this precondition (v3 of this run
    failed because the requests had already EOS'd before the scale-in)."""
    deadline = time.time() + timeout_s
    last = -1
    while time.time() < deadline:
        last = victim_inflight(victim)
        if last >= 1:
            break
        time.sleep(0.5)
    ev.log("victim_inflight_confirmed", victim=list(victim), inflight=last)
    if last < 1:
        raise RuntimeError(f"victim {victim} never reported inflight>=1 (last={last})")
    return last


def pid_for_port(port: int, ev=None):
    """Find a LISTENing PID with psutil, then a namespace-robust procfs
    fallback.

    Dump the listen table on failure for diagnosis.
    """
    import psutil

    for conn in psutil.net_connections(kind="tcp"):
        if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN and conn.pid:
            return conn.pid
    # /proc fallback: find the socket inode for the port, then the pid whose
    # fd links to it.
    hexport = f"{port:04X}"
    inodes = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as f:
                next(f)
                for line in f:
                    parts = line.split()
                    if parts[1].split(":")[1] == hexport and parts[3] == "0A":  # LISTEN
                        inodes.add(parts[9])
        except OSError:
            continue
    for pid_dir in os.listdir("/proc"):
        if not pid_dir.isdigit():
            continue
        fd_dir = f"/proc/{pid_dir}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    target = os.readlink(f"{fd_dir}/{fd}")
                except OSError:
                    continue
                if target.startswith("socket:["):
                    inode = target[8:-1]
                    if inode in inodes:
                        return int(pid_dir)
        except OSError:
            continue
    if ev is not None:
        try:
            table = [
                (c.laddr.port if c.laddr else None, c.status, c.pid)
                for c in psutil.net_connections(kind="tcp")
                if c.status == psutil.CONN_LISTEN
            ]
            ev.log("pid_for_port_failed", port=port, listen_table=table[:40])
        except Exception:  # noqa: BLE001
            pass
    return None


class LongRequest:
    """One long generation in a background thread; records the attributed
    engine, finish reason and success."""

    def __init__(self, prompt: str, max_new_tokens: int):
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.result = {}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        t = time.time()
        try:
            out = http_post(
                "/generate",
                {
                    "messages": [{"role": "user", "content": self.prompt}],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": self.max_new_tokens,
                        "ignore_eos": True,
                    },
                },
                timeout=900,
            )
            self.result = {
                "ok": True,
                "engine": f"{out.get('engine_host')}:{out.get('engine_port')}",
                "finish_reason": out.get("finish_reason"),
                "len": len(out.get("response", "")),
                "latency": round(time.time() - t, 3),
            }
        except Exception as exc:  # noqa: BLE001
            self.result = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "latency": round(time.time() - t, 3)}

    def start(self):
        self.thread.start()
        return self

    def join(self, timeout: float = 900):
        self.thread.join(timeout=timeout)
        return self.result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--genrm-num-gpus", type=int, default=1)
    args_cli = parser.parse_args()

    out_dir = args_cli.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", f"failure_injection_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    ev = Evidence(out_dir)
    ev.log("e2e_start", model_path=args_cli.model_path, out_dir=out_dir)

    import ray
    from ray import serve
    from ray.util.placement_group import placement_group_table, remove_placement_group

    from relax.components.genrm import GenRM
    from relax.core.service import create_placement_group
    from relax.utils.utils import get_serve_url

    verdicts: dict = {}
    functional_error = None
    pg = None
    app_started = False
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
        serve.run(GenRM.bind(None, pg, cfg.genrm_num_gpus, cfg, "genrm"), name="genrm", route_prefix="/genrm")
        app_started = True
        global GENRM_BASE
        GENRM_BASE = get_serve_url("/genrm")
        from urllib.parse import urlsplit, urlunsplit

        _u = urlsplit(GENRM_BASE)
        GENRM_BASE = urlunsplit((_u.scheme, f"127.0.0.1:{_u.port}", _u.path, "", ""))
        ev.log("serve_run", url=GENRM_BASE)

        base = wait_for_engines(cfg.genrm_num_gpus)
        initial_ids = engine_ids(base)
        ev.log("baseline_engines", ids=sorted(map(list, initial_ids)))

        prompt = (
            "You are a meticulous mathematics reviewer. Reproduce the full derivation of "
            "the quadratic formula step by step, check each algebraic step, and list every "
            "assumption you make. Take your time and be exhaustive. "
        ) * 3

        # ------------------------------------------------------------------ #
        # S1: long request across drain (happy path)
        # ------------------------------------------------------------------ #
        ids_after = scale_out_to_2(ev, cfg.genrm_num_gpus)
        elastic = (ids_after - initial_ids).pop()
        ev.log("s1_engines", initial=sorted(map(list, initial_ids)), elastic=list(elastic))

        reqs = [LongRequest(f"[s1-{i}]\n{prompt}", 4000).start() for i in range(2)]
        await_victim_inflight(ev, elastic)
        rid = submit_scale("scale_in", cfg.genrm_num_gpus, 600.0)
        ev.log("s1_scale_in_submitted", request_id=rid)
        final = poll_status("scale_in", rid, {"COMPLETED", "FAILED"}, 600)
        results = [r.join() for r in reqs]
        ev.log("s1_done", scale_in=final["status"], requests=results)
        # The long generations are capped by max_new_tokens on purpose; the
        # drain proof only needs them to complete successfully on the victim.
        verdicts["s1_scale_in_completed"] = final["status"] == "COMPLETED"
        verdicts["s1_both_long_requests_ok"] = all(r.get("ok") for r in results)
        verdicts["s1_requests_served_by_both_engines"] = len({r.get("engine") for r in results if r.get("ok")}) == 2
        verdicts["s1_final_capacity"] = engines()["current"]

        # ------------------------------------------------------------------ #
        # S2: watcher deadline abort -> dirty FAILED -> parked lifecycle ->
        #     fixed-victim reconcile completes the original operation
        # ------------------------------------------------------------------ #
        ids_after = scale_out_to_2(ev, cfg.genrm_num_gpus)
        elastic2 = (ids_after - initial_ids).pop()
        ev.log("s2_engines", ids=sorted(map(list, ids_after)), victim=list(elastic2))
        # Two long requests so one is in flight on the victim (round-robin
        # alternates engines). With ignore_eos, 8000 tokens decode ~17 s at
        # ~470 tok/s -- comfortably outliving the 5 s operation deadline, so
        # the watcher provably aborts while the victim request is still
        # running. The precondition is asserted, not assumed (v3 lesson).
        longs2 = [LongRequest(f"[s2-{i}]\n{prompt}", 8000).start() for i in range(2)]
        await_victim_inflight(ev, elastic2)
        rid2 = submit_scale("scale_in", cfg.genrm_num_gpus, 5.0)  # deadline << decode
        ev.log("s2_scale_in_submitted", request_id=rid2, timeout_secs=5.0)

        # The watcher aborts at the deadline and fails the registry dirty;
        # new scale operations for the model must be rejected meanwhile.
        t_abort = time.time()
        final2 = poll_status("scale_in", rid2, {"COMPLETED", "FAILED"}, 180)
        ev.log("s2_registry_failed_dirty", status=final2["status"], cleanup_required=final2.get("cleanup_required"))
        verdicts["s2_deadline_abort_failed_dirty"] = final2["status"] == "FAILED" and bool(
            final2.get("cleanup_required")
        )
        try:
            http_post("/scale_in", {"num_replicas": cfg.genrm_num_gpus, "timeout_secs": 60.0})
            mutex_held = False
        except RuntimeError:
            mutex_held = True
        verdicts["s2_mutex_held_while_dirty"] = mutex_held
        ev.log("s2_mutex_check", held=mutex_held)

        # The manager lifecycle is parked in its uninterruptible drain wait;
        # it surfaces as physical completion only after the drain timeout
        # (default 600 s) because the aborted watcher no longer confirms the
        # drain. Reconcile is rejected/keeps reporting cleanup_required until
        # then; acceptance with cleanup_required=False proves the park ended
        # and the original operation's cleanup finished.
        rec = None
        deadline = time.time() + 900
        while time.time() < deadline:
            try:
                rec = http_post(f"/scale_in/{rid2}/reconcile", {})
                ev.log("s2_reconcile_result", **{k: rec.get(k) for k in ("status", "cleanup_required", "detail")})
                if rec.get("cleanup_required") is False:
                    break
            except RuntimeError as exc:
                ev.log("s2_reconcile_retry", error=str(exc)[:120])
            time.sleep(10.0)
        verdicts["s2_reconcile_cleared_park"] = bool(rec) and rec.get("cleanup_required") is False
        verdicts["s2_reconcile_terminal_status"] = bool(rec) and rec.get("status") in ("COMPLETED", "FAILED")
        verdicts["s2_abort_to_reconcile_park_s"] = round(time.time() - t_abort, 1)
        results2 = [r.join() for r in longs2]
        ev.log("s2_requests", requests=results2)
        verdicts["s2_long_requests_ok"] = all(r.get("ok", False) for r in results2)
        verdicts["s2_final_capacity"] = engines()["current"]

        # ------------------------------------------------------------------ #
        # S3: SIGKILL the victim while it drains a long request; the request
        #     must be retried onto the surviving engine
        # ------------------------------------------------------------------ #
        ids_after = scale_out_to_2(ev, cfg.genrm_num_gpus)
        elastic3 = (ids_after - initial_ids).pop()
        ev.log("s3_engines", initial=sorted(map(list, initial_ids)), victim=list(elastic3))
        longs3 = [LongRequest(f"[s3-{i}]\n{prompt}", 8000).start() for i in range(2)]
        await_victim_inflight(ev, elastic3)
        rid3 = submit_scale("scale_in", cfg.genrm_num_gpus, 600.0)
        ev.log("s3_scale_in_submitted", request_id=rid3)

        # Victim selection is newest-first (RFC), i.e. the elastic engine.
        # Wait until the op is DRAINING with a live request on the victim,
        # then SIGKILL the victim process.
        deadline = time.time() + 120
        while time.time() < deadline:
            st = http_get(f"/scale_in/{rid3}")
            if st["status"] in ("DRAINING", "REMOVING"):
                break
            time.sleep(1.0)
        victim_port = elastic3[1]
        pid = None
        for _ in range(10):  # the listener is momentarily busy, retry
            pid = pid_for_port(victim_port, ev)
            if pid:
                break
            time.sleep(1.0)
        ev.log("s3_victim_identified", port=victim_port, pid=pid, status=st["status"])
        if pid is None:
            raise RuntimeError(f"could not find PID for victim port {victim_port}")
        os.kill(pid, 9)
        ev.log("s3_victim_killed", pid=pid)

        final3 = poll_status("scale_in", rid3, {"COMPLETED", "FAILED"}, 600)
        results3 = [r.join() for r in longs3]
        ev.log("s3_done", scale_in=final3["status"], requests=results3)
        initial_engine_strs = {f"{h}:{p}" for h, p in initial_ids}
        verdicts["s3_scale_in_completed_after_crash"] = final3["status"] == "COMPLETED"
        # The crash interrupts the victim's in-flight request; the component
        # must retry it onto the surviving engine (attribution follows the
        # retry re-pick), so every surviving reply comes from the initial
        # engine and no request is lost.
        verdicts["s3_all_requests_survived_on_initial_engine"] = all(
            r.get("ok") and r.get("engine") in initial_engine_strs for r in results3
        )
        verdicts["s3_initial_engine_survived"] = initial_ids <= engine_ids(engines())
        verdicts["s3_final_capacity"] = engines()["current"]

        verdicts["E2E_PASS"] = all(v for k, v in verdicts.items() if isinstance(v, bool))
        ev.log("verdicts", **verdicts)
    except Exception as exc:  # noqa: BLE001
        functional_error = exc
        ev.log("functional_error", error=f"{type(exc).__name__}: {exc}")
    finally:
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
