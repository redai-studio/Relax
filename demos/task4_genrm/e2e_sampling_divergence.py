# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Task 4 sampling-divergence E2E: does replica-invariant server seeding
survive *divergent request histories*?

The reward-consistency run (``e2e_reward_consistency.py``) probes two engines
whose request histories are essentially synchronized (round-robin from the
first request), so both RNG streams advance in lockstep. A production
autoscaler scale-out is different: the initial engine A has already served
an arbitrary number of stochastic requests when the elastic engine B boots
with the same server seed but a fresh RNG state. This driver closes that
evidence gap:

  1. engine A alone serves ``--divergence-requests`` stochastic judge
     requests (official sampling, distinct dataset rows);
  2. ``scale_out`` boots elastic engine B (same ``args.seed``);
  3. the same fixed probe inputs are sent through the service and attributed
     per engine; per-input verdict *sets* must agree across engines.

Verdict: ``flip_count == 0`` -> consistency holds after divergent request
histories for this tested configuration. Any flip is recorded in full (case,
per-engine verdicts, texts) -- the follow-up protocol (request-level
``sampling_seed`` etc.) is a code change, never a parameter tune.

Usage (repo root, python with ray/sglang/torch, GPUs free):

    python demos/task4_genrm/e2e_sampling_divergence.py \
        --model-path /path/to/Qwen3-0.6B \
        --dataset /path/to/dapo-math-17k.jsonl
"""

import argparse
import json
import os
import re
import sys
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
        print(f"[{entry['t']:8.3f}s] {event} {json.dumps(fields, ensure_ascii=False)[:200]}", flush=True)

    def dump(self) -> None:
        with open(os.path.join(self.out_dir, "events.json"), "w") as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)


import requests  # noqa: E402


def http_get(path: str, timeout: float = 60):
    r = requests.get(f"{GENRM_BASE}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def http_post(path: str, body: dict, timeout: float = 300):
    r = requests.post(f"{GENRM_BASE}{path}", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


import importlib.util  # noqa: E402


_spec = importlib.util.spec_from_file_location(
    "_task4_dapo_genrm",
    os.path.join(REPO_ROOT, "relax", "engine", "rewards", "dapo_genrm.py"),
)
_dapo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dapo)
_format_messages = _dapo._format_messages

_STRICT_RE = re.compile(r"^\s*(?:Judgement:\s*)?([01])\s*$")


def strict_parse(text: str):
    m = _STRICT_RE.match((text or "").strip())
    return int(m.group(1)) if m else None


def loose_parse(text: str):
    prediction = (text or "").strip()
    if "Judgement:" in prediction:
        prediction = prediction.split("Judgement:")[-1].strip()
    head = prediction[:16]
    if "1" in head:
        return 1
    if "0" in head:
        return 0
    return None


def load_rows(dataset_path: str, start: int, count: int):
    """``count`` judge-shaped requests from dataset rows, skipping the first
    ``start`` usable rows.

    Each row yields one request (question + ground truth as the model answer).
    Rows used here are disjoint from the probe inputs by construction (the
    probe takes rows ``[0, 2*num_pairs)``). The skip is applied to *usable*
    rows, so the file is scanned until ``start + count`` usable rows were seen
    (review finding: slicing a ``count``-capped list silently returned fewer
    rows than requested).
    """
    rows = []
    seen = 0
    with open(dataset_path) as f:
        for line in f:
            if len(rows) >= count:
                break
            row = json.loads(line)
            prompt = row.get("prompt") or []
            question = next((turn["content"] for turn in reversed(prompt) if turn.get("role") == "user"), None)
            label = row.get("label")
            if not question or label is None or str(label).strip() == "":
                continue
            seen += 1
            if seen <= start:
                continue
            rows.append({"question": question, "ground_truth": str(label).strip()})
    return rows


def load_probe_inputs(dataset_path: str, num_pairs: int) -> list:
    """Same fixed probe set as the reward-consistency driver (positive +
    corrupted-negative pairs from the first rows)."""
    cases = []
    for row in load_rows(dataset_path, 0, 4 * num_pairs):
        gt = row["ground_truth"]
        cases.append({"id": f"pos-{len(cases)}", "question": row["question"], "ground_truth": gt, "model_answer": gt})
        cases.append(
            {"id": f"neg-{len(cases)}", "question": row["question"], "ground_truth": gt, "model_answer": f"{gt} 999"}
        )
        if len(cases) >= 2 * num_pairs:
            break
    return cases


def run_scale_op(ev: "Evidence", direction: str, target: int, expect_terminal: str, timeout_s: float) -> dict:
    body = http_post(f"/{direction}", {"num_replicas": target, "timeout_secs": timeout_s})
    ev.log(f"{direction}_submitted", request_id=body.get("request_id"), status=body.get("status"))
    if body.get("status") == "NOOP":
        raise RuntimeError(f"{direction} unexpectedly NOOP: {body}")
    rid = body["request_id"]
    seen: list = []
    final = None
    last = None
    deadline = time.time() + timeout_s + 300
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
    ev.log(f"{direction}_final", status=final["status"], current=final.get("current"), transitions=seen)
    return {"request_id": rid, "transitions": seen, "final": final}


def engines() -> dict:
    snap = http_get("/engines")
    return {"current": snap.get("current"), "engines": snap.get("engines") or []}


def engine_ids(snap: dict) -> set:
    return {(e["host"], e["port"]) for e in snap["engines"]}


def wait_for_engines(expected: int, timeout_s: float = 900) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            snap = engines()
            if snap["current"] == expected:
                return snap
        except Exception as exc:
            print(f"    waiting for serve replica: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(5.0)
    raise RuntimeError(f"engines did not reach {expected} within {timeout_s}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--genrm-num-gpus", type=int, default=1)
    parser.add_argument("--divergence-requests", type=int, default=300)
    parser.add_argument("--num-pairs", type=int, default=25)
    parser.add_argument("--repeats-per-input", type=int, default=8)
    parser.add_argument("--scale-out-timeout", type=float, default=900.0)
    parser.add_argument("--scale-in-timeout", type=float, default=900.0)
    args_cli = parser.parse_args()

    out_dir = args_cli.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", f"sampling_divergence_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    ev = Evidence(out_dir)
    ev.log("e2e_start", model_path=args_cli.model_path, dataset=args_cli.dataset, out_dir=out_dir)

    import ray
    from ray import serve

    from relax.components.genrm import GenRM
    from relax.core.service import create_placement_group
    from relax.utils.utils import get_serve_url

    verdicts: dict = {}
    replies: list = []
    functional_error = None
    pg = None
    app_started = False
    cleanup_pass = True
    cleanup_errors: list = []
    leftover_free_gpus = None
    phase = "setup"

    try:
        ray.init(ignore_reinit_error=True)
        ev.log("ray_init", gpus=ray.cluster_resources().get("GPU", 0))

        cfg = Namespace(
            genrm_model_path=os.path.abspath(args_cli.model_path),
            genrm_num_gpus=args_cli.genrm_num_gpus,
            genrm_num_gpus_per_engine=1,
            genrm_engine_config={"mem_fraction_static": 0.85},
            genrm_sampling_config={
                "temperature": 0.1,
                "top_p": 1.0,
                "top_k": -1,
                "max_response_len": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            },
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
        ev.log("phase0_engines", ids=sorted(map(list, initial_ids)))
        initial_engine = f"{next(iter(initial_ids))[0]}:{next(iter(initial_ids))[1]}"

        # ---------------------------------------------------------------- #
        # Phase 1: engine A alone serves N stochastic requests (rows disjoint
        # from the probe set), advancing its RNG state N draws beyond a
        # freshly booted replica.
        # ---------------------------------------------------------------- #
        divergence_rows = load_rows(args_cli.dataset, 4 * args_cli.num_pairs, args_cli.divergence_requests)
        if len(divergence_rows) < args_cli.divergence_requests:
            raise RuntimeError(
                f"only {len(divergence_rows)} divergence rows available, need {args_cli.divergence_requests}"
            )
        served_before = {e["host"] + ":" + str(e["port"]): e.get("served", 0) for e in engines()["engines"]}
        phase = "divergence"
        for i, row in enumerate(divergence_rows):
            messages = _format_messages(row["question"], row["ground_truth"], row["ground_truth"])
            out = http_post("/generate", {"messages": messages})
            eng = f"{out.get('engine_host')}:{out.get('engine_port')}"
            if eng != initial_engine:
                raise RuntimeError(f"divergence request {i} served by {eng}, expected only {initial_engine}")
        served_after = {e["host"] + ":" + str(e["port"]): e.get("served", 0) for e in engines()["engines"]}
        a_served = served_after.get(initial_engine, 0) - served_before.get(initial_engine, 0)
        ev.log(
            "divergence_phase_done",
            requests=args_cli.divergence_requests,
            engine_a_served_delta=a_served,
            history_divergence_request_count=a_served,
        )

        # ---------------------------------------------------------------- #
        # Phase 2: scale out; B boots with the same server seed (replica-
        # invariant) but a fresh RNG state.
        # ---------------------------------------------------------------- #
        run_scale_op(ev, "scale_out", cfg.genrm_num_gpus + 1, "ACTIVE", args_cli.scale_out_timeout)
        after_out = wait_for_engines(cfg.genrm_num_gpus + 1, timeout_s=60)
        elastic_ids = engine_ids(after_out) - initial_ids
        if len(elastic_ids) != 1:
            raise RuntimeError(f"expected exactly 1 elastic engine, got {sorted(map(list, elastic_ids))}")
        elastic_id = next(iter(elastic_ids))
        elastic_engine = f"{elastic_id[0]}:{elastic_id[1]}"
        ev.log("phase2_engines", initial=initial_engine, elastic=elastic_engine)

        # ---------------------------------------------------------------- #
        # Phase 3: fixed probe inputs through the service; round-robin
        # attribution covers both engines per input.
        # ---------------------------------------------------------------- #
        cases = load_probe_inputs(args_cli.dataset, args_cli.num_pairs)
        if not cases:
            raise RuntimeError(f"no usable probe inputs from {args_cli.dataset}")
        ev.log("probe_inputs_loaded", cases=len(cases))
        phase = "probe"
        for case in cases:
            messages = _format_messages(case["question"], case["ground_truth"], case["model_answer"])
            for _ in range(args_cli.repeats_per_input):
                t = time.time()
                out = http_post("/generate", {"messages": messages})
                replies.append(
                    {
                        "case": case["id"],
                        "engine": f"{out.get('engine_host')}:{out.get('engine_port')}",
                        "finish_reason": out.get("finish_reason"),
                        "completion_tokens": out.get("completion_tokens"),
                        "text": out.get("response", ""),
                        "strict": strict_parse(out.get("response", "")),
                        "loose": loose_parse(out.get("response", "")),
                        "latency": round(time.time() - t, 3),
                    }
                )
        ev.log("probe_phase_done", replies=len(replies))

        with open(os.path.join(out_dir, "replies.json"), "w") as f:
            json.dump(replies, f, indent=2, ensure_ascii=False)

        # ---------------------------------------------------------------- #
        # Analysis: per-engine verdict sets must agree per input.
        # ---------------------------------------------------------------- #
        attribution_failures = []
        parse_gate_failures = []
        flips = []
        n_agree = 0
        for case in cases:
            per_engine = {}
            for label, engine in (("A_initial", initial_engine), ("B_elastic", elastic_engine)):
                case_replies = [r for r in replies if r["case"] == case["id"] and r["engine"] == engine]
                if not case_replies:
                    attribution_failures.append({"case": case["id"], "engine": label})
                    per_engine[label] = None
                    continue
                vals = {r["strict"] for r in case_replies if r["strict"] is not None}
                if not vals:
                    parse_gate_failures.append({"case": case["id"], "engine": label})
                    per_engine[label] = None
                    continue
                per_engine[label] = sorted(vals)
            if per_engine.get("A_initial") is None or per_engine.get("B_elastic") is None:
                continue
            if per_engine["A_initial"] == per_engine["B_elastic"]:
                n_agree += 1
            else:
                flips.append(
                    {
                        "case": case["id"],
                        "a_initial": per_engine["A_initial"],
                        "b_elastic": per_engine["B_elastic"],
                        "a_texts": [
                            r["text"] for r in replies if r["case"] == case["id"] and r["engine"] == initial_engine
                        ],
                        "b_texts": [
                            r["text"] for r in replies if r["case"] == case["id"] and r["engine"] == elastic_engine
                        ],
                    }
                )

        truncated = [r for r in replies if r["finish_reason"] not in (None, "stop")]
        compared = n_agree + len(flips)
        verdicts = {
            "experiment": "sampling-consistency under divergent request histories",
            "history_divergence_request_count": a_served,
            "sampling": cfg.genrm_sampling_config,
            "seed": cfg.seed,
            "cases": len(cases),
            "probe_replies_total": len(replies),
            "attribution_covers_both_engines": not attribution_failures,
            "attribution_failures": attribution_failures[:10],
            "parse_gate_independent": not parse_gate_failures,
            "parse_gate_failures": parse_gate_failures[:10],
            "no_truncated_replies": not truncated,
            "truncated_count": len(truncated),
            "compared_cases": compared,
            "flip_count": len(flips),
            "flip_rate": round(len(flips) / compared, 4) if compared else None,
            "flips": flips,
            "engine_a": initial_engine,
            "engine_b": elastic_engine,
        }
        verdicts["E2E_PASS"] = (
            verdicts["attribution_covers_both_engines"]
            and verdicts["parse_gate_independent"]
            and verdicts["no_truncated_replies"]
            and verdicts["flip_count"] == 0
        )
        ev.log("verdicts", **{k: v for k, v in verdicts.items() if isinstance(v, (bool, int, float, str, type(None)))})

        # Scale back in: the removed engine must be exactly the elastic one.
        run_scale_op(ev, "scale_in", cfg.genrm_num_gpus, "COMPLETED", args_cli.scale_in_timeout)
        final_ids = engine_ids(wait_for_engines(cfg.genrm_num_gpus, timeout_s=60))
        verdicts["scale_in_removed_exactly_elastic"] = final_ids == initial_ids
        ev.log("final_engines", ids=sorted(map(list, final_ids)))
        verdicts["E2E_PASS"] = verdicts["E2E_PASS"] and verdicts["scale_in_removed_exactly_elastic"]

    except Exception as exc:  # noqa: BLE001
        functional_error = exc
        import traceback

        traceback.print_exc()
        if phase == "probe":
            with open(os.path.join(out_dir, "replies.json"), "w") as f:
                json.dump(replies, f, indent=2, ensure_ascii=False)
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
            leftover_free_gpus=leftover_free_gpus,
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
        ev.dump()
    except Exception:  # noqa: BLE001
        pass
    print(
        json.dumps({k: v for k, v in verdicts.items() if isinstance(v, (bool, int, float, str, type(None)))}, indent=2)
    )
    return 0 if verdicts["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
