# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Task 4 reward-consistency E2E: identical inputs, both engines, real
protocol.

Exercises the production dapo-genrm judge protocol end-to-end: the exact
prompt template, in-context examples and loose parser from
``relax/engine/rewards/dapo_genrm.py`` (imported, not reimplemented), thinking
disabled via ``chat_template_kwargs`` (a hybrid-thinking judge would burn the
token budget on a <think> block), and per-reply engine attribution via the
``/generate`` response's ``engine_host``/``engine_port``/``finish_reason``
fields.

For every fixed input (positive: model answer == ground truth; negative:
corrupted answer) the binary verdicts produced by the initial and the elastic
engine must agree exactly. Parse success is gated independently per engine —
"a judge that fails to parse on both engines returns 0 on both" does not
count as agreement. A strict parser (``Judgement: 1``/``0`` only) is reported
next to the production loose parse; the greedy arm (temperature 0) is the
primary consistency criterion, the official-sampling arm (temperature 0.1,
per-engine seeds differ) additionally quantifies sampling stability.

Usage (repo root, python with ray/sglang/torch, GPUs free):

    python demos/task4_genrm/e2e_reward_consistency.py \
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


# ---------------------------------------------------------------------------
# Evidence recorder (same contract as the other task4 drivers)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Judge protocol: imported from the production reward path. Loaded by file
# path so this driver does not require the full rewards package (whose
# __init__ pulls optional deps like pylatexenc); dapo_genrm itself only
# needs re/httpx/the genrm client.
# ---------------------------------------------------------------------------
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
    """Strict verdict: the reply is exactly 'Judgement: 1'/'0' (or bare
    1/0)."""
    m = _STRICT_RE.match((text or "").strip())
    return int(m.group(1)) if m else None


def loose_parse(text: str):
    """Exact replica of the production parser in dapo_genrm (what training
    consumes); reported alongside the strict parse to quantify its noise."""
    prediction = (text or "").strip()
    if "Judgement:" in prediction:
        prediction = prediction.split("Judgement:")[-1].strip()
    head = prediction[:16]
    if "1" in head:
        return 1
    if "0" in head:
        return 0
    return None


# ---------------------------------------------------------------------------
# Fixed inputs from the real dataset: known-correct positives and negatives
# ---------------------------------------------------------------------------
def load_inputs(dataset_path: str, num_pairs: int) -> list:
    """``num_pairs`` dataset rows become 2 cases each: a positive (model
    answer.

    == ground truth, expected verdict 1) and a negative (corrupted answer,
    expected verdict 0).

    The ``model_answer`` plays the role of the extracted actor answer (what
    ``_extract_answer`` hands the judge).
    """
    cases = []
    with open(dataset_path) as f:
        for line in f:
            if len(cases) >= 2 * num_pairs:
                break
            row = json.loads(line)
            prompt = row.get("prompt") or []
            question = next((turn["content"] for turn in reversed(prompt) if turn.get("role") == "user"), None)
            label = row.get("label")
            if not question or label is None or str(label).strip() == "":
                continue
            gt = str(label).strip()
            cases.append(
                {
                    "id": f"pos-{len(cases)}",
                    "question": question,
                    "ground_truth": gt,
                    "model_answer": gt,
                    "expected": 1,
                }
            )
            cases.append(
                {
                    "id": f"neg-{len(cases)}",
                    "question": question,
                    "ground_truth": gt,
                    "model_answer": f"{gt} 999",
                    "expected": 0,
                }
            )
    return cases


# ---------------------------------------------------------------------------
# Scale orchestration (same recording contract as e2e_genrm_scale.py)
# ---------------------------------------------------------------------------
def run_scale_op(ev: "Evidence", direction: str, target: int, expect_terminal: str, timeout_s: float) -> dict:
    body = http_post(f"/{direction}", {"num_replicas": target, "timeout_secs": timeout_s})
    ev.log(f"{direction}_submitted", request_id=body.get("request_id"), status=body.get("status"))
    if body.get("status") == "NOOP":
        raise RuntimeError(f"{direction} unexpectedly NOOP: {body}")
    rid = body["request_id"]

    seen: list = []
    deadline = time.time() + timeout_s + 300
    final = None
    last = None
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
            print(f"    waiting for serve replica: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(5.0)
    raise RuntimeError(f"engines did not reach {expected} within {timeout_s}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--genrm-num-gpus", type=int, default=1, help="initial engine count")
    parser.add_argument("--num-pairs", type=int, default=25, help="dataset rows -> 2x cases (pos+neg)")
    parser.add_argument("--repeats-per-input", type=int, default=8, help="requests per input per arm (round-robin)")
    parser.add_argument("--scale-out-timeout", type=float, default=900.0)
    parser.add_argument("--scale-in-timeout", type=float, default=900.0)
    args_cli = parser.parse_args()

    out_dir = args_cli.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", f"reward_consistency_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    ev = Evidence(out_dir)
    ev.log("e2e_start", model_path=args_cli.model_path, dataset=args_cli.dataset, out_dir=out_dir)

    import ray
    from ray import serve
    from ray.util.placement_group import placement_group_table, remove_placement_group

    from relax.components.genrm import GenRM
    from relax.core.service import create_placement_group
    from relax.utils.utils import get_serve_url

    # Cleanup contract: identical to the other task4 drivers.
    verdicts: dict = {}
    replies: list = []
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
            # Official judge sampling (run-qwen3-4B-8xgpu-genrm.sh), with
            # thinking disabled: a hybrid-thinking judge would spend the
            # budget inside <think> and the verdict would be truncated.
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

        # Scale out; the elastic engine is whoever is new.
        run_scale_op(ev, "scale_out", cfg.genrm_num_gpus + 1, "ACTIVE", args_cli.scale_out_timeout)
        after_out = wait_for_engines(cfg.genrm_num_gpus + 1, timeout_s=60)
        ids_after = engine_ids(after_out)
        elastic_ids = ids_after - initial_ids
        if len(elastic_ids) != 1:
            raise RuntimeError(f"expected exactly 1 elastic engine, got {sorted(map(list, elastic_ids))}")
        elastic_id = next(iter(elastic_ids))
        ev.log("phase1_engines", initial=sorted(map(list, initial_ids)), elastic=list(elastic_id))

        cases = load_inputs(args_cli.dataset, args_cli.num_pairs)
        if not cases:
            raise RuntimeError(f"no usable inputs from {args_cli.dataset}")
        ev.log("inputs_loaded", cases=len(cases), positives=sum(1 for c in cases if c["expected"] == 1))

        arms = {
            # Primary criterion: greedy — deterministic per engine; any
            # cross-engine verdict difference is real divergence.
            "greedy": {"temperature": 0.0},
            # Official sampling config; per-engine seeds differ
            # (args.seed + rank), so this arm quantifies sampling stability.
            "official": None,
        }

        for arm, params in arms.items():
            for case in cases:
                messages = _format_messages(case["question"], case["ground_truth"], case["model_answer"])
                for _ in range(args_cli.repeats_per_input):
                    body = {"messages": messages}
                    if params is not None:
                        body["sampling_params"] = params
                    t = time.time()
                    out = http_post("/generate", body)
                    replies.append(
                        {
                            "arm": arm,
                            "case": case["id"],
                            "expected": case["expected"],
                            "engine": f"{out.get('engine_host')}:{out.get('engine_port')}",
                            "finish_reason": out.get("finish_reason"),
                            "completion_tokens": out.get("completion_tokens"),
                            "text": out.get("response", ""),
                            "strict": strict_parse(out.get("response", "")),
                            "loose": loose_parse(out.get("response", "")),
                            "latency": round(time.time() - t, 3),
                        }
                    )
            ev.log(f"arm_{arm}_done", replies=sum(1 for r in replies if r["arm"] == arm))

        with open(os.path.join(out_dir, "replies.json"), "w") as f:
            json.dump(replies, f, indent=2, ensure_ascii=False)

        # ------------------------------------------------------------------ #
        # Analysis: attribution coverage, completeness, parse gate, agreement
        # ------------------------------------------------------------------ #
        initial_engine = f"{list(initial_ids)[0][0]}:{list(initial_ids)[0][1]}"
        elastic_engine = f"{elastic_id[0]}:{elastic_id[1]}"

        def engine_verdicts(arm, case_id, engine):
            vals = {
                r["strict"]
                for r in replies
                if r["arm"] == arm and r["case"] == case_id and r["engine"] == engine and r["strict"] is not None
            }
            return vals

        attribution_failures = []
        truncated = []
        parse_gate_failures = []
        strict_mismatches = []
        loose_mismatches = []
        correctness = {"initial": {1: 0, 0: 0}, "elastic": {1: 0, 0: 0}}

        for arm in arms:
            for case in cases:
                per_engine = {}
                for label, engine in (("initial", initial_engine), ("elastic", elastic_engine)):
                    arm_replies = [
                        r for r in replies if r["arm"] == arm and r["case"] == case["id"] and r["engine"] == engine
                    ]
                    if not arm_replies:
                        attribution_failures.append({"arm": arm, "case": case["id"], "engine": label})
                        per_engine[label] = None
                        continue
                    vals = engine_verdicts(arm, case["id"], engine)
                    if not vals:
                        parse_gate_failures.append({"arm": arm, "case": case["id"], "engine": label})
                        per_engine[label] = None
                        continue
                    per_engine[label] = vals.pop() if len(vals) == 1 else "mixed"
                    if per_engine[label] == case["expected"]:
                        correctness[label][case["expected"]] += 1
                if per_engine.get("initial") is not None and per_engine.get("elastic") is not None:
                    if per_engine["initial"] != per_engine["elastic"]:
                        if arm == "greedy":
                            strict_mismatches.append({"case": case["id"], **per_engine})
                        else:
                            loose_mismatches.append({"case": case["id"], **per_engine})

        truncated = [r for r in replies if r["finish_reason"] not in (None, "stop")]
        incomplete = [r for r in replies if not (r.get("text") or "").strip()]

        n_cases = len(cases)
        verdicts = {
            "attribution_covers_both_engines": not attribution_failures,
            "attribution_failures": attribution_failures[:10],
            "all_replies_complete": not incomplete,
            "no_truncated_replies": not truncated,
            "truncated_count": len(truncated),
            "parse_gate_independent": not parse_gate_failures,
            "parse_gate_failures": parse_gate_failures[:10],
            # Primary acceptance: deterministic (greedy) scoring must be
            # identical across engines for every input.
            "greedy_verdicts_identical_across_engines": not strict_mismatches,
            "greedy_mismatches": strict_mismatches[:10],
            # Informational: the official sampling config (temperature 0.1)
            # is stochastic per engine -- engines run different seeds
            # (args.seed + rank in sglang_engine), so a verdict can flip
            # across repeats on ambiguous inputs. This characterizes the
            # deployed config, not scaling correctness; the rate and the
            # affected cases are reported, not gated.
            "official_sampling_instability_rate": (
                len(loose_mismatches) / n_cases if loose_mismatches is not None else None
            ),
            "official_sampling_unstable_cases": loose_mismatches[:10],
            "judge_correctness_initial": correctness["initial"],
            "judge_correctness_elastic": correctness["elastic"],
            "cases": n_cases,
            "replies_total": len(replies),
            "initial_engine": initial_engine,
            "elastic_engine": elastic_engine,
            # A 0.6B judge may misjudge; consistency is the acceptance
            # criterion, correctness is reported per engine.
            "judge_correctness_is_informational": True,
        }
        verdicts["E2E_PASS"] = all(
            v for k, v in verdicts.items() if isinstance(v, bool) and k not in ("judge_correctness_is_informational",)
        )
        ev.log("verdicts", **{k: v for k, v in verdicts.items() if k != "judge_correctness_is_informational"})

        # Scale back in (the elastic engine must drain and release cleanly).
        run_scale_op(ev, "scale_in", cfg.genrm_num_gpus, "COMPLETED", args_cli.scale_in_timeout)
        after_in = wait_for_engines(cfg.genrm_num_gpus, timeout_s=60)
        verdicts["scale_in_removed_exactly_elastic"] = engine_ids(after_in) == initial_ids
        verdicts["E2E_PASS"] = verdicts["E2E_PASS"] and verdicts["scale_in_removed_exactly_elastic"]
        ev.log("phase2_engines", ids=sorted(map(list, engine_ids(after_in))))
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
