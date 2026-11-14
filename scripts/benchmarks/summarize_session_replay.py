# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Summarize the fixed 32-arm replay study from raw metadata, without rerunning
it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any


LENGTHS = ("short", "long")
CONCURRENCIES = (1, 4)
ARMS = ("baseline", "candidate")
PAIRS = 4
EXPORTS = 64


def percentile(values: Sequence[float], q: float = 99) -> float:
    ordered = sorted(values)
    if not ordered or not all(math.isfinite(value) for value in ordered):
        raise ValueError("Missing or nonfinite percentile input")
    position = (len(ordered) - 1) * q / 100
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def near(actual: float, expected: float, label: str) -> None:
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"{label} does not reproduce its raw observations: {actual} != {expected}")


def describe(values: list[float]) -> dict[str, Any]:
    return dict(
        values=values,
        mean=statistics.mean(values),
        sample_std=statistics.stdev(values),
        minimum=min(values),
        maximum=max(values),
        scope="Descriptive within-VM repeated-process/order sensitivity; no population CI",
    )


def inspect_arm(row: dict[str, Any], length: str, concurrency: int, pair: int) -> dict[str, Any]:
    if row["length"] != length or row["concurrency"] != concurrency or row["order_seed"] != pair:
        raise ValueError("Raw arm parameters differ from the fixed matrix")
    if row["formal_evaluation"] or not row["cleanup_complete"] or row["mock_processes_are_os_processes"]:
        raise ValueError("Unexpected replay scope or incomplete owned cleanup")
    cases = row["selection"]["cases"]
    if len(cases) != EXPORTS or len({case["source_row"] for case in cases}) != EXPORTS:
        raise ValueError("The arm does not contain the declared64 distinct source records")
    by_session = {f"replay-{index}": case for index, case in enumerate(cases)}
    all_sessions = set(by_session)
    if row["declared_submission_session_order"] != list(by_session):
        raise ValueError("Declared submission order is inconsistent with the actual case-to-session map")
    if set(row["training_sha256_by_session"]) != all_sessions:
        raise ValueError("Missing or extra completed training outputs")
    if len(row["export_session_order"]) != EXPORTS or set(row["export_session_order"]) != all_sessions:
        raise ValueError("Recorded production builder entries do not account for every declared export")
    expected_calls = sum(case["assistant_turns"] for case in cases)
    for case in cases:
        if (
            len(case["response_tokens"]) != case["assistant_turns"]
            or sum(case["response_tokens"]) != case["total_response_tokens"]
        ):
            raise ValueError("Recorded turn/token accounting differs")
    if (
        row["completed_exports"] != EXPORTS
        or row["resident_sessions_at_start"] != EXPORTS
        or row["total_backend_calls"] != expected_calls
        or row["timed_backend_calls"] != EXPORTS
        or row["prepare_backend_calls"] != expected_calls - EXPORTS
    ):
        raise ValueError("Dropped work or incorrect backend phase accounting")
    if (
        row["peak_clients"] != concurrency
        or row["peak_backend_inflight"] != concurrency
        or row["peak_sync_builder_active"] != 1
    ):
        raise ValueError("The requested concurrency did not become effective")
    cohort_backend = []
    cohort_ready = []
    if len(row["cohorts"]) != EXPORTS // concurrency:
        raise ValueError("Unexpected cohort count")
    for index, cohort in enumerate(row["cohorts"]):
        expected_sessions = set(list(by_session)[index * concurrency : (index + 1) * concurrency])
        expected_rows = {by_session[sid]["source_row"] for sid in expected_sessions}
        if (
            cohort["size"] != concurrency
            or cohort["backend_arrivals"] != concurrency
            or cohort["export_ready_arrivals"] != concurrency
            or len(cohort["backend_session_order"]) != concurrency
            or set(cohort["backend_session_order"]) != expected_sessions
            or len(cohort["export_ready_source_rows"]) != concurrency
            or set(cohort["export_ready_source_rows"]) != expected_rows
        ):
            raise ValueError("Cohort source ownership or arrival accounting mismatch")
        cohort_backend.extend(cohort["backend_session_order"])
        cohort_ready.extend(cohort["export_ready_source_rows"])
    if len(set(cohort_backend)) != EXPORTS or len(set(cohort_ready)) != EXPORTS:
        raise ValueError("A source request was duplicated across cohorts")
    case_set = sorted(cases, key=lambda value: value["source_row"])
    expected_set = hashlib.sha256(json.dumps(case_set, sort_keys=True).encode()).hexdigest()
    if row["selection"]["case_set_sha256"] != expected_set or row["selection"]["order_seed"] != pair:
        raise ValueError("Source-set/order identity mismatch")
    source = row["selection"]["source"]
    if (
        source["repo"] != "SWE-Gym/OpenHands-SFT-Trajectories"
        or source["revision"] != "4aaa5a4a4b5861f4799d2336908760c190ac3b17"
        or source["sha256"] != "ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1"
    ):
        raise ValueError("Unexpected actual replay corpus")
    tokenizer = row["tokenizer_source"]
    if (
        tokenizer["id"] != "Qwen/Qwen3-0.6B"
        or tokenizer["revision"]
        != "c1899de289a04d12100db370d81485cdf75e47ca"  # gitleaks:allow — public pinned HF revision
        or not tokenizer["files"]
    ):
        raise ValueError("Unexpected reconstructed-token protocol")
    warmup = row["complete_chain_warmup"]
    warmup_case = min(cases, key=lambda value: value["source_row"])
    if (
        warmup["source_row"] != warmup_case["source_row"]
        or warmup["completed_exports"] != 1
        or warmup["backend_calls"] != warmup_case["assistant_turns"]
        or not warmup["cleanup_complete"]
    ):
        raise ValueError("Whole-chain warmup was not complete or changed with arrival order")
    clients = row["client_seconds"]
    builders = row["builder_seconds"]
    if len(clients) != EXPORTS or len(builders) != EXPORTS or min(clients) <= 0 or min(builders) < 0:
        raise ValueError("Missing per-export timing observations")
    near(row["client_p99_seconds"], percentile(clients), "client P99")
    near(row["builder_p99_seconds"], percentile(builders), "builder P99")
    probes = row["probes"]
    window = row["measurement_window"]
    start, end, interval = window["start"], window["end"], window["interval_seconds"]
    if end <= start or interval != 0.005 or len(probes) != row["probe_count"] or not probes:
        raise ValueError("Invalid or missing event-loop measurement window")
    tolerance = 1e-6
    if probes[0]["due"] < start - tolerance or probes[0]["due"] - start > interval + tolerance:
        raise ValueError("The first nominal short request was omitted or predates the window")
    if probes[-1]["due"] > end + tolerance or end - probes[-1]["due"] > interval + tolerance:
        raise ValueError("Nominal short requests were not drained through the full window")
    for index, probe in enumerate(probes):
        if (
            not all(math.isfinite(probe[key]) for key in ("due", "start", "end"))
            or probe["end"] < probe["start"]
            or probe["start"] < probe["due"] - tolerance
        ):
            raise ValueError("Invalid raw short-request clock observations")
        if index and not math.isclose(probe["due"] - probes[index - 1]["due"], interval, abs_tol=tolerance):
            raise ValueError("A nominal short-request arrival was skipped")
    near(row["loop_p99_seconds"], percentile([p["start"] - p["due"] for p in probes]), "loop P99")
    near(row["short_request_p99_seconds"], percentile([p["end"] - p["due"] for p in probes]), "short health P99")
    near(row["exports_per_second"], EXPORTS / row["wall_seconds"], "export throughput")
    if not math.isclose(row["wall_seconds"], end - start, rel_tol=0.001, abs_tol=0.001):
        raise ValueError("Reported export window does not match raw window clocks")
    for key in ("wall_seconds", "cpu_seconds", "peak_rss_kib_including_imports_and_preparation"):
        if not math.isfinite(row[key]) or row[key] <= 0:
            raise ValueError("Invalid resource observation")
    return dict(
        case_set_sha256=expected_set,
        training_by_source_row={
            str(case["source_row"]): row["training_sha256_by_session"][sid] for sid, case in by_session.items()
        },
        warmup=dict(
            source_row=warmup["source_row"],
            backend_calls=warmup["backend_calls"],
            training_sha256=warmup["training_sha256"],
        ),
        tokenizer=tokenizer,
        source=source,
        main_backend_calls=expected_calls,
        probe_count=len(probes),
        empirical_p99_small_sample=len(probes) < 100,
    )


def collect(directory: Path) -> dict[str, Any]:
    workload_rows = {}
    references = {}
    totals = dict(main_exports=0, warmup_exports=0, main_backend_calls=0, warmup_backend_calls=0)
    for length in LENGTHS:
        for concurrency in CONCURRENCIES:
            pairs = []
            for pair in range(PAIRS):
                arms = {}
                audits = {}
                for arm in ARMS:
                    path = directory / f"arm-{pair:02d}-{length}-c{concurrency}-{arm}.json"
                    row = json.loads(path.read_text(encoding="utf-8-sig"))
                    audit = inspect_arm(row, length, concurrency, pair)
                    arms[arm] = row
                    audits[arm] = audit
                    totals["main_exports"] += row["completed_exports"]
                    totals["warmup_exports"] += row["complete_chain_warmup"]["completed_exports"]
                    totals["main_backend_calls"] += audit["main_backend_calls"]
                    totals["warmup_backend_calls"] += audit["warmup"]["backend_calls"]
                    reference = references.setdefault(length, audit)
                    # Across seeds replay-i names refer to other rows: compare by
                    # immutable source-row identity, never by the shifted session key.
                    for key in ("case_set_sha256", "training_by_source_row", "warmup", "tokenizer", "source"):
                        if audit[key] != reference[key]:
                            raise ValueError(
                                f"{length} source/training identity changed across order seeds or concurrency"
                            )
                a, b = arms["baseline"], arms["candidate"]
                if (
                    a["selection"] != b["selection"]
                    or a["training_sha256_by_session"] != b["training_sha256_by_session"]
                ):
                    raise ValueError("Paired arms differ in their actual declared work/training outputs")
                if a["loop_p99_seconds"] <= 0:
                    raise ValueError("Zero baseline loop P99 cannot define a relative reduction")
                pairs.append(
                    dict(
                        pair=pair,
                        seed=pair,
                        planned_order=list(ARMS) if pair % 2 == 0 else list(reversed(ARMS)),
                        baseline=a,
                        candidate=b,
                        loop_ratio=b["loop_p99_seconds"] / a["loop_p99_seconds"],
                        small_probe_sample=audits["baseline"]["empirical_p99_small_sample"]
                        or audits["candidate"]["empirical_p99_small_sample"],
                    )
                )
            ratios = [row["loop_ratio"] for row in pairs]
            metrics = {
                arm: {
                    key: describe([row[arm][key] for row in pairs])
                    for key in (
                        "loop_p99_seconds",
                        "short_request_p99_seconds",
                        "client_p99_seconds",
                        "builder_p99_seconds",
                        "wall_seconds",
                        "cpu_seconds",
                        "exports_per_second",
                        "peak_rss_kib_including_imports_and_preparation",
                    )
                }
                for arm in ARMS
            }
            workload_rows[f"{length}-c{concurrency}"] = dict(
                pairs=pairs,
                loop_ratios=describe(ratios),
                mean_paired_loop_p99_reduction=1 - statistics.mean(ratios),
                metrics=metrics,
                all_four_pairs_at_least_half=all(value <= 0.5 for value in ratios),
                any_small_empirical_probe_sample=any(row["small_probe_sample"] for row in pairs),
            )
    if totals["main_exports"] != 2048 or totals["warmup_exports"] != 32:
        raise ValueError("The fixed study did not complete2048main plus32warmup exports")
    primary = workload_rows["long-c4"]
    return dict(
        workloads=workload_rows,
        totals=totals,
        all_input_training_and_completion_gates=True,
        primary="long-c4 mean of paired candidate/baseline loop-P99 ratios",
        primary_reduction=primary["mean_paired_loop_p99_reduction"],
        numerical_50_percent_threshold_met=primary["mean_paired_loop_p99_reduction"] >= 0.5,
        independent_methodology_review_still_required=True,
        interpretation="Finite empirical P99 with all raw probes. Within-VM repeated-process/order spread is descriptive; no population CI. Source/helper/runner/environment/watchdog gates require separate frozen receipts. This collector is not whole-task acceptance.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.directory)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
