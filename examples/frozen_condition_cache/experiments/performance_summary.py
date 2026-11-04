"""Aggregate the preregistered 3 workloads x 3 A/B pairs, retaining every
run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from experiments.evidence import atomic_json


def cumulative_payback(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Report observed training-loop crossings from all80 recorded updates.

    Initialization is reported separately. The warmup extrapolation is labelled
    projected, never substituted for an observed cumulative crossing.
    """
    timelines = [row.get("resources_and_cache", []) for row in (a, b)]
    if any(len(rows) != 80 for rows in timelines):
        raise ValueError("Observed payback requires exactly80 cold + warm + steady update records per path")
    cumulative_a = cumulative_b = 0.0
    wins = []
    for ar, br in zip(*timelines):
        seconds = [ar["wall_seconds"], br["wall_seconds"]]
        if any(not math.isfinite(value) or value < 0 for value in seconds):
            raise ValueError("Update wall_seconds must be finite and nonnegative")
        cumulative_a += seconds[0]
        cumulative_b += seconds[1]
        wins.append(cumulative_b < cumulative_a)
    crossing = next((index + 1 for index, win in enumerate(wins) if win), None)
    sustained = next((index + 1 for index in range(len(wins)) if all(wins[index:])), None)
    saving = (a["measured_seconds"] - b["measured_seconds"]) / 64
    warmup_a = a.get("warmup_seconds", sum(row["wall_seconds"] for row in timelines[0][:16]))
    warmup_b = b.get("warmup_seconds", sum(row["wall_seconds"] for row in timelines[1][:16]))
    return {
        "cold_training_cumulative_online_seconds": cumulative_a,
        "cold_training_cumulative_cache_seconds": cumulative_b,
        "first_observed_cumulative_crossing_update": crossing,
        "crossing_is_sustained_to_end": crossing is not None and all(wins[crossing - 1 :]),
        "first_observed_sustained_crossing_update": sustained,
        "payback_scope": "actual80 training-loop update wall times; initialization excluded and reported separately",
        "projected_payback_updates_from_warmup": max(0, warmup_b - warmup_a) / saving if saving > 0 else None,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 18:
        raise ValueError("Exactly 18 recorded segments are required; no best-of filtering")
    indexed = {(r["workload"], r["pair"], r["variant"]): r for r in records}
    workloads = sorted({r["workload"] for r in records})
    if len(indexed) != 18 or len(workloads) != 3:
        raise ValueError("Expected three unique workloads, each three online/cache pairs")
    output = {}
    for workload in workloads:
        pairs = []
        for pair in (0, 1, 2):
            a = indexed[(workload, pair, "online")]["result"]
            b = indexed[(workload, pair, "cache")]["result"]
            if a["native_contract"] != b["native_contract"]:
                raise ValueError("A/B native data/model/optimizer/precision contracts differ")
            for row in (a, b):
                if row["status"] != "completed" or row["warmup_updates"] != 16 or row["measured_updates"] != 64:
                    raise ValueError("Incomplete or changed fixed timing segment")
            if a["cache_budget_per_rank"] != 0 or b["cache_budget_per_rank"] <= 0:
                raise ValueError("Expected fixed online A and bounded-cache B")
            ratio = a["measured_seconds"] / b["measured_seconds"]
            pairs.append(
                {
                    "pair": pair,
                    "steady_speedup": ratio,
                    "throughput_gain_percent": (ratio - 1) * 100,
                    "online_seconds": a["measured_seconds"],
                    "cache_seconds": b["measured_seconds"],
                    **cumulative_payback(a, b),
                    "online_initialization_plus_total_seconds": a["initialization_plus_total_seconds"],
                    "cache_initialization_plus_total_seconds": b["initialization_plus_total_seconds"],
                    "final_cache_stats": b["resources_and_cache"][-1]["cache"],
                }
            )
        ratios = [p["steady_speedup"] for p in pairs]
        median = statistics.median(ratios)
        output[workload] = {
            "pairs": pairs,
            "median_speedup": median,
            "minimum_speedup": min(ratios),
            "maximum_speedup": max(ratios),
            "median_at_least_5_percent": median >= 1.05,
            "median_at_least_10_percent": median >= 1.10,
            "all_pairs_at_least_5_percent": min(ratios) >= 1.05,
        }
    return {
        "schema": 1,
        "completed_segments": 18,
        "workloads": output,
        "selection": "all preregistered runs; no outlier or failed-run deletion",
        "scope": "native SFT; cold initialization reported separately; Ray/TQ excluded",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="JSON list of {workload,pair:0..2,variant:online|cache,run_dir}",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.runs.read_text())
    for record in records:
        record["result"] = json.loads((Path(record["run_dir"]) / "result.json").read_text())
    atomic_json(args.output, summarize(records))


if __name__ == "__main__":
    main()
