# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Recompute every published H200 performance pair using Python's standard
library."""

from __future__ import annotations

import gzip
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    """Read an ordinary JSON or a deterministic gzip copy."""
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def recompute(root: Path) -> dict[str, Any]:
    """Recompute all three A/B pairs for each workload, including
    regressions."""
    indexed = {}
    for path in sorted((root / "performance").glob("*.json*")):
        match = re.fullmatch(r"(.+)-pair([123])-([ab])\.json(?:\.gz)?", path.name)
        if match is None:
            raise ValueError(f"Unexpected performance filename: {path.name}")
        workload, pair, variant = match.groups()
        key = workload, int(pair), variant
        if key in indexed:
            raise ValueError(f"Duplicate performance record: {key}")
        indexed[key] = read_json(path)
    workloads = sorted({key[0] for key in indexed})
    if len(indexed) != 18 or len(workloads) != 3:
        raise ValueError("All18 segments from three workloads are required")
    output = {}
    for workload in workloads:
        pairs = []
        for pair in (1, 2, 3):
            a, b = (indexed[(workload, pair, variant)] for variant in ("a", "b"))
            if a["native_contract"] != b["native_contract"]:
                raise ValueError("A/B native contracts differ")
            if a["sample_id_occurrences"] != b["sample_id_occurrences"]:
                raise ValueError("A/B sample orders differ")
            for row in (a, b):
                if row["status"] != "completed" or row["warmup_updates"] != 16 or row["measured_updates"] != 64:
                    raise ValueError("Incomplete or changed timing window")
                if len(row["resources_and_cache"]) != 80:
                    raise ValueError("Expected all80 cold/warm/steady update times")
            cumulative_a = cumulative_b = 0.0
            wins = []
            for ar, br in zip(a["resources_and_cache"], b["resources_and_cache"]):
                cumulative_a += ar["wall_seconds"]
                cumulative_b += br["wall_seconds"]
                wins.append(cumulative_b < cumulative_a)
            crossing = next((index + 1 for index, win in enumerate(wins) if win), None)
            pairs.append(
                {
                    "pair": pair,
                    "online_seconds": a["measured_seconds"],
                    "cached_seconds": b["measured_seconds"],
                    "gain": a["measured_seconds"] / b["measured_seconds"] - 1,
                    "cumulative_online_seconds": cumulative_a,
                    "cumulative_cached_seconds": cumulative_b,
                    "first_observed_crossing_update": crossing,
                    "crossing_sustained_to_end": crossing is not None and all(wins[crossing - 1 :]),
                }
            )
        gains = [pair["gain"] for pair in pairs]
        output[workload] = {
            "pairs": pairs,
            "mean_gain": statistics.mean(gains),
            "median_gain": statistics.median(gains),
            "min_gain": min(gains),
            "max_gain": max(gains),
            "sample_stddev_gain": statistics.stdev(gains),
        }
    return {
        "performance": output,
        "stored_numerical_comparison_status": {
            model: read_json(root / "numerical" / f"{model}.json.gz")["strict_acceptance"]
            for model in ("sensenova", "qwen_edit")
        },
    }


if __name__ == "__main__":
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    sys.stdout.write(json.dumps(recompute(directory), indent=2) + "\n")
