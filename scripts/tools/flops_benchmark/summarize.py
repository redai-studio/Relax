# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Summarize paired runs using supervised tokens over complete training-loop
time."""

import argparse
import json
import math
import statistics
from pathlib import Path

from relax.utils.logging_utils import get_logger
from scripts.tools.flops_benchmark.common import file_digest, read_metrics, write_json


logger = get_logger(__name__)


def summarize_run(manifest: dict, log: Path, *, start: int, steps: int, warmup: int) -> dict:
    if not 0 <= warmup < steps or start < 0 or start + steps > len(manifest["batches"]):
        raise ValueError("Invalid measurement interval")
    perf, training = read_metrics(log, "perf"), read_metrics(log, "train")
    rows = []
    for step in range(start, start + steps):
        data, train = perf[step], training[step]
        elapsed = float(data["perf/step_time"])
        supervised_rate = float(data["perf/step_resp_token_per_s"])
        expected = manifest["batches"][step]
        if not math.isfinite(elapsed) or elapsed <= 0 or not math.isfinite(supervised_rate):
            raise ValueError(f"Non-finite or non-positive timing at step {step}")
        if not math.isfinite(train["train/loss"]) or train["train/global_batch_size"] != manifest["batch_size"]:
            raise ValueError(f"Loss or effective batch differs at step {step}")
        if not math.isclose(elapsed * supervised_rate, expected["effective_tokens"], rel_tol=1e-8, abs_tol=1e-4):
            raise ValueError(f"Supervised token count differs from frozen data at step {step}")
        if data["perf/actor_train_tokens"] != expected["input_tokens"]:
            raise ValueError(f"Input token count differs from frozen data at step {step}")
        if step >= start + warmup:
            rows.append(
                {
                    "step": step,
                    "seconds": elapsed,
                    "effective_tokens": expected["effective_tokens"],
                    "input_tokens": expected["input_tokens"],
                    "loss": train["train/loss"],
                }
            )
    seconds = math.fsum(row["seconds"] for row in rows)
    supervised = sum(row["effective_tokens"] for row in rows)
    inputs = sum(row["input_tokens"] for row in rows)
    return {
        "log": str(log),
        "log_sha256": file_digest(log),
        "warmup": warmup,
        "measured_steps": rows,
        "seconds": seconds,
        "effective_tokens": supervised,
        "input_tokens": inputs,
        "effective_token_per_s": supervised / seconds,
        "input_token_per_s": inputs / seconds,
    }


def summarize(manifest_path: Path, runs_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    runs = json.loads(runs_path.read_text())
    results = []
    pairs: dict[int, dict] = {}
    for run in runs:
        if run["status"] != "SUCCEEDED":
            raise ValueError("All compared jobs must finish successfully")
        result = summarize_run(
            manifest, Path(run["log"]), start=run["start_step"], steps=run["steps"], warmup=run["warmup"]
        )
        result.update(arm=run["arm"], pair=run["pair"], configuration_id=run["configuration_id"])
        results.append(result)
        if run["arm"] in {"baseline", "flops"}:
            pair = pairs.setdefault(run["pair"], {})
            if run["arm"] in pair:
                raise ValueError("Duplicate experiment arm in a pair")
            pair[run["arm"]] = result
    gains = []
    for pair in pairs.values():
        if set(pair) != {"baseline", "flops"}:
            raise ValueError("Incomplete baseline/FLOPs experiment pair")
        a, b = pair["baseline"], pair["flops"]
        if a["configuration_id"] != b["configuration_id"] or a["effective_tokens"] != b["effective_tokens"]:
            raise ValueError("Paired runs have different configurations or effective tokens")
        if [r["step"] for r in a["measured_steps"]] != [r["step"] for r in b["measured_steps"]]:
            raise ValueError("Paired runs have different measurement intervals")
        gains.append(b["effective_token_per_s"] / a["effective_token_per_s"] - 1)
    if not gains:
        raise ValueError("No complete performance pairs")
    return {
        "version": 1,
        "manifest_sha256": file_digest(manifest_path),
        "runs": results,
        "paired_speedups": gains,
        "median_speedup": statistics.median(gains),
        "min_speedup": min(gains),
        "max_speedup": max(gains),
        "meets_10_percent": statistics.median(gains) >= 0.10,
        "meets_20_percent": statistics.median(gains) >= 0.20,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.manifest, args.runs)
    write_json(args.output, result)
    logger.info("Paired median speedup %.2f%%; report=%s", result["median_speedup"] * 100, args.output)


if __name__ == "__main__":
    main()
