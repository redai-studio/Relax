# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Aggregate real loop throughput from complete Ray training logs.

Token rate is sum(real unpadded input tokens) / sum(train_wait + train).
Supervised-token rate is reported separately. Both exclude dummy/padding rows.
Never average individual per-step token rates.
"""

import argparse
import ast
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


def read_window(log: str, warmup: int, steps: int) -> dict[str, Any]:
    if warmup < 0 or steps <= 0:
        raise ValueError("Require nonnegative warmup and positive measured steps")
    pattern = re.compile(r"\bperf (\d+): (\{[^\n]*\})")
    records = {}
    for match in pattern.finditer(log):
        step = int(match[1])
        if step in records:
            raise ValueError(f"Duplicate perf record for rollout {step}; do not concatenate replica logs")
        records[step] = ast.literal_eval(match[2])
    selected = []
    for step in range(warmup, warmup + steps):
        if step not in records:
            raise ValueError(f"Missing measured rollout {step}")
        record = records[step]
        seconds, tokens, supervised_rate = [
            float(record[key])
            for key in (
                "perf/step_time",
                "perf/actor_train_tokens",
                "perf/step_resp_token_per_s",
            )
        ]
        if not all(math.isfinite(value) and value > 0 for value in (seconds, tokens, supervised_rate)):
            raise ValueError("Timing/token measurements must be finite and positive")
        supervised = supervised_rate * seconds
        if not math.isclose(tokens, round(tokens), abs_tol=1e-5) or not math.isclose(
            supervised, round(supervised), rel_tol=0, abs_tol=1e-4
        ):
            raise ValueError("Expected integer real token counts")
        if supervised > tokens + 1e-4:
            raise ValueError("Supervised tokens exceed real input tokens")
        selected.append(dict(rollout=step, seconds=seconds, tokens=round(tokens), supervised_tokens=round(supervised)))
    total_time = sum(row["seconds"] for row in selected)
    return dict(
        records=selected,
        seconds=total_time,
        tokens=sum(row["tokens"] for row in selected),
        input_tokens_per_s=sum(row["tokens"] for row in selected) / total_time,
        supervised_tokens_per_s=sum(row["supervised_tokens"] for row in selected) / total_time,
    )


def summarize(directories: list[Path], *, warmup: int, steps: int, minimum_runs: int = 3) -> dict[str, Any]:
    if minimum_runs < 3:
        raise ValueError("At least three independent repetitions are required")
    groups: dict[str, list[dict[str, Any]]] = {}
    invariant = None
    coverage = None
    if len({path.resolve() for path in directories}) != len(directories):
        raise ValueError("The same run directory cannot count as independent repetitions")
    for directory in directories:
        metadata = json.loads((directory / "run.json").read_text())
        status = json.loads((directory / "status.json").read_text())
        if status["status"] != "SUCCEEDED" or metadata["capture"] or metadata["dry_run"]:
            raise ValueError("Only successful, non-capture, actual training runs may enter throughput comparison")
        current = {
            key: metadata[key]
            for key in (
                "invariant_command",
                "versions",
                "launch_environment",
                "source_sha256",
                "workload_sha256",
                "model_revision",
                "data_revision",
                "native_source_sha256",
                "transfer_queue_source_sha256",
                "devices",
            )
        }
        if invariant is not None and current != invariant:
            raise ValueError("Model, data, source, environment or non-scheduling CLI changed between runs")
        invariant = current
        window = read_window((directory / "training.log").read_text(), warmup, steps)
        current_coverage = [(row["rollout"], row["tokens"], row["supervised_tokens"]) for row in window["records"]]
        if coverage is not None and current_coverage != coverage:
            raise ValueError("The compared windows consumed different real input/supervised token counts")
        coverage = current_coverage
        groups.setdefault(metadata["mode"], []).append({"directory": str(directory), **window})
    if "A" not in groups or len(groups) < 2 or any(len(runs) < minimum_runs for runs in groups.values()):
        raise ValueError("Need baseline A and a candidate, each with at least three independent runs")
    modes = {}
    for mode, runs in groups.items():
        rates = [run["input_tokens_per_s"] for run in runs]
        modes[mode] = dict(
            runs=runs,
            median_input_tokens_per_s=statistics.median(rates),
            min_input_tokens_per_s=min(rates),
            max_input_tokens_per_s=max(rates),
            stdev_input_tokens_per_s=statistics.stdev(rates),
        )
    baseline = modes["A"]["median_input_tokens_per_s"]
    for mode in modes.values():
        ratio = mode["median_input_tokens_per_s"] / baseline
        mode.update(gain_percent=100 * (ratio - 1), reaches_10_percent=ratio >= 1.10, reaches_20_percent=ratio >= 1.20)
    return dict(
        warmup=warmup,
        measured_steps=steps,
        invariant=invariant,
        modes=modes,
        scope="Training-loop input tokens / (train_wait + train); startup/checkpoint time excluded",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.runs, warmup=args.warmup, steps=args.steps)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                mode: {key: value for key, value in result.items() if key != "runs"}
                for mode, result in report["modes"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
