# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Summarize complete SFT condition-cache runs without importing GPU libraries.

Each run contains events-rank{rank}.jsonl with init/step/close records. Step
wall_seconds includes the complete, synchronized training loop. Throughput uses
the slowest rank at each step and counts global_samples once. Encoding timings
are diagnostic only. Missing measurements remain null; incomplete or different
workloads never produce a speedup claim. Use separate --low-reuse-* runs to
report cache overhead when conditions rarely repeat. Record the exact source
commit and environment in the experiment manifest; this parser validates the
training events rather than an external source-archive format.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


COUNTERS = (
    "local_hit",
    "effective_hit",
    "miss",
    "forced_recompute",
    "encoding_calls",
    "eligible_calls",
    "repeated_h2d_bytes",
    "cache_read_bytes",
    "cache_write_bytes",
)
TIMINGS = ("condition_seconds", "encode_seconds", "cache_lookup_seconds", "cache_store_seconds")
RESOURCES = (
    "cache_bytes",
    "gpu_allocated_bytes",
    "gpu_reserved_bytes",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "cpu_rss_bytes",
    "shm_used_bytes",
    "disk_bytes",
)
NON_SEMANTIC = {
    "cache_enabled",
    "cache_max_bytes",
    "cache_dir",
    "profile",
    "run_dir",
    "run_id",
    "condition_cache",
    "condition_cache_max_bytes",
    "condition_cache_dir",
    "condition_profile",
    "condition_cache_run_dir",
    "save",
    "save_dir",
    "log_dir",
    "created_unix",
    "relax_condition_cache",
    "relax_condition_profile",
    "relax_condition_cache_max_bytes",
    "relax_condition_cache_dir",
    "relax_condition_cache_run_dir",
}


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{field} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{field} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _semantic_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _semantic_config(item) for key, item in value.items() if key.lower() not in NON_SEMANTIC}
    if isinstance(value, list):
        return [_semantic_config(item) for item in value]
    return value


def read_events(path: Path) -> list[dict[str, Any]]:
    """Read the runtime's raw JSONL events, ignoring unrelated driver logs."""
    paths = sorted(path.glob("events-rank*.jsonl")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"No event files in {path}")
    result = []
    for source in paths:
        for line_number, raw in enumerate(source.read_text().splitlines(), 1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {source}:{line_number}") from exc
            if isinstance(event, dict) and event.get("stage") in ("init", "step", "close"):
                result.append(event)
    if not result:
        raise ValueError(f"No condition-cache events in {path}")
    return result


def _window(steps: list[dict[str, Any]]) -> dict[str, Any]:
    seconds = sum(step["wall_seconds"] for step in steps)
    samples = sum(step["global_samples"] for step in steps)
    return {
        "steps": len(steps),
        "samples": samples,
        "seconds": seconds,
        "samples_per_second": samples / seconds if seconds else None,
    }


def _job_measurement(path: Path) -> tuple[str | None, float | None, bool | None]:
    source = (path if path.is_dir() else path.parent) / "job.json"
    if not source.is_file():
        return None, None, None
    try:
        job = json.loads(source.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed Ray job metadata: {source}") from exc
    if not isinstance(job, dict):
        raise ValueError(f"Ray job metadata must be an object: {source}")
    status = job.get("status")
    if status != "SUCCEEDED":
        return status, None, False
    start = _number(job.get("start_time"), "Ray job start_time")
    end = _number(job.get("end_time"), "Ray job end_time")
    if end <= start:
        raise ValueError("Succeeded Ray job must have end_time after start_time")
    return status, (end - start) / 1000, True


def summarize_run(path: Path, cold_steps: int = 16) -> dict[str, Any]:
    if cold_steps < 0:
        raise ValueError("cold_steps must be nonnegative")
    events = read_events(path)
    init, close, by_step = {}, {}, defaultdict(dict)
    issues = []
    for event in events:
        rank = _integer(event.get("rank"), "rank")
        stage = event["stage"]
        destination = (
            init if stage == "init" else close if stage == "close" else by_step[_integer(event.get("step"), "step")]
        )
        if rank in destination:
            raise ValueError(f"Duplicate {stage} event for rank {rank} in {path}")
        destination[rank] = event
    if not init:
        raise ValueError(f"Missing init events in {path}")
    first = init[min(init)]
    config = first.get("config")
    validation_lane = first.get("verification") is True or (
        isinstance(config, dict) and config.get("conditioning_verification") is True
    )
    world = _integer(first.get("world_size"), "world_size", positive=True)
    expected = set(range(world))
    if set(init) != expected:
        issues.append("missing or unexpected init ranks")
    if set(close) != expected:
        issues.append("missing or unexpected close ranks")
    signature_keys = ("world_size", "global_batch_size", "model_fingerprint", "metadata_sha256")
    signature = {key: first.get(key) for key in signature_keys}
    signature.update({key: first.get(key) for key in ("start_step", "expected_steps")})
    signature["config"] = _semantic_config(first.get("config"))
    if any(signature[key] is None for key in (*signature_keys, "config")):
        issues.append("missing comparison provenance")
    for event in init.values():
        current = {key: event.get(key) for key in signature_keys}
        current.update({key: event.get(key) for key in ("start_step", "expected_steps")})
        current["config"] = _semantic_config(event.get("config"))
        if (
            current != signature
            or event.get("mode") != first.get("mode")
            or event.get("run_id") != first.get("run_id")
        ):
            issues.append("rank configurations disagree")
            break
    steps = []
    raw_steps = []
    sample_order = []
    sample_order_known = True
    for step, ranks in sorted(by_step.items()):
        if set(ranks) != expected:
            issues.append(f"step {step} has incomplete ranks")
            continue
        samples = {_integer(row.get("global_samples"), "global_samples", positive=True) for row in ranks.values()}
        if len(samples) != 1:
            raise ValueError(f"Rank sample counts disagree at step {step} in {path}")
        if first.get("global_batch_size") is not None and samples != {first["global_batch_size"]}:
            raise ValueError(f"Effective batch changed at step {step} in {path}")
        if all(row.get("local_samples") is not None for row in ranks.values()):
            local = sum(_integer(row["local_samples"], "local_samples") for row in ranks.values())
            if local != next(iter(samples)):
                raise ValueError(f"Local samples do not sum to the global batch at step {step}")
        if all(row.get("sample_ids") is not None for row in ranks.values()):
            order = []
            for rank, row in sorted(ranks.items()):
                ids = row["sample_ids"]
                if not isinstance(ids, list):
                    raise ValueError(f"sample_ids must be a list at step {step}")
                for sample_id in ids:
                    _integer(sample_id, "sample_id")
                if not validation_lane and row.get("local_samples") is not None and len(ids) != row["local_samples"]:
                    raise ValueError(f"sample_ids length differs from local_samples at step {step}")
                order.append([rank, ids])
            sample_order.append([step, order])
        else:
            sample_order_known = False
        for row in ranks.values():
            if row.get("loss") is not None:
                if (
                    isinstance(row["loss"], bool)
                    or not isinstance(row["loss"], (int, float))
                    or not math.isfinite(row["loss"])
                ):
                    raise ValueError(f"Non-finite loss at step {step}")
        steps.append(
            {
                "step": step,
                "global_samples": next(iter(samples)),
                "wall_seconds": max(
                    _number(row.get("wall_seconds"), "wall_seconds", positive=True) for row in ranks.values()
                ),
            }
        )
        raw_steps.extend(ranks.values())
    indices = [row["step"] for row in steps]
    if indices and indices != list(range(indices[0], indices[-1] + 1)):
        issues.append("non-contiguous completed steps")
    if indices and first.get("start_step") is not None and indices[0] != first["start_step"]:
        issues.append("first completed step differs from the planned start")
    for rank, event in close.items():
        if event.get("completed_steps") != len(by_step):
            issues.append(f"rank {rank} close count differs from observed steps")
        if event.get("status") is not None and event["status"] != "complete":
            issues.append(f"rank {rank} did not complete normally")
    if first.get("expected_steps") is not None and len(by_step) != first["expected_steps"]:
        issues.append("observed steps differ from the planned run length")
    if not steps:
        issues.append("no complete training steps")
    job_status, job_seconds, job_complete = _job_measurement(path)
    if job_complete is False:
        issues.append("Ray job did not finish successfully")
    totals = {}
    coverage = {}
    for field in COUNTERS:
        values = [_number(row[field], field) for row in raw_steps if row.get(field) is not None]
        coverage[field] = {"observed": len(values), "expected": len(raw_steps)}
        totals[field] = sum(values) if len(values) == len(raw_steps) and values else None
    timings = {}
    for field in TIMINGS:
        values = [_number(row[field], field) for row in raw_steps if row.get(field) is not None]
        timings[field] = {
            "rank_step_mean_seconds": statistics.mean(values) if values else None,
            "rank_step_max_seconds": max(values) if values else None,
            "observed": len(values),
            "expected": len(raw_steps),
        }
    resources = {}
    for field in RESOURCES:
        peaks = {}
        for event in events:
            if event.get(field) is not None:
                value = _number(event[field], field)
                peaks[event["rank"]] = max(peaks.get(event["rank"], 0), value)
        resources[field] = {
            "max_observed_bytes": max(peaks.values()) if peaks else None,
            "sum_rank_peaks_bytes": sum(peaks.values()) if peaks and field.startswith(("gpu_", "cpu_")) else None,
            "observed_ranks": len(peaks),
        }

    def maximum(field: str, records: dict[int, dict[str, Any]]) -> float | None:
        if set(records) != expected or any(row.get(field) is None for row in records.values()):
            return None
        return max(_number(row[field], field) for row in records.values())

    hits, encodes = totals["effective_hit"], totals["encoding_calls"]
    resume_checkpoint = config.get("resume_checkpoint") if isinstance(config, dict) else None
    fresh_initialization = first.get("start_step") == 0 and not resume_checkpoint
    phase_counters = {}
    for name, records in (("cold", raw_steps[: cold_steps * world]), ("steady", raw_steps[cold_steps * world :])):
        counters = {}
        for field in COUNTERS:
            values = [_number(row[field], field) for row in records if row.get(field) is not None]
            counters[field] = sum(values) if values and len(values) == len(records) else None
        phase_counters[name] = counters
    return {
        "path": str(path),
        "run_id": first.get("run_id"),
        "mode": first.get("mode"),
        "validation_lane": validation_lane,
        "fresh_initialization": fresh_initialization,
        "resume_checkpoint": resume_checkpoint,
        "complete": not issues,
        "issues": sorted(set(issues)),
        "comparison_signature": signature,
        "step_samples": [[row["step"], row["global_samples"]] for row in steps],
        "sample_order": sample_order if sample_order_known and steps else None,
        "cold": _window(steps[:cold_steps]),
        "steady": _window(steps[cold_steps:]),
        "training": _window(steps),
        "time_init_seconds": maximum("time_init_seconds", init),
        "total_seconds": maximum("total_seconds", close),
        "close_seconds": maximum("close_seconds", close),
        "total_seconds_scope": "trainer initialization and training; excludes model/driver initialization",
        "job_status": job_status,
        "job_wall_seconds": job_seconds,
        "job_complete": job_complete,
        "counter_totals": totals,
        "counter_coverage": coverage,
        "phase_counter_totals": phase_counters,
        "effective_hit_rate": hits / (hits + encodes)
        if hits is not None and encodes is not None and hits + encodes
        else None,
        "effective_hit_rate_denominator": "effective_hit + encoding_calls; includes uncacheable online encodes",
        "timings": timings,
        "resources": resources,
        "resource_note": "Shared cache/shm/disk snapshots use max, never rank sums. CPU RSS sums include shared-page double counting; sum(rank GPU peaks) is an upper bound, not a synchronized peak.",
    }


def _variation(values: list[float]) -> dict[str, Any]:
    mean = statistics.mean(values) if values else None
    stdev = statistics.stdev(values) if len(values) > 1 else None
    return {
        "runs": len(values),
        "mean": mean,
        "stdev": stdev,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "cv": stdev / mean if stdev is not None and mean else None,
    }


def compare_runs(online: list[dict[str, Any]], cached: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    runs = online + cached
    if not online or not cached:
        reasons.append("both online and cached runs are required")
    if any(not run["complete"] for run in runs):
        reasons.append("incomplete run or missing provenance")
    if any(run.get("validation_lane") for run in runs):
        reasons.append("numerical validation lane cannot be used as a performance comparison")
    if any(not run.get("fresh_initialization") for run in runs):
        reasons.append("automatic speedup requires fresh initialization without a resume checkpoint")
    if any(run["mode"] not in ("online", "cached") for run in runs):
        reasons.append("unknown run mode")
    if any(run["mode"] != "online" for run in online) or any(run["mode"] != "cached" for run in cached):
        reasons.append("run mode does not match its comparison group")
    if runs and any(
        run["comparison_signature"] != runs[0]["comparison_signature"]
        or run["step_samples"] != runs[0]["step_samples"]
        for run in runs[1:]
    ):
        reasons.append("configuration, provenance or effective samples differ")
    order_known = bool(runs) and all(run.get("sample_order") is not None for run in runs)
    if not order_known:
        reasons.append("actual sample ordering is missing")
    elif any(run["sample_order"] != runs[0]["sample_order"] for run in runs[1:]):
        reasons.append("actual effective sample order differs")
    if any(run["steady"]["samples_per_second"] is None for run in runs):
        reasons.append("no steady-state window after the cold steps")
    result: dict[str, Any] = {
        "comparable": not reasons,
        "reasons": reasons,
        "sample_order_verified": order_known
        and not any(run["sample_order"] != runs[0]["sample_order"] for run in runs[1:]),
    }
    if reasons:
        return result
    a = _variation([run["steady"]["samples_per_second"] for run in online])
    b = _variation([run["steady"]["samples_per_second"] for run in cached])
    speedup = b["mean"] / a["mean"] - 1
    samples = runs[0]["training"]["samples"]
    online_cold = statistics.mean(run["cold"]["seconds"] for run in online)
    cached_cold = statistics.mean(run["cold"]["seconds"] for run in cached)
    init_a = [run["time_init_seconds"] for run in online]
    init_b = [run["time_init_seconds"] for run in cached]
    startup_extra = (
        max(0.0, statistics.mean(init_b) - statistics.mean(init_a))
        if all(value is not None for value in init_a + init_b)
        else None
    )
    cold_extra = max(0.0, cached_cold - online_cold)
    savings_per_sample = 1 / a["mean"] - 1 / b["mean"]
    overhead = startup_extra + cold_extra if startup_extra is not None else None
    payback = overhead / savings_per_sample if overhead is not None and savings_per_sample > 0 else None
    total_a, total_b = [run["total_seconds"] for run in online], [run["total_seconds"] for run in cached]
    job_a, job_b = [run["job_wall_seconds"] for run in online], [run["job_wall_seconds"] for run in cached]
    have_job_timing = all(value is not None for value in job_a + job_b)
    online_job_rate = _variation([samples / value for value in job_a]) if have_job_timing else None
    cached_job_rate = _variation([samples / value for value in job_b]) if have_job_timing else None
    result.update(
        {
            "online_steady_samples_per_second": a,
            "cached_steady_samples_per_second": b,
            "steady_throughput_improvement": speedup,
            "observed_at_least_five_percent": speedup >= 0.05,
            "repeated_measurements": min(len(online), len(cached)) >= 2,
            "cold_epoch_extra_seconds": cold_extra,
            "startup_extra_seconds": startup_extra,
            "estimated_additional_samples_to_amortize": payback,
            "estimated_additional_epochs_to_amortize": payback / runs[0]["cold"]["samples"]
            if payback is not None and runs[0]["cold"]["samples"]
            else None,
            "payback_note": "Estimate after the measured cold window, using its actual excess time plus startup excess; encoding/store times are diagnostics and are not charged a second time. Confirm with cumulative wall-time measurements.",
            "online_mean_trainer_seconds": statistics.mean(total_a) if all(x is not None for x in total_a) else None,
            "cached_mean_trainer_seconds": statistics.mean(total_b) if all(x is not None for x in total_b) else None,
            "online_mean_job_wall_seconds": statistics.mean(job_a) if have_job_timing else None,
            "cached_mean_job_wall_seconds": statistics.mean(job_b) if have_job_timing else None,
            "online_job_samples_per_second": online_job_rate,
            "cached_job_samples_per_second": cached_job_rate,
            "full_job_throughput_improvement": cached_job_rate["mean"] / online_job_rate["mean"] - 1
            if have_job_timing
            else None,
            "job_timing_note": "Ray job.json includes driver/model startup and job teardown."
            if have_job_timing
            else "Full Ray job duration unknown: provide successful job.json for every run. Trainer times do not include model/driver startup.",
            "effective_samples_per_run": samples,
        }
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for group in ("online", "cached", "low-reuse-online", "low-reuse-cached"):
        parser.add_argument(f"--{group}", type=Path, nargs="+", action="extend", default=[])
    parser.add_argument("--cold-steps", type=int, default=16, help="First epoch for a 128-row / GBS8 run")
    parser.add_argument("--output", type=Path, help="Write the full JSON report; stdout remains readable")
    args = parser.parse_args(argv)
    if args.cold_steps < 0:
        parser.error("--cold-steps must be nonnegative")
    groups = {}
    try:
        for group in ("online", "cached", "low_reuse_online", "low_reuse_cached"):
            groups[group] = [summarize_run(path, args.cold_steps) for path in getattr(args, group)]
        if not any(groups.values()):
            parser.error("provide at least one run directory")
        report = {
            "schema_version": 1,
            "cold_steps": args.cold_steps,
            "runs": groups,
            "comparison": compare_runs(groups["online"], groups["cached"]),
            "low_reuse_comparison": compare_runs(groups["low_reuse_online"], groups["low_reuse_cached"]),
        }
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for group, runs in groups.items():
        for run in runs:
            rate = run["steady"]["samples_per_second"]
            print(
                f"{group}: {run['path']} complete={run['complete']} steady={rate if rate is not None else 'unknown'} samples/s"
            )
    for name in ("comparison", "low_reuse_comparison"):
        result = report[name]
        if result["comparable"]:
            print(
                f"{name}: steady throughput {result['steady_throughput_improvement']:+.2%}; repeated={result['repeated_measurements']}"
            )
        else:
            print(f"{name}: not compared: {'; '.join(result['reasons'])}")
    if not args.output:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main(sys.argv[1:])
