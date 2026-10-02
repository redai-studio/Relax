# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Repeated ordinary/managed LoRA throughput experiment, without publication.

Run with --help. All engines are deployed and stopped by the existing
acceptance runner. Both arms use the patched SGLang build: this measures the
managed path's incremental cost, not the entire patch versus upstream. Native
generation is measured after binding; Agentic binding latency is outside this
experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import signal
import statistics
import subprocess
import time
from pathlib import Path

from relax.utils.logging_utils import get_logger

from . import performance, support


logger = get_logger(__name__)


def analyze(blocks: list[dict], target_percent: float, *, seed: int = 20260923) -> dict:
    """Bootstrap whole balanced blocks, never individual tokens or requests.

    Percentile intervals are approximate and assume independent blocks.
    Temporal correlation or systematic bias is not removed by increasing
    bootstrap draws. Use a fixed number of blocks; repeated peeking until PASS
    is not supported.
    """
    if len(blocks) < 6 or not math.isfinite(target_percent) or target_percent <= 0:
        raise ValueError("need at least six complete blocks and a positive finite target")
    engines = set(blocks[0]["windows"][0]["metrics"]["token_throughput"])
    ratios = {engine: [] for engine in sorted(engines)}
    for block in blocks:
        windows = block["windows"]
        modes = [window["mode"] for window in windows]
        if modes not in (
            ["ordinary", "managed", "managed", "ordinary"],
            ["managed", "ordinary", "ordinary", "managed"],
        ):
            raise ValueError("each block must contain a complete ABBA or BAAB sequence")
        for window in windows:
            rates = window["metrics"]["token_throughput"]
            if set(rates) != engines or any(not math.isfinite(v) or v <= 0 for v in rates.values()):
                raise ValueError("missing engine or invalid throughput")
        for engine in ratios:
            # Difference of mean log rates is the block's paired log ratio.
            ratios[engine].append(
                sum(
                    (1 if window["mode"] == "managed" else -1)
                    * math.log(window["metrics"]["token_throughput"][engine])
                    / 2
                    for window in windows
                )
            )
    rng = random.Random(seed)
    samples = [[rng.randrange(len(blocks)) for _ in blocks] for _ in range(20000)]
    # Bonferroni intervals across aggregate + individual engines; do not hide
    # a single-engine regression behind an aggregate improvement.
    tail = 0.05 / (2 * len(ratios))
    results = {}
    for engine, values in ratios.items():
        draws = sorted(100 * -math.expm1(statistics.fmean(values[i] for i in indices)) for indices in samples)
        lower, upper = draws[int(tail * len(draws))], draws[math.ceil((1 - tail) * len(draws)) - 1]
        verdict = "PASS" if upper <= target_percent else "FAIL" if lower > target_percent else "INCONCLUSIVE"
        if max(values) == min(values):
            verdict = "INCONCLUSIVE"  # No measured variance is not proof of sub-per-mille precision.
        results[engine] = {
            "overhead_percent": 100 * -math.expm1(statistics.fmean(values)),
            "overhead_ci_percent": [lower, upper],
            "verdict": verdict,
            "block_overhead_percent": [100 * -math.expm1(value) for value in values],
        }
    verdicts = {result["verdict"] for result in results.values()}
    return {
        "verdict": "FAIL" if "FAIL" in verdicts else "PASS" if verdicts == {"PASS"} else "INCONCLUSIVE",
        "target_percent": target_percent,
        "metrics": results,
        "method": "paired log-rate ratio; 20000 whole-block percentile bootstrap draws; Bonferroni family 95%",
        "bootstrap_seed": seed,
        "limitations": "Approximate intervals assume independent blocks; systematic bias and cross-block drift remain. "
        "Both arms share GPUs with idle peer engines and use patched SGLang. This does not measure patch-wide cost, "
        "first-bind latency, or publication cost. PASS applies only to this fixed workload and inference profile.",
    }


def gpu_state(gpus: list[dict]) -> dict:
    """Sample outside timed windows; never change clocks or power settings."""
    fields = "uuid,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,memory.used"
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                ",".join(gpu["uuid"] for gpu in gpus),
                "--query-gpu=" + fields,
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        return {
            "at": time.time(),
            "devices": [dict(zip(fields.split(","), row, strict=True)) for row in csv.reader(output.splitlines())],
        }
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"error": str(error)}


async def run(ctx) -> None:
    profile = ctx.config["overhead_profile"]
    bindings = await support.native_bindings(ctx)
    blocks = []
    ctx.report["overhead_blocks"] = blocks
    checkpoint = Path(ctx.config["output"]) / "overhead.json"
    for index in range(0 if profile.get("diagnostics_only") else profile["rounds"]):
        order = ["ordinary", "managed", "managed", "ordinary"]
        if index % 2:
            order = ["managed" if mode == "ordinary" else "ordinary" for mode in order]
        block = {"index": index, "windows": []}
        blocks.append(block)
        for mode in order:
            logger.info("Overhead block %s/%s: %s", index + 1, profile["rounds"], mode)
            hardware_before = gpu_state(ctx.config["gpus"])
            options = {"observe": False, "warmup_seconds": profile["warmup_seconds"]}
            if mode == "ordinary":
                metrics = await performance.ordinary_baseline(
                    ctx, bindings, concurrency=profile["concurrency"], **options
                )
                if metrics is None:
                    raise AssertionError("ordinary reference is required")
            else:
                metrics = await performance.traffic_window(ctx, bindings * profile["concurrency"], **options)
            raw_path = Path(ctx.config["output"]) / f"overhead-{index:02d}-{len(block['windows'])}-{mode}.json"
            raw_path.write_text(json.dumps(ctx.report["traffic_raw"].pop()) + "\n")
            block["windows"].append(
                {
                    "mode": mode,
                    "metrics": metrics,
                    "raw_file": str(raw_path),
                    "hardware_before": hardware_before,
                    "hardware_after": gpu_state(ctx.config["gpus"]),
                }
            )
            # Partial evidence survives later failures; no interim significance decisions.
            checkpoint.write_text(json.dumps({"verdict": "INCOMPLETE", "blocks": blocks}, indent=2) + "\n")
    result = analyze(blocks, profile["target_percent"]) if blocks else {"verdict": "NOT_RUN"}
    if profile.get("diagnostics", True):
        from .overhead_diagnostics import collect

        logger.info("Formal windows finished; starting separate intrusive CPU/GPU diagnostics")
        # Persist formal evidence before any profiler can fail.
        checkpoint.write_text(json.dumps({**result, "blocks": blocks}, indent=2) + "\n")
        result["diagnostics"] = await collect(ctx, bindings)
    ctx.report["overhead_analysis"] = result
    checkpoint.write_text(json.dumps({**result, "blocks": blocks}, indent=2) + "\n")
    # Completion of a diagnostic is distinct from meeting its overhead target.
    ctx.report["checks"] = {"steady_state_overhead": {"status": "PASS", "analysis": result}}


def main() -> int:
    from .__main__ import run_suite

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gpus", nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--rounds", type=int, default=12, help="Fixed balanced blocks (four windows each); minimum 6")
    parser.add_argument("--window-seconds", type=float, default=120)
    parser.add_argument(
        "--warmup-seconds", type=float, default=30, help="Additional warmup after every client completes a request"
    )
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent requests per engine, 1..32")
    parser.add_argument("--target-percent", type=float, default=0.05)
    parser.add_argument(
        "--deterministic", action="store_true", help="Apply the same deterministic Triton profile to both arms"
    )
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument(
        "--diagnostics-only", action="store_true", help="Collect detailed traces without the two-hour comparison"
    )
    diagnostics.add_argument(
        "--skip-diagnostics", action="store_true", help="Only run the formal throughput comparison"
    )
    parser.add_argument(
        "--server-timings",
        action="store_true",
        help="Enable native metrics on BOTH arms (adds observability cost to both); useful with --diagnostics-only",
    )
    args = parser.parse_args()
    if args.rounds < 6 or not 1 <= args.concurrency <= 32:
        parser.error("--rounds must be >= 6; --concurrency must be 1..32")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.window_seconds, args.warmup_seconds, args.target_percent)
    ):
        parser.error("window, warmup and target must be positive finite values")
    args.tasks = ["performance"]
    args.overhead_profile = {
        "overhead_profile": {
            "rounds": args.rounds,
            "warmup_seconds": args.warmup_seconds,
            "concurrency": args.concurrency,
            "target_percent": args.target_percent,
            "diagnostics": not args.skip_diagnostics,
            "diagnostics_only": args.diagnostics_only,
        },
        "native_metrics": args.server_timings,
        "traffic_window_seconds": args.window_seconds,
        "timeout_seconds": math.ceil(4 * args.rounds * (args.window_seconds + args.warmup_seconds + 240) + 2400),
    }
    import shutil

    if not args.skip_diagnostics and not shutil.which("py-spy"):
        logger.warning("py-spy not found: Python sampling will be INCOMPLETE; native CPU/GPU traces remain enabled")
    logger.info(
        "Collecting %s windows; measurement + warmup alone %.1f minutes; graph/cache remain enabled",
        0 if args.diagnostics_only else 4 * args.rounds,
        0 if args.diagnostics_only else 4 * args.rounds * (args.window_seconds + args.warmup_seconds) / 60,
    )
    started = time.monotonic()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        run_suite(args)
    finally:
        signal.signal(signal.SIGTERM, previous)
    result = json.loads((args.output / "performance" / "overhead.json").read_text())
    logger.info(
        "Overhead target verdict %s after %.1f minutes; report: %s",
        result["verdict"],
        (time.monotonic() - started) / 60,
        args.output / "performance" / "overhead.json",
    )
    if "diagnostics" in result:
        logger.info(
            "Detailed diagnostics: %s; report %s",
            result["diagnostics"]["status"],
            args.output / "performance" / "diagnostics" / "report.json",
        )
    if result["verdict"] == "FAIL":
        return 1
    if result.get("diagnostics", {}).get("status", "COMPLETE") != "COMPLETE":
        return 2
    return {"PASS": 0, "NOT_RUN": 0, "INCONCLUSIVE": 2}[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
