# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU repetition benchmark.

Run from the repository root with PYTHONPATH=.
"""

import argparse
import json
import platform
import random
import resource
import statistics
import string
import subprocess
import sys
import time
import tracemalloc
import zlib
from functools import partial
from pathlib import Path
from typing import Any

from relax.utils.repetition import detect_repetition


def benchmark_case(length: int, pattern: str, runs: int, batch_size: int) -> dict[str, Any]:
    if pattern == "repetitive":
        text = "a" * length
    else:
        text = "".join(random.Random(42).choices(string.ascii_letters + string.digits, k=length))
        if pattern in ("beginning", "middle", "end"):
            repeat_length = min(length, 10_000)
            start = {"beginning": 0, "middle": (length - repeat_length) // 2, "end": length - repeat_length}[pattern]
            text = text[:start] + "a" * repeat_length + text[start + repeat_length :]
    result = detect_repetition(text)
    timings = {}
    boolean_detector = partial(detect_repetition, stop_after_first_hit=True)
    for detector, prefix in ((detect_repetition, ""), (boolean_detector, "boolean_")):
        detector(text)
        wall_times, cpu_times = [], []
        for _ in range(runs):
            wall_start, cpu_start = time.perf_counter(), time.process_time()
            for _ in range(batch_size):
                detector(text)
            wall_times.append((time.perf_counter() - wall_start) * 1_000)
            cpu_times.append((time.process_time() - cpu_start) * 1_000)
        timings[f"{prefix}median_wall_ms"] = round(statistics.median(wall_times), 3)
        timings[f"{prefix}median_cpu_ms"] = round(statistics.median(cpu_times), 3)
    # Separate allocation measurement avoids adding tracing overhead to timings.
    tracemalloc.start()
    detect_repetition(text)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "characters": length,
        "pattern": pattern,
        "batch_size": batch_size,
        "window_count": result.window_count,
        "hit_count": len(result.hit_windows),
        **timings,
        "peak_traced_bytes": peak,
        "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1_024,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lengths", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["control", "beginning", "middle", "end", "repetitive"],
        default=["control", "middle", "repetitive"],
    )
    parser.add_argument("--worker", nargs=2, metavar=("LENGTH", "PATTERN"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runs <= 0 or args.batch_size <= 0 or any(length <= 0 for length in args.lengths):
        parser.error("--runs, --batch-size and --lengths must be positive")
    if args.worker:
        report = benchmark_case(int(args.worker[0]), args.worker[1], args.runs, args.batch_size)
    else:
        cases = []
        for length in args.lengths:
            for pattern in args.patterns:
                worker = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--runs",
                        str(args.runs),
                        "--batch-size",
                        str(args.batch_size),
                        "--worker",
                        str(length),
                        pattern,
                    ],
                    check=True,
                    text=True,
                    capture_output=True,
                )
                cases.append(json.loads(worker.stdout))
        report = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "zlib": zlib.ZLIB_RUNTIME_VERSION,
            "runs": args.runs,
            "seed": 42,
            "window_size": 10_000,
            "stride": 5_000,
            "threshold": 10.0,
            "timing_notes": (
                "Times cover batch_size calls on the same input. Unprefixed times measure full diagnostics; "
                "boolean_* measures stop_after_first_hit=True. Both use the current implementation."
            ),
            "memory_notes": (
                "Traced peak measures one full diagnostic call, not the batch or early-exit path; "
                "RSS is a fresh process including imports and input creation."
            ),
            "cases": cases,
        }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
