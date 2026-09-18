# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU time and peak-memory benchmark for the full-text repetition scan.

Not a pytest module: run it directly to reproduce the numbers quoted in
``docs/en/guide/repetition-detection.md``::

    python tests/utils/benchmark_repetition_scan.py
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

from relax.utils.repetition import (
    REPETITION_WINDOW_SIZE_CHARS,
    REPETITION_WINDOW_STRIDE_CHARS,
    has_repetition,
    repetition_window_bounds,
    scan_repetition,
)


def _incompressible(n: int, salt: str = "") -> str:
    """Worst case for the scan: no window can short-circuit on a hit."""
    chunks = []
    total = 0
    i = 0
    while total < n:
        chunk = hashlib.sha256(f"{salt}:{i}".encode()).hexdigest()
        chunks.append(chunk)
        total += len(chunk)
        i += 1
    return "".join(chunks)[:n]


def _case_text(length: int, case: str, salt: str = "bench") -> str:
    if case == "repetitive":
        return ("spam" * ((length + 3) // 4))[:length]
    text = _incompressible(length, salt)
    if case != "clean":
        span = min(length, 20_000)
        start = {"beginning": 0, "middle": (length - span) // 2, "end": length - span}[case]
        text = text[:start] + ("spam" * ((span + 3) // 4))[:span] + text[start + span :]
    return text


def _peak_rss_mib(length: int, case: str) -> float:
    """Measure total lifetime RSS in a fresh interpreter for each scan case."""
    output = subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve()), "--rss-worker", str(length), case], text=True
    )
    return float(output)


def benchmark(length: int, repeats: int, case: str = "clean") -> dict:
    text = _case_text(length, case)
    num_windows = len(repetition_window_bounds(length))
    scan_repetition(text)
    wall_durations = []
    cpu_durations = []
    for _ in range(repeats):
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        report = scan_repetition(text)
        cpu_durations.append(time.process_time() - cpu_start)
        wall_durations.append(time.perf_counter() - wall_start)
    assert report.has_repetition == (case != "clean")

    tracemalloc.start()
    scan_repetition(text)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mean_wall = round(1000 * sum(wall_durations) / repeats, 2)
    return {
        "chars": length,
        "case": case,
        "num_windows": num_windows,
        "has_repetition": report.has_repetition,
        "wall_mean_ms": mean_wall,
        "cpu_mean_ms": round(1000 * sum(cpu_durations) / repeats, 2),
        "min_ms": round(1000 * min(wall_durations), 2),
        "max_ms": round(1000 * max(wall_durations), 2),
        "peak_mem_mib": round(peak / 1024 / 1024, 2),
        "peak_rss_mib": _peak_rss_mib(length, case),
    }


def benchmark_batch(length: int, batch_size: int, repeats: int, position: str) -> dict:
    """Measure only the online boolean predicate and mean, not other rollout
    metrics.

    Reuse an immutable response to exclude input construction from scan timing.
    Repeat position matters: a tail hit still scans all preceding clean windows.

    The mean is computed in plain Python rather than with numpy: it is the same
    value the online metric aggregates, and keeping it dependency-free lets this
    benchmark run wherever the detector itself runs.
    """
    text = _case_text(length, position, "batch")

    expected = float(position != "clean")
    assert float(has_repetition(text)) == expected
    durations = []
    cpu_durations = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        cpu_start = time.process_time()
        flags = [int(has_repetition(text)) for _ in range(batch_size)]
        fraction = sum(flags) / len(flags)
        cpu_durations.append(time.process_time() - cpu_start)
        durations.append(time.perf_counter() - start_time)
        assert fraction == expected
    return {
        "chars": length,
        "batch_size": batch_size,
        "position": position,
        "wall_mean_ms": round(1000 * sum(durations) / len(durations), 2),
        "cpu_mean_ms": round(1000 * sum(cpu_durations) / len(cpu_durations), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument("--batches", action="store_true", help="Also time clean/beginning/middle/end batch cases.")
    parser.add_argument("--rss-worker", nargs=2, metavar=("LENGTH", "CASE"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rss_worker:
        import resource

        length, case = args.rss_worker
        report = scan_repetition(_case_text(int(length), case))
        assert report.has_repetition == (case != "clean")
        # macOS reports bytes; Linux reports KiB. This is total lifetime RSS,
        # including the interpreter, imports, input construction and scan.
        divisor = 1024**2 if sys.platform == "darwin" else 1024
        print(round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor, 2))  # noqa: T201
        return
    if args.repeats <= 0 or any(length < REPETITION_WINDOW_SIZE_CHARS for length in args.lengths):
        parser.error("repeats must be positive and lengths must be at least one window")

    rows = [
        benchmark(length, args.repeats, case) for length in args.lengths for case in ("clean", "middle", "repetitive")
    ]
    batches = (
        [
            benchmark_batch(length, size, args.repeats, position)
            for size, length in [(512, 40_000), (1024, 40_000), (512, 200_000)]
            for position in ("clean", "beginning", "middle", "end")
        ]
        if args.batches
        else []
    )
    environment = {"python": platform.python_version(), "machine": platform.machine(), "system": platform.platform()}

    if args.json:
        print(
            json.dumps(
                {"environment": environment, "repeats": args.repeats, "scans": rows, "batches": batches}, indent=2
            )
        )  # noqa: T201
        return

    print(environment)  # noqa: T201 - benchmark CLI output
    header = f"window={REPETITION_WINDOW_SIZE_CHARS} stride={REPETITION_WINDOW_STRIDE_CHARS} repeats={args.repeats}"
    print(header)  # noqa: T201 - benchmark CLI output
    print(  # noqa: T201 - benchmark CLI output
        f"{'chars':>10} {'case':>10} {'windows':>8} {'wall ms':>9} {'CPU ms':>9} {'traced MiB':>10} {'RSS MiB':>9}"
    )
    for row in rows:
        print(  # noqa: T201 - benchmark CLI output
            f"{row['chars']:>10} {row['case']:>10} {row['num_windows']:>8} {row['wall_mean_ms']:>9} "
            f"{row['cpu_mean_ms']:>9} {row['peak_mem_mib']:>10} {row['peak_rss_mib']:>9}"
        )
    for row in batches:
        print(row)  # noqa: T201 - benchmark CLI output


if __name__ == "__main__":
    main()
