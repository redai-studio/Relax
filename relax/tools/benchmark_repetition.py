# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reproducible full-response repetition benchmarks, without optional
dependencies.

Run ``python -m relax.tools.benchmark_repetition --output
benchmarks/repetition/local.json``. Every case/API uses a fresh subprocess.
Timings exclude imports/input generation and tracemalloc. Process peak memory
includes interpreter, imports, input creation, warmup and timed scans;
tracemalloc measures a separate scan after input creation.
"""

import argparse
import ctypes
import json
import math
import os
import platform
import random
import statistics
import string
import subprocess
import sys
import tempfile
import time
import tracemalloc
import zlib
from pathlib import Path

from relax.utils.logging_utils import get_logger
from relax.utils.repetition import RepetitionConfig, analyze_repetition, has_repetition, iter_repetition_windows


logger = get_logger(__name__)
CASES = ("control", "middle_repeat", "repeated", "chinese_middle_repeat")


def make_response(length: int, case: str, seed: int) -> str:
    """Generate deterministic text; middle-repeat cases retain a normal
    suffix."""
    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + string.punctuation + " "
    if case == "chinese_middle_repeat":
        alphabet = "".join(chr(code) for code in range(0x4E00, 0x7000))
    if case == "repeated":
        return ("The response repeats this sentence. " * (length // 35 + 1))[:length]
    text = "".join(rng.choices(alphabet, k=length))
    if case in ("middle_repeat", "chinese_middle_repeat"):
        # At 10k the repeated middle is intentionally diluted by normal ends;
        # at 100k/1m at least one full window is inside the repeated region.
        start, end = length // 4, 3 * length // 4
        phrase = "重复文本用于诊断。" if case == "chinese_middle_repeat" else "Repeated diagnostic sentence. "
        middle = (phrase * ((end - start) // len(phrase) + 1))[: end - start]
        text = text[:start] + middle + text[end:]
    return text


def process_peak_bytes() -> int:
    """Read native OS process high-water memory, including native zlib
    allocations."""
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "median": statistics.median(values),
        "minimum": ordered[0],
        "p95_nearest_rank": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "runs": values,
    }


def run_case(length: int, case: str, api: str, repeats: int, seed: int) -> dict:
    text = make_response(length, case, seed)
    config = RepetitionConfig()
    function = analyze_repetition if api == "detailed" else has_repetition
    warmup_start = time.perf_counter()
    function(text, config)  # Warm caches outside timing; no window is omitted.
    warmup_seconds = time.perf_counter() - warmup_start
    # Windows process_time can advance in 15.625 ms ticks. Batch full calls for
    # at least approximately 100 ms, then normalize to time per response.
    batch_calls = max(1, math.ceil(0.1 / max(warmup_seconds, 1e-6)))
    baseline_peak = process_peak_bytes()
    cpu, wall = [], []
    for _ in range(repeats):
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        for _ in range(batch_calls):
            result = function(text, config)
        cpu.append((time.process_time() - cpu_start) / batch_calls)
        wall.append((time.perf_counter() - wall_start) / batch_calls)
    native_peak = process_peak_bytes()
    tracemalloc.start()
    traced_result = function(text, config)
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del traced_result
    expected_windows = sum(1 for _ in iter_repetition_windows(length, config))
    if api == "detailed" and result.scanned_windows != expected_windows:
        raise AssertionError("Detailed scan did not visit all windows")
    hit = result.has_repetition if api == "detailed" else result
    if case == "control" and hit:
        raise AssertionError("Deterministic control unexpectedly classified as repeated")
    return {
        "characters": len(text),
        "utf8_bytes": len(text.encode("utf-8")),
        "case": case,
        "api": api,
        "calls_per_timing_batch": batch_calls,
        "has_repetition": hit,
        "full_scan_windows": expected_windows,
        "actual_scanned_windows": result.scanned_windows if api == "detailed" else None,
        "cpu_seconds": summarize(cpu),
        "wall_seconds": summarize(wall),
        "process_peak_bytes_before_timing": baseline_peak,
        "process_peak_bytes": native_peak,
        "scan_tracemalloc_peak_bytes": traced_peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--lengths", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--worker-case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-api", choices=("detailed", "boolean"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 5 or any(length <= 0 for length in args.lengths):
        parser.error("repeats must be at least 5 and lengths must be positive")
    if bool(args.worker_case) != bool(args.worker_api):
        parser.error("worker-case and worker-api must be provided together")
    if args.worker_case:
        report = run_case(args.lengths[0], args.worker_case, args.worker_api, args.repeats, args.seed)
    else:
        results = []
        with tempfile.TemporaryDirectory(prefix="relax-repetition-benchmark-") as temporary:
            child_output = Path(temporary) / "case.json"
            for length in args.lengths:
                for case in CASES:
                    for api in ("detailed", "boolean"):
                        command = [
                            sys.executable,
                            "-m",
                            "relax.tools.benchmark_repetition",
                            "--output",
                            str(child_output),
                            "--repeats",
                            str(args.repeats),
                            "--lengths",
                            str(length),
                            "--seed",
                            str(args.seed),
                            "--worker-case",
                            case,
                            "--worker-api",
                            api,
                        ]
                        subprocess.run(command, check=True)
                        results.append(json.loads(child_output.read_text(encoding="utf-8")))
                        logger.info("Benchmarked %s characters, %s, %s", length, case, api)
        report = {
            "schema_version": 1,
            "environment": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpus": os.cpu_count(),
                "python": sys.version,
                "zlib_build": zlib.ZLIB_VERSION,
                "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
            },
            "configuration": {"window_size": 10_000, "stride": 5_000, "threshold": 10.0},
            "repeats": args.repeats,
            "seed": args.seed,
            "methodology": {
                "isolation": "Fresh subprocess per case, length and API; one unmeasured warmup.",
                "timing": "Seconds per response; full calls batched to approximately 100 ms to mitigate CPU clock "
                "quantization. Input/imports and tracemalloc excluded. p95 uses nearest rank.",
                "process_peak": "OS peak working set/RSS in bytes; includes interpreter, imports, input and warmup. "
                "Captured before tracemalloc; not an incremental allocation estimate.",
                "tracemalloc": "Separate scan peak in bytes, tracing started after input/imports; "
                "includes allocations visible to Python tracer, not all native allocations.",
                "boolean": "May stop at first hit. Control must scan every window; detailed always scans all.",
                "middle_repeat": "Middle half repeated, first/last quarters deterministic random text. "
                "10k mixed window may correctly remain below threshold.",
            },
            "results": results,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
