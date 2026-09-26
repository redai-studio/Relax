# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU async scheduling benchmark: session-level vs per-request permits.

Simulates ``capacity=1`` rollout scheduling with long multi-turn sessions plus
short single-turn requests, using the real ``InferencePermitManager`` for the
per-request policy. Reports short-request latency, total wall time, and peak
concurrent model requests for both policies.

This is a *scheduling* benchmark on synthetic per-turn durations (no GPU, no
model, no network); it validates the concurrency bound and the fairness win of
per-request permits, not model throughput.

Usage:
    PYTHONPATH=. python scripts/benchmarks/benchmark_request_permit.py --warmups 1 --runs 9
"""

import argparse
import asyncio
import statistics
from time import monotonic

from relax.engine.rollout.request_permit import InferencePermitManager


async def _model_request(duration: float, counters: dict[str, int]) -> None:
    counters["inflight"] += 1
    counters["peak"] = max(counters["peak"], counters["inflight"])
    await asyncio.sleep(duration)
    counters["inflight"] -= 1


async def _run_once(per_request: bool, cfg: dict) -> tuple[float, list[float], int]:
    counters = {"inflight": 0, "peak": 0}
    short_latencies: list[float] = []
    mgr = InferencePermitManager(cfg["capacity"])
    sem = mgr.semaphore

    async def long_session() -> None:
        if per_request:
            for _ in range(cfg["long_turns"]):
                async with mgr.permit():  # permit held only around the model request
                    await _model_request(cfg["model_s"], counters)
                await asyncio.sleep(cfg["env_s"])  # env/tool runs OUTSIDE the permit
        else:
            async with sem:  # session-level: hold the slot for the whole session
                for _ in range(cfg["long_turns"]):
                    await _model_request(cfg["model_s"], counters)
                    await asyncio.sleep(cfg["env_s"])  # env/tool runs INSIDE the lock

    async def short_request() -> None:
        start = monotonic()
        if per_request:
            async with mgr.permit():
                await _model_request(cfg["short_model_s"], counters)
        else:
            async with sem:
                await _model_request(cfg["short_model_s"], counters)
        short_latencies.append(monotonic() - start)

    async def launch_shorts() -> None:
        # Let long sessions grab the slot first, then short requests arrive.
        await asyncio.sleep(cfg["model_s"] * 0.5)
        await asyncio.gather(*[short_request() for _ in range(cfg["n_short"])])

    start = monotonic()
    tasks = [asyncio.create_task(long_session()) for _ in range(cfg["n_long"])]
    tasks.append(asyncio.create_task(launch_shorts()))
    await asyncio.gather(*tasks)
    total = monotonic() - start
    return total, short_latencies, counters["peak"]


async def _bench(cfg: dict, warmups: int, runs: int) -> dict[bool, tuple[float, float, int]]:
    results: dict[bool, tuple[float, float, int]] = {}
    for per_request in (False, True):
        for _ in range(warmups):
            await _run_once(per_request, cfg)
        totals: list[float] = []
        short_medians: list[float] = []
        peaks: list[int] = []
        for _ in range(runs):
            total, shorts, peak = await _run_once(per_request, cfg)
            totals.append(total)
            short_medians.append(statistics.median(shorts))
            peaks.append(peak)
        results[per_request] = (statistics.median(totals), statistics.median(short_medians), max(peaks))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--capacity", type=int, default=1)
    parser.add_argument("--n-long", type=int, default=4)
    parser.add_argument("--long-turns", type=int, default=3)
    parser.add_argument("--model-s", type=float, default=0.02)
    parser.add_argument("--env-s", type=float, default=0.02)
    parser.add_argument("--n-short", type=int, default=4)
    parser.add_argument("--short-model-s", type=float, default=0.005)
    args = parser.parse_args()

    cfg = {
        "capacity": args.capacity,
        "n_long": args.n_long,
        "long_turns": args.long_turns,
        "model_s": args.model_s,
        "env_s": args.env_s,
        "n_short": args.n_short,
        "short_model_s": args.short_model_s,
    }
    results = asyncio.run(_bench(cfg, args.warmups, args.runs))

    def ms(seconds: float) -> str:
        return f"{seconds * 1000:.2f} ms"

    print(
        f"CPU async scheduling benchmark (capacity={args.capacity}, {args.warmups} warmup / {args.runs} runs, median)"
    )
    print(
        f"scenario: {args.n_long} long sessions x {args.long_turns} turns "
        f"(model {ms(args.model_s)} + env {ms(args.env_s)} per turn), "
        f"{args.n_short} short requests (model {ms(args.short_model_s)})"
    )
    print()
    header = f"{'Permit scope':<18}{'Median total':<16}{'Median short-req latency':<28}{'Peak model reqs'}"
    print(header)
    for per_request, label in ((False, "Session-level"), (True, "Per-request")):
        total, short, peak = results[per_request]
        print(f"{label:<18}{ms(total):<16}{ms(short):<28}{peak}")

    for per_request in (False, True):
        assert results[per_request][2] <= args.capacity, "peak concurrent requests exceeded capacity"

    session_short = results[False][1]
    request_short = results[True][1]
    if session_short > 0:
        delta = (request_short - session_short) / session_short * 100
        print()
        print(f"Short-request latency: {ms(session_short)} -> {ms(request_short)} ({delta:.1f}%)")


if __name__ == "__main__":
    main()
