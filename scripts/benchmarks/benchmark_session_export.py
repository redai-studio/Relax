# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only Session export replay with open-loop short requests.

Run from the repository root with python -m
scripts.benchmarks.benchmark_session_export. Each measured variant runs in a
fresh subprocess. No model download, GPU, Ray cluster, network service, or
inference engine is needed.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import hashlib
import json
import math
import os
import platform
import pstats
import resource
import statistics
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from relax.agentic.pipeline import RuntimeGroupError, SessionExportTransport
from relax.agentic.session.export import SessionExportWorker
from relax.agentic.session.service import (
    AgenticSessionShard,
    ResidentGroup,
    SessionResources,
    _SessionRecord,
    _SessionResultCell,
)
from relax.agentic.session.state import SessionForest


class ReplayTokenizer:
    """Deterministic CPU decoder; the only replaced model-facing operation."""

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        return bytes(32 + token % 95 for token in token_ids).decode("ascii")


class MockBackend:
    def __init__(self) -> None:
        self.tokenizer = ReplayTokenizer()
        self.compiler = SimpleNamespace(apply_chat_template_kwargs={})
        self.closed_sessions: list[str] = []
        self.stopped = False

    async def generate(self, length: int) -> tuple[list[int], list[float]]:
        await asyncio.sleep(0)
        return [i % 1024 for i in range(length)], [-0.125 - (i % 7) / 16 for i in range(length)]

    async def abort_request(self, request_id: str) -> None:
        pass

    async def close_session(self, session_id: str, **kwargs: Any) -> bool:
        self.closed_sessions.append(session_id)
        return True

    async def shutdown(self) -> None:
        self.stopped = True


class CompletedProcess:
    def __init__(self, output: Any) -> None:
        self.output = output
        self.terminated = 0

    async def terminate_and_join(self) -> None:
        self.terminated += 1


class InlineWorker:
    """The original synchronous dispatch, with the identical assembly path."""

    async def run(self, materialize: Callable[[], Any]) -> Any:
        return materialize()

    async def shutdown(self) -> None:
        pass


def make_shard(mode: str = "thread") -> Any:
    shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = SimpleNamespace(
        agentic_session_lifecycle=True, use_rollout_routing_replay=False, num_layers=2, moe_router_topk=2
    )
    shard._groups = {}
    shard._session_records = {}
    shard._train_generation_open = False
    shard._state_revision = 0
    shard._state_changed = asyncio.Event()
    shard._permit_cleanup_tasks = set()
    shard._lifecycle_close_count = 0
    shard._lifecycle_close_failure_count = 0
    shard._generation_backend = MockBackend()
    shard._export_worker = InlineWorker() if mode == "inline" else SessionExportWorker()
    return shard


async def make_session(shard: Any, number: int, tokens: int, turns: int = 8, rich: bool = False) -> Any:
    """Commit synthetic inference results using the real Forest node
    methods."""
    session_id = f"session-{number}"
    forest = SessionForest.create_empty(
        session_id=session_id,
        group_index=number,
        index=number,
        label="replay",
        train_metadata={"loss": "grpo"},
        metadata={"fixture": {"seed": 0}},
    )
    parent = forest.root_state_hash
    remaining = tokens
    for turn in range(turns):
        # Allocate exact total tokens, including tool observations.
        obs_length = min(32, max(1, remaining // (2 * (turns - turn))))
        obs_tokens = [31] * obs_length
        obs = forest.append_obs(
            parent_state_hash=parent,
            rollout_id=turn,
            abort_count=0,
            messages_delta=[{"role": "user", "content": f"question-{turn}"}],
            train_token_delta=obs_tokens,
            rollout_token_delta=list(obs_tokens),
            multimodal_train_inputs_delta={"pixels": np.arange(4)} if rich and turn == 0 else None,
        )
        remaining -= obs_length
        resp_length = remaining // (turns - turn)
        token_ids, logprobs = await shard._generation_backend.generate(resp_length)
        routes = None
        if rich:
            routes = np.full((forest.train_token_count(obs.state_hash) + resp_length - 1, 2, 2), turn, np.int32)
        resp = forest.append_resp(
            parent_state_hash=obs.state_hash,
            rollout_id=turn,
            abort_count=turn % 2,
            messages_delta=[{"role": "assistant", "content": f"answer-{turn}"}],
            token_delta=token_ids,
            logprob_delta=logprobs,
            topk_token_ids_delta=np.tile(np.arange(4, dtype=np.int32), (resp_length, 1)) if rich else None,
            topk_logprob_delta=np.full((resp_length, 4), -0.5, np.float32) if rich else None,
            weight_version_delta=["v1"],
            spec_delta={"spec_verify_ct": 1, "completion_token_num": resp_length},
            prefix_cache_delta={"cached_tokens": 8, "total_prompt_tokens": obs_length},
            status="completed",
            rollout_routed_experts=routes,
            export_metadata_patch={"request_id": f"request-{turn}", "nested": {"turn": turn}},
            wall_elapsed_s=0.02,
            generation_elapsed_s=0.01,
        )
        remaining -= resp_length
        parent = resp.state_hash
    assert remaining == 0 and forest.train_token_count(parent) == tokens
    cell = _SessionResultCell()
    group = ResidentGroup("train", f"group-{number}", {session_id: cell}, [])
    session = _SessionRecord(group, session_id, {}, cell, forest=forest)
    process = CompletedProcess(SimpleNamespace(reward={"score": 1.0}, metadata={"source": "agent"}, records=()))
    process_wait = asyncio.create_task(asyncio.sleep(0))
    watcher = asyncio.create_task(asyncio.sleep(0))
    session.resources = SessionResources(process, process_wait, watcher)
    group.sessions.append(session)
    shard._groups[group.group_id] = group
    shard._session_records[session_id] = session
    return session


def content_digest(value: Any) -> str:
    """Hash every payload field, excluding only nondeterministic trace
    times."""
    digest = hashlib.sha256()

    def visit(item: Any, in_trace: bool = False) -> None:
        if isinstance(item, np.ndarray):
            digest.update(f"array:{item.dtype}:{item.shape}:".encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict:")
            for key in sorted(item):
                # Trace event values use wall clocks; training metadata stays.
                if in_trace and key == "events":
                    continue
                visit(key)
                visit(item[key], in_trace or key == "agentic_trace")
        elif isinstance(item, (tuple, list)):
            digest.update(f"{type(item).__name__}:{len(item)}:".encode())
            for child in item:
                visit(child, in_trace)
        elif isinstance(item, float):
            digest.update(b"float:" + struct.pack("!d", item))
        else:
            digest.update(f"{type(item).__name__}:{item!r}:".encode())

    visit(value)
    return digest.hexdigest()


def distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        **{f"p{p}_ms": ordered[max(0, math.ceil(len(ordered) * p / 100) - 1)] * 1000 for p in (50, 95, 99)},
        "max_ms": ordered[-1] * 1000,
    }


async def run_case(args: argparse.Namespace) -> dict[str, Any]:
    shard = make_shard(args.mode)
    # One full-sized export warms assembly, NumPy and the worker thread.
    warmup = await make_session(shard, -1, args.tokens, rich=args.rich)
    await shard._finalize_terminal_session(warmup)
    shard._groups.clear()
    del warmup
    sessions = [await make_session(shard, n, args.tokens, rich=args.rich) for n in range(args.exports)]
    short_session = await make_session(shard, args.exports, 128, turns=1)
    short_session.resp_state_hash_by_request_id["short"] = next(iter(short_session.forest.leaf_state_hashes))
    await asyncio.sleep(0)

    if args.profile:
        output = sessions[0].resources.process.output
        profiler = cProfile.Profile()
        profiler.runcall(
            shard._build_session_export_transport,
            session=sessions[0],
            reward=output.reward,
            metadata=output.metadata,
            output_records=(),
            finalize_events={},
        )
        pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(18)
        await shard.shutdown()
        return {}

    export_latencies: list[float] = []
    short_latencies: list[float] = []
    ticker_delays: list[float] = []
    completed = 0
    limit = asyncio.Semaphore(args.concurrency)
    loop = asyncio.get_running_loop()
    start = loop.time()
    cpu_start = time.process_time()
    due = start + 0.05
    exports_done = asyncio.Event()
    stop_at = start + args.min_window

    async def export(session: Any) -> None:
        nonlocal completed
        await asyncio.sleep(max(0, due - loop.time()))
        async with limit:
            await shard._finalize_terminal_session(session)
        if not isinstance(session.result_cell.payload, SessionExportTransport):
            raise RuntimeError(f"Export failed: {session.result_cell.payload}")
        export_latencies.append(loop.time() - due)
        completed += 1

    async def short_requests() -> None:
        tick = 0
        while True:
            deadline = start + tick * args.interval
            if exports_done.is_set() and deadline >= stop_at:
                break
            await asyncio.sleep(max(0, deadline - loop.time()))
            if exports_done.is_set() and deadline >= stop_at:
                break
            # This real short Session method acquires another Session's lock
            # and updates its trace. Intended send time includes producer lag.
            ok = await shard.mark_chat_service_response_ready(
                session_id=short_session.session_id,
                request_id="short",
                remote_return_at=0.0,
                response_ready_at=0.0,
                http_return_at=0.0,
            )
            assert ok
            short_latencies.append(loop.time() - deadline)
            tick += 1

    async def ticker() -> None:
        tick = 0
        while True:
            deadline = start + tick * args.interval
            if exports_done.is_set() and deadline >= stop_at:
                break
            await asyncio.sleep(max(0, deadline - loop.time()))
            if exports_done.is_set() and deadline >= stop_at:
                break
            ticker_delays.append(max(0, loop.time() - deadline))
            tick += 1

    probes = [asyncio.create_task(short_requests()), asyncio.create_task(ticker())]
    await asyncio.gather(*(export(session) for session in sessions))
    export_end = loop.time()
    export_wall = export_end - due
    stop_at = max(stop_at, export_end)
    exports_done.set()
    await asyncio.gather(*probes)
    measured_wall = loop.time() - start
    cpu_seconds = time.process_time() - cpu_start
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    offered_short_requests = math.ceil((stop_at - start) / args.interval)
    assert len(short_latencies) == len(ticker_delays) == offered_short_requests
    payloads = [session.result_cell.payload.exports for session in sessions]
    # Outside the measured window, include reconstruction in correctness checks.
    from relax.agentic.pipeline import TrainingFieldArtifact

    digests = [content_digest(payload) for payload in payloads]
    reconstructed_digests = [
        content_digest([TrainingFieldArtifact(item["sample_payload"]).to_sample().to_dict() for item in payload])
        for payload in payloads
    ]
    await shard.shutdown()
    assert not shard._groups and not shard._session_records
    assert not any(t.name.startswith("session-export") for t in threading.enumerate())
    assert not any(isinstance(session.result_cell.payload, RuntimeGroupError) for session in sessions)
    return {
        "mode": args.mode,
        "tokens": args.tokens,
        "concurrency": args.concurrency,
        "offered_exports": args.exports,
        "completed_exports": completed,
        "total_export_tokens": args.exports * args.tokens,
        "export_throughput_per_s": completed / export_wall,
        "export_wall_s": export_wall,
        "measurement_wall_s": measured_wall,
        "cpu_s": cpu_seconds,
        "peak_rss_mib": peak_rss_mib,
        "offered_short_requests": offered_short_requests,
        "probe_schedule_s": stop_at - start,
        "event_loop": distribution(ticker_delays),
        "short_request": distribution(short_latencies),
        "export_request": distribution(export_latencies),
        "digests": digests,
        "reconstructed_digests": reconstructed_digests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=131072)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 131072])
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--exports", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.005)
    parser.add_argument("--min-window", type=float, default=1.0)
    parser.add_argument("--rich", action="store_true", help="Include top-k, R3 and multimodal training inputs")
    parser.add_argument("--json-out", type=Path, default=Path("session-export-results.json"))
    parser.add_argument("--mode", choices=["inline", "thread"], default="thread")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if (
        min(args.tokens, *args.lengths) < 128
        or min(args.exports, args.concurrency, *args.concurrencies, args.repeats) < 1
    ):
        parser.error("Use token lengths >=128 and positive counts")
    if args.interval <= 0 or args.min_window <= 0:
        parser.error("Probe interval and measurement window must be positive")
    if args.child or args.profile:
        result = asyncio.run(run_case(args))
        if not args.profile:
            print(json.dumps(result))
        return

    result = {
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "numpy": np.__version__,
            "cpu_model": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "unknown",
            ),
        },
        "configuration": vars(args) | {"json_out": str(args.json_out)},
        "runs": [],
        "comparisons": [],
    }
    for tokens in args.lengths:
        for concurrency in args.concurrencies:
            case_runs = []
            for repeat in range(args.repeats):
                # Alternate order to limit thermal/background-load bias.
                for mode in ["inline", "thread"] if repeat % 2 == 0 else ["thread", "inline"]:
                    cmd = [
                        sys.executable,
                        "-m",
                        __name__,
                        "--child",
                        "--mode",
                        mode,
                        "--tokens",
                        str(tokens),
                        "--concurrency",
                        str(concurrency),
                        "--exports",
                        str(args.exports),
                        "--interval",
                        str(args.interval),
                        "--min-window",
                        str(args.min_window),
                    ]
                    # __name__ is __main__ when launched as a module.
                    cmd[2] = "scripts.benchmarks.benchmark_session_export"
                    if args.rich:
                        cmd.append("--rich")
                    run = json.loads(subprocess.check_output(cmd, text=True, timeout=300))
                    run["repeat"] = repeat
                    case_runs.append(run)
                    result["runs"].append(run)
                    print(
                        f"{tokens=} {concurrency=} {repeat=} {mode}: "
                        f"loop P99={run['event_loop']['p99_ms']:.2f}ms "
                        f"short P99={run['short_request']['p99_ms']:.2f}ms "
                        f"exports={run['completed_exports']} CPU={run['cpu_s']:.3f}s",
                        flush=True,
                    )
                    args.json_out.parent.mkdir(parents=True, exist_ok=True)
                    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
            expected = case_runs[0]
            assert all(
                run["digests"] == expected["digests"]
                and run["reconstructed_digests"] == expected["reconstructed_digests"]
                and run["completed_exports"] == args.exports
                for run in case_runs
            ), "Work/content mismatch"
            medians = {
                mode: statistics.median(run["event_loop"]["p99_ms"] for run in case_runs if run["mode"] == mode)
                for mode in ("inline", "thread")
            }
            result["comparisons"].append(
                {
                    "tokens": tokens,
                    "concurrency": concurrency,
                    "content_equal": True,
                    "event_loop_p99_reduction_pct": 100 * (1 - medians["thread"] / medians["inline"]),
                }
            )
            args.json_out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
