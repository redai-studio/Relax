# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU benchmark: Agentic Session export vs. event-loop responsiveness.

Long Agentic trajectories finish in waves on one ``AgenticSessionShard`` while
short ``health()`` requests keep arriving at a fixed rate. Each export runs the
production finalize path (``_finalize_terminal_session`` -> Forest sample
assembly -> payload compaction -> trajectory publication -> ``_finish_session``)
on Forests filled through ``append_obs`` / ``append_resp``, which stands in for
the inference backend. Token IDs come from tokenizing this repository's own
Python sources, so decode cost matches real code-agent text.

The parent process runs every ``--root`` checkout in its own fresh child
process, alternating the order between repeats, then checks that all variants
exported identical training content and prints a comparison table.

Metrics (one value per run; the table shows the median and [min, max]):
  * loop lag: a coroutine targets an absolute 1 ms tick; lag is how late each
    tick fires. P99 is nearest-rank over every tick inside the window.
  * short request latency: from the request's scheduled arrival to its
    response, so time spent queued behind a blocked loop is included.
  * export latency: from the wave's scheduled arrival to the Session result
    being published, including waiting for the export worker.
  * export throughput: exports / (last export done - first wave arrival).
  * CPU: ``time.process_time()`` of the whole process over the window.
  * peak RSS: VmRSS sampled every 5 ms over the window.

Usage (from a checkout containing this script; BASE is e.g. a worktree of main):
    git worktree add --detach /tmp/relax-base origin/main
    python scripts/benchmarks/benchmark_session_export.py \\
        --root base=/tmp/relax-base --root optimized=. \\
        --tokenizer /path/to/Qwen3-0.6B --repeats 5 --output export-bench.json

Pin both variants to the same CPUs with ``taskset`` for stable numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


# --------------------------------------------------------------------------- child


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _rss_mib() -> float:
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def _corpus_token_ids(tokenizer: Any, repo_root: Path, minimum: int) -> list[int]:
    ids: list[int] = []
    for path in sorted((repo_root / "relax").rglob("*.py")):
        ids.extend(tokenizer.encode(path.read_text(encoding="utf-8"), add_special_tokens=False))
        if len(ids) >= minimum:
            return ids
    while len(ids) < minimum:
        ids.extend(ids)
    return ids


def _build_forest(service: Any, state: Any, cfg: dict[str, Any], corpus: list[int], offset: int) -> Any:
    def take(count: int) -> list[int]:
        nonlocal offset
        start = offset % (len(corpus) - count)
        offset += count
        return corpus[start : start + count]

    forest = state.SessionForest.create_empty(session_id=f"s{offset}", metadata={"bench": True})
    parent = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "task"}],
        train_token_delta=take(cfg["prompt_tokens"]),
        rollout_token_delta=[],
    ).state_hash
    for turn in range(cfg["turns"]):
        tokens = take(cfg["response_tokens"])
        parent = forest.append_resp(
            parent_state_hash=parent,
            rollout_id=0,
            abort_count=0,
            messages_delta=[{"role": "assistant", "content": f"turn {turn}"}],
            token_delta=tokens,
            logprob_delta=[-((token % 997) / 997.0) for token in tokens],
            weight_version_delta=["7"] * len(tokens),
            export_metadata_patch={"request_id": f"r{turn}"},
        ).state_hash
        if turn + 1 < cfg["turns"]:
            parent = forest.append_obs(
                parent_state_hash=parent,
                rollout_id=0,
                abort_count=0,
                messages_delta=[{"role": "tool", "content": f"observation {turn}"}],
                train_token_delta=take(cfg["observation_tokens"]),
                rollout_token_delta=[],
            ).state_hash
    return forest


def _make_shard(service: Any, tokenizer: Any) -> Any:
    shard = object.__new__(service.AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = SimpleNamespace(agentic_session_lifecycle=False, use_rollout_routing_replay=False)
    shard._groups = {}
    shard._session_records = {}
    shard._train_generation_open = True
    shard._state_revision = 0
    shard._state_changed = asyncio.Event()
    shard._permit_cleanup_tasks = set()
    shard._trajectory_recorder = service.ShardTrajectoryRecorder(SimpleNamespace(), shard_ordinal=0)
    shard._generation_backend = SimpleNamespace(
        tokenizer=tokenizer,
        compiler=SimpleNamespace(apply_chat_template_kwargs={}),
        abort_request=None,
    )
    return shard


def _attach_session(service: Any, shard: Any, forest: Any, group_id: str) -> Any:
    async def terminate_and_join() -> None:
        return None

    cell = service._SessionResultCell()
    group = service.ResidentGroup("train", group_id, {forest.session_id: cell}, [], ownership="runtime")
    process = SimpleNamespace(
        output=SimpleNamespace(reward=1.0, metadata={}, records=()), terminate_and_join=terminate_and_join
    )
    session = service._SessionRecord(group, forest.session_id, {}, cell, forest=forest)
    session.resources = SimpleNamespace(process=process, process_wait=None)
    group.sessions.append(session)
    shard._groups[group_id] = group
    shard._session_records[forest.session_id] = session
    return session


def _digest(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("tokens", "rollout_tokens", "loss_mask", "rollout_log_probs", "weight_versions", "response"):
        value = payload.get(key)
        if hasattr(value, "tolist"):
            value = value.tolist()
        digest.update(key.encode() + json.dumps(value).encode())
    digest.update(payload["prompt"].encode())
    return digest.hexdigest()


async def _measure(service: Any, state: Any, tokenizer: Any, cfg: dict[str, Any], corpus: list[int]) -> dict:
    loop = asyncio.get_running_loop()
    shard = _make_shard(service, tokenizer)
    total = cfg["waves"] * cfg["sessions"]
    sessions = [
        _attach_session(service, shard, _build_forest(service, state, cfg, corpus, i * 7919), f"g{i}")
        for i in range(total + 1)
    ]
    # Warm-up: one export and one short request outside the window.
    await shard._finalize_terminal_session(sessions[-1])
    await shard.health()
    gc.collect()

    start = loop.time() + 0.05
    duration = cfg["duration_s"]
    lags: list[float] = []
    short_latencies: list[float] = []
    export_latencies: list[float] = []
    export_done: list[float] = []
    rss: list[float] = []
    stop = threading.Event()

    def sample_rss() -> None:
        while not stop.wait(0.005):
            rss.append(_rss_mib())

    async def ticker() -> None:
        # Every 1 ms deadline is recorded, including those missed while the
        # loop was blocked, so a long stall weighs as much as it lasted.
        deadline = start
        end = start + duration
        while deadline < end:
            await asyncio.sleep(max(0.0, deadline - loop.time()))
            now = loop.time()
            while deadline <= now and deadline < end:
                lags.append((now - deadline) * 1e3)
                deadline += 0.001

    async def short_request(at: float) -> None:
        await asyncio.sleep(max(0.0, at - loop.time()))
        await shard.health()
        short_latencies.append((loop.time() - at) * 1e3)

    async def export(session: Any, at: float) -> None:
        await asyncio.sleep(max(0.0, at - loop.time()))
        await shard._finalize_terminal_session(session)
        done = loop.time()
        export_latencies.append((done - at) * 1e3)
        export_done.append(done)

    interval = 1.0 / cfg["short_rate"]
    wave_gap = duration / cfg["waves"]
    work = [short_request(start + i * interval) for i in range(int(duration * cfg["short_rate"]))]
    work += [
        export(sessions[wave * cfg["sessions"] + i], start + wave * wave_gap)
        for wave in range(cfg["waves"])
        for i in range(cfg["sessions"])
    ]
    sampler = threading.Thread(target=sample_rss, daemon=True)
    rss_before = _rss_mib()
    cpu_before = time.process_time()
    sampler.start()
    await asyncio.gather(ticker(), *work)
    cpu = time.process_time() - cpu_before
    stop.set()
    sampler.join()

    digests = []
    for session in sessions[:total]:
        transport = session.result_cell.payload
        assert isinstance(transport, service.SessionExportTransport), transport
        digests.append(_digest(transport.exports[0]["sample_payload"]))
    tokens = len(sessions[0].result_cell.payload.exports[0]["sample_payload"]["tokens"])
    executor = getattr(shard, "_export_executor", None)
    if executor is not None:
        executor.shutdown(wait=True)
    return {
        "trajectory_tokens": tokens,
        "exports_completed": len(export_done),
        "short_requests_completed": len(short_latencies),
        "loop_lag_ms": {"p50": _percentile(lags, 0.5), "p99": _percentile(lags, 0.99), "max": max(lags)},
        "short_request_ms": {
            "p50": _percentile(short_latencies, 0.5),
            "p99": _percentile(short_latencies, 0.99),
            "max": max(short_latencies),
        },
        "export_ms": {"p50": _percentile(export_latencies, 0.5), "p99": _percentile(export_latencies, 0.99)},
        "export_throughput_per_s": len(export_done) / (max(export_done) - start),
        "cpu_s": cpu,
        "peak_rss_delta_mib": max(rss, default=rss_before) - rss_before,
        "content_digest": hashlib.sha256("".join(digests).encode()).hexdigest(),
    }


def _child(args: argparse.Namespace) -> None:
    sys.path.insert(0, args.child_root)
    from transformers import AutoTokenizer

    import relax.agentic.pipeline as pipeline
    from relax.agentic.session import service, state

    cfg = json.loads(args.child_config)
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    per_session = cfg["prompt_tokens"] + cfg["turns"] * (cfg["response_tokens"] + cfg["observation_tokens"])
    corpus = _corpus_token_ids(tokenizer, Path(cfg["corpus_root"]), per_session * 4)
    result = asyncio.run(_measure(service, state, tokenizer, cfg, corpus))
    result["source_sha256"] = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()[:16]
        for name, module in (("service", service), ("state", state), ("pipeline", pipeline))
    }
    print(json.dumps(result))


# -------------------------------------------------------------------------- parent

_SUMMARY = (
    ("loop lag P99 (ms)", ("loop_lag_ms", "p99")),
    ("loop lag max (ms)", ("loop_lag_ms", "max")),
    ("short request P50 (ms)", ("short_request_ms", "p50")),
    ("short request P99 (ms)", ("short_request_ms", "p99")),
    ("export latency P50 (ms)", ("export_ms", "p50")),
    ("export latency P99 (ms)", ("export_ms", "p99")),
    ("export throughput (/s)", ("export_throughput_per_s",)),
    ("process CPU (s)", ("cpu_s",)),
    ("peak RSS delta (MiB)", ("peak_rss_delta_mib",)),
)


def _value(run: dict[str, Any], path: tuple[str, ...]) -> float:
    for key in path:
        run = run[key]
    return float(run)


def _parent(args: argparse.Namespace) -> None:
    roots = dict(spec.split("=", 1) for spec in args.root)
    cfg = {
        "tokenizer": args.tokenizer,
        # Every variant tokenizes the same sources, so workloads are identical.
        "corpus_root": os.path.abspath(next(iter(roots.values()))),
        "turns": args.turns,
        "prompt_tokens": args.prompt_tokens,
        "response_tokens": args.response_tokens,
        "observation_tokens": args.observation_tokens,
        "sessions": args.sessions,
        "waves": args.waves,
        "duration_s": args.duration,
        "short_rate": args.short_rate,
    }
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
    runs: dict[str, list[dict[str, Any]]] = {name: [] for name in roots}
    for repeat in range(args.repeats):
        order = list(roots) if repeat % 2 == 0 else list(reversed(roots))
        for name in order:
            command = [sys.executable, os.path.abspath(__file__), "--child-root", os.path.abspath(roots[name])]
            output = subprocess.run(
                [*command, "--child-config", json.dumps(cfg)], env=env, check=True, capture_output=True, text=True
            ).stdout
            run = json.loads(output.strip().splitlines()[-1])
            runs[name].append(run)
            print(f"repeat {repeat} {name}: loop P99 {run['loop_lag_ms']['p99']:.2f} ms", file=sys.stderr)

    digests = {run["content_digest"] for name in runs for run in runs[name]}
    expected_exports = args.sessions * args.waves
    expected_short = int(args.duration * args.short_rate)
    for name, name_runs in runs.items():
        for run in name_runs:
            if run["exports_completed"] != expected_exports or run["short_requests_completed"] != expected_short:
                raise SystemExit(f"{name}: a run did not complete every scheduled request")
    if len(digests) != 1:
        raise SystemExit(f"training content differs between variants: {sorted(digests)}")

    names = list(roots)
    tokens = runs[names[0]][0]["trajectory_tokens"]
    print(
        f"{args.waves} waves x {args.sessions} concurrent exports of {tokens} tokens, "
        f"{args.short_rate:g} short req/s for {args.duration:g} s; {args.repeats} runs per variant; "
        "identical training content across variants."
    )
    header = f"| metric | {' | '.join(names)} | change |"
    print(header)
    print("|" + " --- |" * (len(names) + 2))
    for label, path in _SUMMARY:
        cells, medians = [], []
        for name in names:
            values = [_value(run, path) for run in runs[name]]
            medians.append(statistics.median(values))
            cells.append(f"{medians[-1]:.2f} [{min(values):.2f}, {max(values):.2f}]")
        change = (medians[-1] - medians[0]) / medians[0] * 100 if medians[0] else float("nan")
        print(f"| {label} | {' | '.join(cells)} | {change:+.1f}% |")
    if args.output:
        Path(args.output).write_text(json.dumps({"config": cfg, "runs": runs}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", action="append", default=[], help="NAME=PATH of a Relax checkout (repeatable)")
    parser.add_argument("--tokenizer", help="HF tokenizer directory or name")
    parser.add_argument("--turns", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--response-tokens", type=int, default=2048)
    parser.add_argument("--observation-tokens", type=int, default=512)
    parser.add_argument("--sessions", type=int, default=8, help="exports finishing together in each wave")
    parser.add_argument("--waves", type=int, default=4)
    parser.add_argument("--duration", type=float, default=8.0, help="seconds of short-request traffic")
    parser.add_argument("--short-rate", type=float, default=200.0, help="short requests per second")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", help="write every run as JSON")
    parser.add_argument("--child-root", help=argparse.SUPPRESS)
    parser.add_argument("--child-config", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_root:
        _child(args)
    elif len(args.root) < 1 or not args.tokenizer:
        parser.error("--root and --tokenizer are required")
    else:
        _parent(args)


if __name__ == "__main__":
    main()
