# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU Session export benchmark with real assembly and a simulated backend.

Run from the repository root with PYTHONPATH=.; model weights and GPUs are not
needed. The default tokenizer is real, but inference and agent processes are
simulated. Each measured repetition runs in a fresh subprocess. See
docs/benchmarks/session-export.md for workload, timing, and interpretation
details.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import selectors
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any


try:
    from process_supervisor import ManagedProcess
except ModuleNotFoundError:
    from scripts.benchmarks.process_supervisor import ManagedProcess


EVENT_PREFIX = "SESSION_EXPORT_BENCH "
ROOT = Path(__file__).resolve().parents[2]


def emit(event: str, **payload: Any) -> None:
    print(EVENT_PREFIX + json.dumps({"event": event, **payload}), flush=True)


def distribution(values: list[float]) -> dict[str, float | int]:
    """Nearest-rank quantiles, with durations represented in milliseconds."""
    ordered = sorted(values)
    if not ordered:
        raise RuntimeError("No observations collected; this run cannot establish latency")

    def percentile(q: float) -> float:
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)] * 1000

    return {
        "count": len(ordered),
        "mean_ms": statistics.fmean(ordered) * 1000,
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": ordered[-1] * 1000,
    }


class SyntheticTokenizer:
    """Diagnostic-only fallback; never label its results as real-tokenizer
    data."""

    def encode(self, text: str, **_kwargs: Any) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return bytes(token_id % 256 for token_id in token_ids).decode("utf-8", errors="replace")


def load_tokenizer(name: str) -> Any:
    if name == "synthetic":
        return SyntheticTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, trust_remote_code=False)


def repeated_tokens(tokenizer: Any, count: int) -> list[int]:
    chunk = tokenizer.encode(
        "Inspect the evidence, call the search tool, and explain the result. ", add_special_tokens=False
    )
    if not chunk:
        raise RuntimeError("Tokenizer returned an empty test sequence")
    return (chunk * math.ceil(count / len(chunk)))[:count]


def make_forest(tokenizer: Any, config: dict[str, Any], index: int) -> Any:
    """Use the production append/lineage representation outside the timed
    region."""
    import numpy as np

    from relax.agentic.session.state import SessionForest

    forest = SessionForest.create_empty(
        session_id=f"bench-{index}",
        group_index=index,
        index=index,
        metadata={"benchmark": True, "context": {"tags": ["search", "multi-turn"]}},
        train_metadata={"source": "session-export-benchmark", "sample": index},
    )
    parent = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "Research the question and support the conclusion."}],
        train_token_delta=repeated_tokens(tokenizer, config["prompt_tokens"]),
        rollout_token_delta=repeated_tokens(tokenizer, config["prompt_tokens"]),
    )
    response_text = tokenizer.decode(repeated_tokens(tokenizer, config["tokens_per_turn"]), skip_special_tokens=False)
    observation_text = tokenizer.decode(
        repeated_tokens(tokenizer, config["observation_tokens"]), skip_special_tokens=False
    )
    for turn in range(config["turns"]):
        tokens = repeated_tokens(tokenizer, config["tokens_per_turn"])
        topk = config["topk"]
        parent = forest.append_resp(
            parent_state_hash=parent.state_hash,
            rollout_id=0,
            abort_count=0,
            messages_delta=[{"role": "assistant", "content": response_text}],
            token_delta=tokens,
            logprob_delta=[-0.125 - (position % 13) / 100 for position in range(len(tokens))],
            topk_token_ids_delta=(np.tile(np.arange(topk, dtype=np.int32), (len(tokens), 1)) if topk else None),
            topk_logprob_delta=(np.full((len(tokens), topk), -0.25, dtype=np.float32) if topk else None),
            weight_version_delta=["policy-0"],
            generation_elapsed_s=0.001,
            wall_elapsed_s=0.002,
            export_metadata_patch={"turn": turn},
        )
        if turn + 1 < config["turns"]:
            observations = repeated_tokens(tokenizer, config["observation_tokens"])
            parent = forest.append_obs(
                parent_state_hash=parent.state_hash,
                rollout_id=0,
                abort_count=0,
                messages_delta=[{"role": "user", "content": observation_text}],
                train_token_delta=observations,
                rollout_token_delta=list(observations),
            )
    return forest


def source_manifest(source_root: Path, modules: tuple[Any, ...]) -> dict[str, Any]:
    """Verify checkout isolation and fingerprint the actual production
    inputs."""
    for module in modules:
        loaded = Path(module.__file__).resolve()
        if not loaded.is_relative_to(source_root):
            raise RuntimeError(f"Mixed checkout import: {loaded}; expected source under {source_root}")
    paths = [
        "relax/agentic/session/service.py",
        "relax/agentic/session/state.py",
        "relax/agentic/pipeline/__init__.py",
        "relax/utils/types.py",
        "relax/agentic/session/export_executor.py",
    ]
    return {
        "root": str(source_root),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip(),
        "git_diff_stat": subprocess.check_output(["git", "diff", "--stat"], cwd=source_root, text=True).strip(),
        "files_sha256": {
            path: hashlib.sha256((source_root / path).read_bytes()).hexdigest()
            for path in paths
            if (source_root / path).is_file()
        },
        "loaded_modules": {module.__name__: str(Path(module.__file__).resolve()) for module in modules},
    }


def content_digest(exports: dict[str, Any]) -> dict[str, str]:
    """Compare actual training values after timing; timestamp metadata is
    excluded."""
    import numpy as np

    fields = (
        "tokens",
        "rollout_tokens",
        "loss_mask",
        "rollout_log_probs",
        "rollout_topk_token_ids",
        "rollout_topk_log_probs",
        "rollout_routed_experts",
        "custom_advantage",
        "reward",
        "response",
        "response_length",
        "prompt",
        "status",
        "weight_versions",
        "_rollout_tokens_shared",
    )

    def update(digest: Any, value: Any) -> None:
        if isinstance(value, np.ndarray):
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
        elif isinstance(value, (list, tuple)):
            digest.update(b"[")
            for item in value:
                update(digest, item)
                digest.update(b",")
            digest.update(b"]")
        elif isinstance(value, dict):
            for key in sorted(value):
                update(digest, key)
                update(digest, value[key])
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())

    results = {}
    for session_id, transport in sorted(exports.items()):
        digest = hashlib.sha256()
        for exported in transport.exports:
            payload = exported["sample_payload"]
            for field in fields:
                digest.update(field.encode())
                update(digest, payload.get(field))
        results[session_id] = digest.hexdigest()
    return results


async def run_worker(config: dict[str, Any]) -> dict[str, Any]:
    import httpx
    from fastapi import FastAPI

    from relax.agentic import pipeline as pipeline_module
    from relax.agentic.session import service as service_module
    from relax.agentic.session import state as state_module
    from relax.agentic.session.service import ResidentGroup, _SessionRecord, _SessionResultCell
    from relax.utils import types as sample_module

    source_root = Path(config.get("source_root", ROOT)).resolve()
    production_modules = (service_module, state_module, pipeline_module, sample_module)
    production_sources = source_manifest(source_root, production_modules)
    tokenizer = load_tokenizer(config["tokenizer"])
    dependency_versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "ray", "transformers", "numpy", "httpx", "fastapi")
    }
    shard_cls = service_module.AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    shard.args = types.SimpleNamespace(use_rollout_routing_replay=False)
    shard._generation_backend = types.SimpleNamespace(
        tokenizer=tokenizer,
        compiler=types.SimpleNamespace(apply_chat_template_kwargs={}),
    )
    shard._session_records = {}
    shard._groups = {}
    if (source_root / "relax/agentic/session/export_executor.py").is_file():
        # A pristine original checkout has no export executor and must call its
        # actual synchronous finalizer.
        from relax.agentic.session.export_executor import SessionExportExecutor

        shard._export_executor = SessionExportExecutor()

    outputs: dict[str, Any] = {}

    async def finish_session(session: Any, outcome: Any) -> None:
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome is None:
            raise RuntimeError(f"Session {session.session_id} was dropped")
        outputs[session.session_id] = outcome

    shard._finish_session = finish_session
    sessions = []
    for index in range(config["exports"]):
        forest = make_forest(tokenizer, config, index)
        cell = _SessionResultCell()
        group = ResidentGroup(rollout_mode="train", group_id=f"group-{index}", result_cells={}, sessions=[])
        session = _SessionRecord(
            group=group, session_id=forest.session_id, session_sampling_params={}, result_cell=cell
        )
        session.forest = forest
        session.resources = types.SimpleNamespace(
            process=types.SimpleNamespace(output=types.SimpleNamespace(reward=1.0, metadata={}, records=())),
        )
        group.sessions.append(session)
        group.result_cells[session.session_id] = cell
        shard._groups[group.group_id] = group
        shard._session_records[session.session_id] = session
        sessions.append(session)

    # Warm the real assembler, tokenizer, and any export worker thread in the
    # process being measured. Reset only this local benchmark fixture afterward.
    await shard._finalize_terminal_session(sessions[0])
    sessions[0].phase = service_module.SessionPhase.ACTIVE
    outputs.clear()

    if config.get("profile"):
        # Python 3.12 profiling from a background worker can collect frames from
        # the concurrently running event loop. Diagnose assembly synchronously
        # on the main thread, before the timed workload and without replacing
        # any production method. These trials remain diagnostic-only.
        profile = cProfile.Profile()

        def profile_assembly() -> None:
            for session in sessions:
                output = session.resources.process.output
                now = time.time()
                shard._build_session_export_transport(
                    session=session,
                    reward=output.reward,
                    metadata=output.metadata,
                    output_records=output.records,
                    finalize_events={
                        "finalize_arrive_at": now,
                        "finalize_lock_acquired_at": now,
                        "finalize_start_at": now,
                    },
                )

        profile.runcall(profile_assembly)
        profile.dump_stats(config["profile"])

    app = FastAPI()
    admission = asyncio.Semaphore(config["concurrency"])

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return await shard.health()

    @app.post("/export/{index}")
    async def export(index: int) -> dict[str, Any]:
        async with admission:
            await shard._finalize_terminal_session(sessions[index])
        return {"session_id": sessions[index].session_id, "completed": True}

    short_latencies: list[float] = []
    export_latencies: list[float] = []
    loop_delays: list[float] = []
    stop = asyncio.Event()
    source_cutoff: float | None = None
    short_tasks: list[asyncio.Task[None]] = []
    loop = asyncio.get_running_loop()
    probe_period = config["probe_ms"] / 1000
    short_period = 1 / config["short_rate"]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://session-benchmark"
    ) as client:
        await client.get("/health")  # HTTP/app dispatch warmup, outside the measurement.
        start = loop.time()

        async def short_request(scheduled: float) -> None:
            response = await client.get("/health")
            response.raise_for_status()
            short_latencies.append(loop.time() - scheduled)

        async def short_source() -> None:
            index = 0
            while index < config["short_requests"]:
                now = loop.time()
                while index < config["short_requests"] and start + index * short_period <= now:
                    short_tasks.append(asyncio.create_task(short_request(start + index * short_period)))
                    index += 1
                if index < config["short_requests"]:
                    await asyncio.sleep(max(0, start + index * short_period - loop.time()))
            await asyncio.gather(*short_tasks)

        async def probe() -> None:
            deadline = start + probe_period
            while True:
                await asyncio.sleep(max(0, deadline - loop.time()))
                now = loop.time() if source_cutoff is None else source_cutoff
                # Record every missed nominal tick, rather than resetting the
                # deadline after stalls (which would hide coordinated omission).
                while deadline <= now:
                    loop_delays.append(now - deadline)
                    deadline += probe_period
                if stop.is_set():
                    break

        scheduled_export = start + config["lead_in_ms"] / 1000

        async def export_request(index: int) -> None:
            await asyncio.sleep(max(0, scheduled_export - loop.time()))
            response = await client.post(f"/export/{index}")
            response.raise_for_status()
            export_latencies.append(loop.time() - scheduled_export)

        cpu_start = time.process_time()
        emit("measurement_start", monotonic=start, rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        sources = [asyncio.create_task(short_source()), asyncio.create_task(probe())]
        try:
            await asyncio.gather(*(export_request(index) for index in range(config["exports"])))
            # Use the last recorded HTTP response timestamp. The coordinator
            # may resume later after other ready short-request callbacks.
            exports_end = scheduled_export + max(export_latencies)
            await sources[0]
            # Allow already due shorts/probes to run once after the final export
            # before stopping. Every variant uses the same fixed drain interval.
            await asyncio.sleep(config["drain_ms"] / 1000)
            source_cutoff = loop.time()
            stop.set()
            await asyncio.gather(*sources)
            await asyncio.gather(*short_tasks)
            end = loop.time()
            cpu_elapsed = time.process_time() - cpu_start
            result = {
                "config": config,
                "dependency_versions": dependency_versions,
                "source_manifest": production_sources,
                "completed_exports": len(outputs),
                "completed_short_requests": len(short_latencies),
                "planned_short_requests": config["short_requests"],
                "planned_short_arrival_span_s": (config["short_requests"] - 1) * short_period,
                "export_short_overlap_s": max(
                    0.0, min(exports_end, start + (config["short_requests"] - 1) * short_period) - scheduled_export
                ),
                "measurement_wall_s": end - start,
                "export_batch_wall_s": exports_end - scheduled_export,
                "exports_per_s": len(outputs) / (exports_end - scheduled_export),
                "cpu_time_s": cpu_elapsed,
                "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "event_loop_delay": distribution(loop_delays),
                "short_request_latency": distribution(short_latencies),
                "export_request_latency": distribution(export_latencies),
            }
            if len(outputs) != config["exports"] or len(short_latencies) != config["short_requests"]:
                raise RuntimeError("Not all requested exports/short requests completed")
            emit("measurement_end")
        finally:
            stop.set()
            for task in sources:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*sources, return_exceptions=True)
            if hasattr(shard, "_export_executor"):
                await shard._export_executor.close()
    result["training_content_sha256"] = content_digest(outputs)
    if source_manifest(source_root, production_modules)["files_sha256"] != production_sources["files_sha256"]:
        raise RuntimeError("Production source changed while this worker was running")
    result["background_threads_after_close"] = [
        thread.name for thread in __import__("threading").enumerate() if "export" in thread.name.lower()
    ]
    return result


def rss_kib(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def isolated_run(config: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Sample the real worker; the dedicated supervisor owns its
    descendants."""
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-config", json.dumps(config)]
    source_root = Path(config.get("source_root", ROOT)).resolve()
    worker_env = dict(os.environ)
    worker_env["PYTHONPATH"] = str(source_root)
    with ManagedProcess(command, cwd=source_root, env=worker_env, startup_timeout=min(timeout, 10.0)) as process:
        assert process.stdout is not None and process.stderr is not None
        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            measured = False
            peak = 0
            result = None
            errors: list[str] = []
            buffers = {"stdout": b"", "stderr": b""}
            deadline = time.monotonic() + timeout
            while selector.get_map():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Benchmark subprocess exceeded {timeout}s: {config}")
                if measured:
                    peak = max(peak, rss_kib(process.pid))
                for key, _mask in selector.select(timeout=0.01):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    name = key.data
                    buffers[name] += chunk
                    while b"\n" in buffers[name]:
                        line, buffers[name] = buffers[name].split(b"\n", 1)
                        text = line.decode(errors="replace")
                        if name == "stdout" and text.startswith(EVENT_PREFIX):
                            event = json.loads(text[len(EVENT_PREFIX) :])
                            if event["event"] == "measurement_start":
                                measured = True
                                peak = rss_kib(process.pid)
                            elif event["event"] == "measurement_end":
                                peak = max(peak, rss_kib(process.pid))
                                measured = False
                            elif event["event"] == "result":
                                result = event["result"]
                        elif text:
                            errors.append(text)
            return_code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
            if return_code != 0 or result is None:
                raise RuntimeError(f"Worker failed ({return_code}):\n" + "\n".join(errors[-40:]))
            result["measured_peak_rss_kib"] = peak
            result["supervision"] = {
                "worker_pid": process.pid,
                "supervisor_pid": process.supervisor_pid,
                "reaped_descendants": process.reaped_descendants,
                "additional_supervisor_processes": 1,
                "rss_and_cpu_scope": "worker only; dedicated supervisor excluded",
            }
            return result
        finally:
            selector.close()


def summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "event_loop_p99_ms": lambda row: row["event_loop_delay"]["p99_ms"],
        "short_request_p99_ms": lambda row: row["short_request_latency"]["p99_ms"],
        "export_request_p99_ms": lambda row: row["export_request_latency"]["p99_ms"],
        "exports_per_s": lambda row: row["exports_per_s"],
        "cpu_time_s": lambda row: row["cpu_time_s"],
        "peak_rss_mib": lambda row: row["measured_peak_rss_kib"] / 1024,
    }
    return {
        name: {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for name, getter in metrics.items()
        if (values := [getter(result) for result in results])
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="+",
        metavar="NAME=CHECKOUT",
        help="Compare native production code from independent checkouts",
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--turns", type=int, nargs="+", default=[8, 64])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--exports", type=int, default=16)
    parser.add_argument("--tokens-per-turn", type=int, default=2048)
    parser.add_argument("--observation-tokens", type=int, default=512)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--short-rate", type=float, default=100.0)
    parser.add_argument("--short-requests", type=int, default=500)
    parser.add_argument("--probe-ms", type=float, default=2.0)
    parser.add_argument("--lead-in-ms", type=float, default=100.0)
    parser.add_argument("--drain-ms", type=float, default=50.0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--json-out", type=Path, default=Path("session-export-results.json"))
    parser.add_argument("--profile", type=Path, help="Separate profiling run only; adds timing overhead")
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        config = json.loads(args.worker_config)
        # Put the requested checkout ahead of the shared harness directory,
        # before the first Relax import. Each worker starts in a fresh process.
        sys.path.insert(0, str(Path(config.get("source_root", ROOT)).resolve()))
        emit("result", result=asyncio.run(run_worker(config)))
        return
    for name in (
        "exports",
        "tokens_per_turn",
        "prompt_tokens",
        "short_rate",
        "short_requests",
        "probe_ms",
        "runs",
        "timeout",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if any(value <= 0 for value in args.turns + args.concurrency):
        parser.error("--turns and --concurrency must contain positive integers")
    if min(args.observation_tokens, args.topk, args.warmups, args.lead_in_ms, args.drain_ms) < 0:
        parser.error("Token counts, warmups, top-k and scheduling intervals must be nonnegative")
    variants: list[dict[str, str]] = []
    if args.variants:
        for value in args.variants:
            if "=" not in value:
                parser.error("Each --variants entry must be NAME=CHECKOUT")
            name, checkout = value.split("=", 1)
            if re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
                parser.error("Variant names may contain only letters, digits, underscore, and hyphen")
            source_root = Path(checkout).expanduser().resolve()
            if not (source_root / "relax/agentic/session/service.py").is_file():
                parser.error(f"Not a Relax source checkout: {source_root}")
            if any(spec["name"] == name for spec in variants):
                parser.error(f"Duplicate variant name: {name}")
            variants.append({"name": name, "source_root": str(source_root)})
    else:
        variants = [{"name": "current", "source_root": str(ROOT)}]
    # The shared local tokenizer must resolve identically after changing cwd.
    if Path(args.tokenizer).exists():
        args.tokenizer = str(Path(args.tokenizer).resolve())
    if args.profile:
        args.profile = args.profile.resolve()
        args.profile.parent.mkdir(parents=True, exist_ok=True)
    labels = [spec["name"] for spec in variants]
    report: dict[str, Any] = {
        "schema_version": 2,
        "diagnostic_only": bool(args.profile),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "torch_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_diff_stat": subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True).strip(),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "supervisor_sha256": hashlib.sha256(
                Path(__file__).with_name("process_supervisor.py").read_bytes()
            ).hexdigest(),
        },
        "variants": variants,
        "method": {
            "transport": "in-process HTTPX ASGI; real shard.health and finalize, simulated generation/process cleanup",
            "source_isolation": "fresh worker per run, checkout cwd and unique PYTHONPATH; loaded production module paths and SHA256 verified",
            "quantile": "nearest-rank ceil(0.99*N); one sample per absolute scheduled probe tick",
            "short_latency": "planned arrival to HTTP response, including scheduler/admission queue time",
            "export_latency": "batch planned arrival to HTTP response, including admission and worker queue time",
            "throughput": "completed exports / maximum recorded export response latency; coordinator resume time excluded",
            "window": "fixed short-arrival schedule and fixed export batch both completed, plus fixed drain",
            "rss": "parent samples worker VmRSS every 10ms during measurement; ru_maxrss also reported",
            "warmup": "unreported full-workload worker runs; each measured worker also finalizes one same-size session and warms health",
            "content": "post-measurement digest of training fields, excluding wall-clock trace metadata",
            "profiling": "optional synchronous main-thread production assembly after warmup and before timing; profile trials are diagnostic-only",
        },
        "cases": [],
    }
    for turns in args.turns:
        for concurrency in args.concurrency:
            base = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key
                not in {"worker_config", "json_out", "turns", "concurrency", "runs", "warmups", "timeout", "variants"}
            }
            base.update(turns=turns, concurrency=concurrency)
            measurements: dict[str, list[dict[str, Any]]] = {name: [] for name in labels}
            for iteration in range(args.warmups + args.runs):
                # Rotate start position: with three variants and three runs,
                # every variant occupies each machine-load slot exactly once.
                offset = iteration % len(variants)
                order = variants[offset:] + variants[:offset]
                for spec in order:
                    name = spec["name"]
                    config = {**base, "variant": name, "source_root": spec["source_root"]}
                    if args.profile:
                        config["profile"] = str(
                            args.profile.with_name(
                                f"{args.profile.stem}-{turns}-{concurrency}-{name}-{iteration}.prof"
                            )
                        )
                    result = isolated_run(config, args.timeout)
                    warmup = iteration < args.warmups
                    print(
                        json.dumps(
                            {
                                "turns": turns,
                                "concurrency": concurrency,
                                "variant": name,
                                "warmup": warmup,
                                "run": iteration - args.warmups,
                                "loop_p99_ms": result["event_loop_delay"]["p99_ms"],
                                "short_p99_ms": result["short_request_latency"]["p99_ms"],
                                "exports_per_s": result["exports_per_s"],
                                "cpu_time_s": result["cpu_time_s"],
                            }
                        ),
                        flush=True,
                    )
                    if not warmup:
                        measurements[name].append(result)
            expected_digest = next(iter(measurements.values()))[0]["training_content_sha256"]
            if any(
                row["training_content_sha256"] != expected_digest for rows in measurements.values() for row in rows
            ):
                raise RuntimeError("Training content differs across variants/repetitions")
            for name, rows in measurements.items():
                fingerprints = [row["source_manifest"]["files_sha256"] for row in rows]
                if any(value != fingerprints[0] for value in fingerprints):
                    raise RuntimeError(f"Production source changed between repetitions: {name}")
            case = {
                "turns": turns,
                "concurrency": concurrency,
                "training_content_equal": True,
                "results": measurements,
                "summary": {name: summary(rows) for name, rows in measurements.items()},
                "comparisons": {},
            }
            for index, reference in enumerate(labels):
                before = case["summary"][reference]
                for name in labels[index + 1 :]:
                    after = case["summary"][name]
                    case["comparisons"][f"{name}_vs_{reference}"] = {
                        "event_loop_p99_reduction_pct": (
                            1 - after["event_loop_p99_ms"]["median"] / before["event_loop_p99_ms"]["median"]
                        )
                        * 100,
                        "short_request_p99_reduction_pct": (
                            1 - after["short_request_p99_ms"]["median"] / before["short_request_p99_ms"]["median"]
                        )
                        * 100,
                        "exports_per_s_change_pct": (
                            after["exports_per_s"]["median"] / before["exports_per_s"]["median"] - 1
                        )
                        * 100,
                        "cpu_time_change_pct": (after["cpu_time_s"]["median"] / before["cpu_time_s"]["median"] - 1)
                        * 100,
                        "peak_rss_change_mib": after["peak_rss_mib"]["median"] - before["peak_rss_mib"]["median"],
                    }
            report["cases"].append(case)
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Results written to {args.json_out.resolve()}")


if __name__ == "__main__":
    main()
