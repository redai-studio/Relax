# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Replay pinned text trajectories through actual Session chat and
finalization.

This CPU-only benchmark uses the production message compiler and a mock
backend. Token IDs are reconstructed with a declared tokenizer; logprobs are
synthetic. Preparation replays all but the last assistant turn outside the
timed window. The timed window includes the last request, backend cohort,
export and cleanup. No recorded tool command is executed. Structured assistant
tool calls require a separate parser adapter and are rejected here rather than
flattened to text.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import random
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from relax.agentic.pipeline import SessionExportTransport, SessionSpec, TrainingFieldArtifact
from relax.agentic.pipeline.runtime import BackendGenerateResult, SGLangMessageCompiler
from relax.agentic.runner import SessionOutput
from relax.agentic.session.service import AgenticSessionShard
from relax.agentic.session.state import check_messages


@dataclass
class ReplayCase:
    row: int
    messages: list[dict[str, Any]]
    assistant_positions: list[int]
    response_tokens: list[list[int]]

    def describe(self) -> dict[str, Any]:
        return {
            "source_row": self.row,
            "message_count": len(self.messages),
            "assistant_turns": len(self.assistant_positions),
            "response_tokens": [len(ids) for ids in self.response_tokens],
            "total_response_tokens": sum(map(len, self.response_tokens)),
            "messages_sha256": hashlib.sha256(
                json.dumps(self.messages, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
        }


def load_cases(args: argparse.Namespace, tokenizer: Any) -> tuple[list[ReplayCase], dict[str, Any]]:
    import pyarrow.parquet as pq

    manifest = json.loads((args.corpus / "provenance.json").read_text())
    source = next(row for row in manifest["files"] if row["file"] == "swegym-openhands.parquet")
    path = args.corpus / source["file"]
    if source["revision"] != "4aaa5a4a4b5861f4799d2336908760c190ac3b17":
        raise ValueError("Unexpected SWE-Gym source revision")
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("Trajectory file differs from its cloud import manifest")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != 491:
        raise ValueError("Unexpected SWE-Gym row count")
    eligible: list[ReplayCase] = []
    rejected: list[dict[str, Any]] = []
    source_row = 0
    for batch in parquet.iter_batches(batch_size=32, columns=["messages"]):
        for row in batch.to_pylist():
            try:
                raw = row["messages"]
                if not raw or any(set(message) != {"role", "content"} for message in raw):
                    raise ValueError("Only the published role/content message shape is supported")
                if any(not isinstance(message["content"], str) for message in raw):
                    raise ValueError(
                        "Content parts need a separately verified adapter; this protocol preserves strings"
                    )
                messages = check_messages(raw)
                if messages != raw:
                    raise ValueError("Canonical normalization would change the recorded message order or contents")
                if not messages or messages[-1]["role"] != "assistant":
                    raise ValueError("A complete replay must end with its recorded assistant response")
                positions = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
                if len(positions) < 2 or positions[0] == 0:
                    raise ValueError("A multi-turn replay needs observations and at least two responses")
                tokens: list[list[int]] = []
                for position in positions:
                    message = messages[position]
                    before, after = check_messages(raw[:position]), check_messages(raw[: position + 1])
                    if after != before + [message] or before != messages[:position]:
                        raise ValueError("Canonical normalization would change a causal source prefix")
                    if set(message) != {"role", "content"} or not isinstance(message["content"], str):
                        raise ValueError("Reasoning or structured responses require a separate verified adapter")
                    ids = list(tokenizer.encode(message["content"], add_special_tokens=False))
                    # Match the exact production decode call, including its default cleanup behavior.
                    if not ids or tokenizer.decode(ids, skip_special_tokens=False) != message["content"]:
                        raise ValueError("The selected tokenizer does not exactly round-trip a recorded response")
                    tokens.append(ids)
                eligible.append(ReplayCase(source_row, messages, positions, tokens))
            except (TypeError, ValueError) as error:
                rejected.append({"source_row": source_row, "reason": str(error)})
            source_row += 1
    ordered = sorted(eligible, key=lambda case: (sum(map(len, case.response_tokens)), case.row))
    if len(ordered) < 2 * args.exports:
        raise ValueError("Not enough distinct eligible trajectories for the fixed short/long selections")
    selected = ordered[: args.exports] if args.length == "short" else ordered[-args.exports :]
    return selected, {
        "source": source,
        "eligible_count": len(eligible),
        "rejected": rejected,
        "selection": "lowest/highest total reconstructed response token counts; ties by source row",
        "cases": [case.describe() for case in selected],
    }


@dataclass
class Cohort:
    size: int
    backend_arrivals: int = 0
    ready_arrivals: int = 0
    backend_ready: asyncio.Event = field(default_factory=asyncio.Event)
    export_ready: asyncio.Event = field(default_factory=asyncio.Event)
    processes: list[Any] = field(default_factory=list)
    backend_session_order: list[str] = field(default_factory=list)

    async def backend_enter(self, session_id: str) -> None:
        self.backend_session_order.append(session_id)
        self.backend_arrivals += 1
        if self.backend_arrivals == self.size:
            self.backend_ready.set()
        await self.backend_ready.wait()

    async def finish_ready(self, process: Any) -> None:
        self.processes.append(process)
        self.ready_arrivals += 1
        if self.ready_arrivals == self.size:
            # Signal all real Session watchers together, outside their CPU builder.
            for member in self.processes:
                member.finished.set()
            self.export_ready.set()
        await self.export_ready.wait()


class MockAgentProcess:
    """An asynchronous process capability, explicitly not an OS child
    process."""

    def __init__(self, row: int) -> None:
        self.finished = asyncio.Event()
        self.output = SessionOutput(reward=1.0, metadata={"source_row": row}, records=())
        self.terminate_calls = 0
        self.timeout_active = False

    async def wait(self) -> None:
        await self.finished.wait()

    async def set_timeout_active(self, active: bool) -> None:
        self.timeout_active = active

    async def terminate_and_join(self) -> None:
        self.terminate_calls += 1
        self.finished.set()


class MockAgentLauncher:
    def __init__(self, cases: dict[str, ReplayCase]) -> None:
        self.cases = cases
        self.processes: dict[str, MockAgentProcess] = {}

    def start_agent(self, session_input: Any) -> MockAgentProcess:
        process = MockAgentProcess(self.cases[session_input.session_id].row)
        self.processes[session_input.session_id] = process
        return process


class RemoteCall:
    def __init__(self, callback: Any) -> None:
        self.remote = callback


class PermitAdapter:
    def __init__(self) -> None:
        self.owned: set[str] = set()
        self.acquire_request_permit = RemoteCall(self.acquire)
        self.release_request_permit = RemoteCall(self.release)
        self.cancel_request_permit = RemoteCall(self.release)

    async def acquire(self, permit_id: str) -> bool:
        if permit_id in self.owned:
            raise AssertionError("Duplicate permit ownership")
        self.owned.add(permit_id)
        return True

    async def release(self, permit_id: str) -> None:
        self.owned.discard(permit_id)


class ReplayBackend:
    def __init__(self, cases: dict[str, ReplayCase], tokenizer: Any, compiler: Any) -> None:
        self.cases = cases
        self.tokenizer = tokenizer
        self.compiler = compiler
        self.positions = {session_id: 0 for session_id in cases}
        self.cohorts: dict[str, Cohort] = {}
        self.tasks: dict[str, asyncio.Task[Any]] = {}
        self.calls = 0
        self.timed_calls = 0
        self.inflight = 0
        self.peak_timed_inflight = 0
        self.closed: list[str] = []

    async def generate(self, **kwargs: Any) -> BackendGenerateResult:
        session_id, request_id = kwargs["session_id"], kwargs["request_id"]
        case = self.cases[session_id]
        position = self.positions[session_id]
        ids = case.response_tokens[position]
        if len(ids) != kwargs["sampling_params"]["max_new_tokens"]:
            raise AssertionError("Recorded response would be truncated or sampled with another budget")
        self.positions[session_id] += 1
        self.calls += 1
        self.tasks[request_id] = asyncio.current_task()
        timed = session_id in self.cohorts
        try:
            if timed:
                self.timed_calls += 1
                self.inflight += 1
                self.peak_timed_inflight = max(self.peak_timed_inflight, self.inflight)
                await self.cohorts[session_id].backend_enter(session_id)
            return BackendGenerateResult(
                new_tokens=list(ids),
                new_log_probs=[-0.125] * len(ids),
                finish_type="stop",
                meta_info={
                    "prompt_tokens": len(kwargs["input_ids"]),
                    "completion_tokens": len(ids),
                    "finish_reason": {"type": "stop"},
                    "weight_version": "mock-replay",
                },
                elapsed=0.0,
            )
        finally:
            if timed:
                self.inflight -= 1
            self.tasks.pop(request_id, None)

    async def abort_request(self, request_id: str) -> None:
        task = self.tasks.get(request_id)
        if task is not None:
            task.cancel()

    async def close_session(self, session_id: str, **kwargs: Any) -> bool:
        self.closed.append(session_id)
        return True

    async def shutdown(self) -> None:
        if self.tasks or self.inflight:
            raise AssertionError("Backend retains active work after Session cleanup")


def make_shard(cases: dict[str, ReplayCase], tokenizer: Any, executor: ThreadPoolExecutor) -> Any:
    shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = SimpleNamespace(
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=max(len(ids) for case in cases.values() for ids in case.response_tokens),
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
        rollout_seed=0,
        fully_async=False,
        partial_rollout=False,
        use_slime_router=False,
        use_rollout_routing_replay=False,
        agentic_reasoning_parser=None,
        agentic_tool_call_parser=None,
        rollout_max_context_len=None,
        eval_max_context_len=None,
        agentic_admission_scope="all",
        agentic_session_lifecycle=True,
    )
    compiler = SGLangMessageCompiler(
        tokenizer=tokenizer,
        processor=None,
        cpu_executor=executor,
        apply_chat_template_kwargs={"enable_thinking": False},
    )
    shard._generation_backend = ReplayBackend(cases, tokenizer, compiler)
    shard._agent_launcher = MockAgentLauncher(cases)
    shard._groups = {}
    shard._session_records = {}
    shard._train_generation_open = True
    shard._state_revision = 0
    shard._state_changed = asyncio.Event()
    shard._sglang_request_limiter = PermitAdapter()
    shard._admission_client = None
    shard._permit_cleanup_tasks = set()
    shard._lifecycle_close_count = 0
    shard._lifecycle_close_failure_count = 0
    return shard


async def replay_turn(shard: Any, session_id: str, case: ReplayCase, turn: int, *, first: bool) -> None:
    position = case.assistant_positions[turn]
    messages = check_messages(copy.deepcopy(case.messages[:position]))
    session = shard._session_records[session_id]
    forest = session.forest
    previous = None if first else session.resp_state_hash_by_request_id[f"req_{session_id}_{turn - 1}"]
    if previous is not None:
        parent, delta = shard._match_parent_state(
            forest=forest, messages=messages, tools=[], chat_template_kwargs={"enable_thinking": False}
        )
        if parent.state_hash != previous or any(message["role"] == "assistant" for message in delta):
            raise AssertionError("Replay history forked or re-imported a recorded assistant as observation")
    pending = asyncio.create_task(
        shard.chat(
            session_id=session_id,
            messages=messages,
            tools=[],
            chat_template_kwargs={"enable_thinking": False},
            max_completion_tokens=len(case.response_tokens[turn]),
            stop=None,
            seed=None,
            logprobs=False,
        )
    )
    try:
        if first:
            revision = -1
            while not shard._groups[session_id].first_request_barrier_crossed:
                changed = asyncio.create_task(shard.wait_for_state_change(revision))
                try:
                    completed, _ = await asyncio.wait((changed, pending), return_when=asyncio.FIRST_COMPLETED)
                    if pending in completed:
                        early = await pending
                        raise RuntimeError(f"First chat finished before its Runtime lease: {early}")
                    revision = (await changed).revision
                finally:
                    if not changed.done():
                        changed.cancel()
                    await asyncio.gather(changed, return_exceptions=True)
            await shard.lease_group(session_id, rollout_id=0)
        response = await pending
    finally:
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
    if "error" in response or response["message"] != case.messages[position]:
        raise AssertionError("Actual Session response differs from the canonical recorded assistant")
    node_hash = session.resp_state_hash_by_request_id[response["request_id"]]
    if forest.full_messages(node_hash) != check_messages(case.messages[: position + 1]):
        raise AssertionError("Actual committed Session history differs from its recorded prefix")


def training_digest(transport: SessionExportTransport) -> str:
    digest = hashlib.sha256()
    for export in transport.exports:
        sample = TrainingFieldArtifact(sample_payload=export["sample_payload"]).to_sample().to_dict()
        for key in (
            "tokens",
            "rollout_tokens",
            "loss_mask",
            "rollout_log_probs",
            "response",
            "response_length",
            "weight_versions",
            "custom_advantage",
            "reward",
            "status",
            "rollout_topk_token_ids",
            "rollout_topk_log_probs",
            "rollout_routed_experts",
        ):
            value = sample.get(key)
            if isinstance(value, np.ndarray):
                digest.update(str((key, value.dtype.str, value.shape)).encode())
                digest.update(value.tobytes())
            else:
                digest.update(json.dumps([key, value], sort_keys=True, default=str).encode())
    return digest.hexdigest()


async def warmup_chain(case: ReplayCase, tokenizer: Any, executor: ThreadPoolExecutor) -> dict[str, Any]:
    """Warm the actual chat/export/cleanup chain with one fixed real record."""
    shard = make_shard({"warmup": case}, tokenizer, executor)
    started = time.perf_counter()
    transport = None
    try:
        await shard.start_group(
            "train",
            "warmup",
            (
                SessionSpec(
                    session_id="warmup",
                    group_index=0,
                    index=case.row,
                    train_metadata={"source": "SWE-Gym/OpenHands-SFT-Trajectories", "source_row": case.row},
                ),
            ),
        )
        session = shard._session_records["warmup"]
        resources = session.resources
        for turn in range(len(case.assistant_positions)):
            await replay_turn(shard, "warmup", case, turn, first=turn == 0)
        resources.process.finished.set()
        await asyncio.shield(resources.watcher_task)
        (transport,) = await shard.take_session_results("warmup", ("warmup",))
        if not isinstance(transport, SessionExportTransport) or len(transport.exports) != 1:
            raise AssertionError("Complete-chain warmup did not publish its one export")
        await shard.release_group("warmup")
        if session.forest is not None or session.resources is not None or not resources.process_wait.done():
            raise AssertionError("Complete-chain warmup retained Session resources")
        if resources.process.terminate_calls != 1 or shard._generation_backend.closed != ["warmup"]:
            raise AssertionError("Complete-chain warmup did not close exactly once")
    finally:
        await shard.shutdown()
    if shard._groups or shard._session_records or shard._permit_cleanup_tasks or shard._sglang_request_limiter.owned:
        raise AssertionError("Warmup left owned resources before the measured workload")
    return {
        "source_row": case.row,
        "completed_exports": 1,
        "backend_calls": shard._generation_backend.calls,
        "seconds": time.perf_counter() - started,
        "cleanup_complete": True,
        "training_sha256": training_digest(transport),
    }


async def measure(args: argparse.Namespace, cases: list[ReplayCase], tokenizer: Any) -> dict[str, Any]:
    case_map = {f"replay-{i}": case for i, case in enumerate(cases)}
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="replay-compiler")
    shard = make_shard(case_map, tokenizer, executor)
    backend = shard._generation_backend
    probe_task = None
    clients: list[asyncio.Task[Any]] = []
    results: dict[str, SessionExportTransport] = {}
    probes: list[dict[str, float]] = []
    export_times: list[float] = []
    export_session_order: list[str] = []
    client_times: list[float] = []
    builder = shard._build_session_export_transport
    builder_active = 0
    builder_peak = 0

    def timed_builder(**kwargs: Any) -> SessionExportTransport:
        nonlocal builder_active, builder_peak
        builder_active += 1
        builder_peak = max(builder_peak, builder_active)
        export_session_order.append(kwargs["session"].session_id)
        started = time.perf_counter()
        try:
            return builder(**kwargs)
        finally:
            export_times.append(time.perf_counter() - started)
            builder_active -= 1

    shard._build_session_export_transport = timed_builder
    try:
        warmup = await warmup_chain(min(cases, key=lambda case: case.row), tokenizer, executor)
        for session_id, case in case_map.items():
            await shard.start_group(
                "train",
                session_id,
                (
                    SessionSpec(
                        session_id=session_id,
                        group_index=0,
                        index=case.row,
                        train_metadata={"source": "SWE-Gym/OpenHands-SFT-Trajectories", "source_row": case.row},
                    ),
                ),
            )
            for turn in range(len(case.assistant_positions) - 1):
                await replay_turn(shard, session_id, case, turn, first=turn == 0)
        expected_calls = sum(len(case.assistant_positions) for case in cases)
        prepare_calls = backend.calls
        if prepare_calls != expected_calls - len(cases):
            raise AssertionError("Incomplete preparation")
        loop = asyncio.get_running_loop()
        interval = args.probe_interval_ms / 1000
        done = asyncio.Event()
        window = {"start": float("inf"), "end": float("inf")}

        async def probe() -> None:
            due = loop.time() + interval
            while True:
                await asyncio.sleep(max(0.0, due - loop.time()))
                if done.is_set() and due > window["end"]:
                    return
                started = loop.time()
                await shard.health()
                ended = loop.time()
                if due >= window["start"]:
                    probes.append({"due": due, "start": started, "end": ended})
                due += interval

        gate = asyncio.Semaphore(args.concurrency)
        cohorts: list[Cohort] = []
        active_clients = 0
        peak_clients = 0
        admitted = 0

        async def client(session_id: str, case: ReplayCase) -> None:
            nonlocal active_clients, peak_clients, admitted
            async with gate:
                ordinal = admitted
                admitted += 1
                cohort_index = ordinal // args.concurrency
                if cohort_index == len(cohorts):
                    cohorts.append(Cohort(min(args.concurrency, len(cases) - ordinal)))
                cohort = cohorts[cohort_index]
                backend.cohorts[session_id] = cohort
                active_clients += 1
                peak_clients = max(peak_clients, active_clients)
                process = shard._agent_launcher.processes[session_id]
                session = shard._session_records[session_id]
                resources = session.resources
                try:
                    await replay_turn(shard, session_id, case, len(case.assistant_positions) - 1, first=False)
                    await cohort.finish_ready(process)
                    await asyncio.shield(resources.watcher_task)
                    (transport,) = await shard.take_session_results(session_id, (session_id,))
                    if not isinstance(transport, SessionExportTransport) or len(transport.exports) != 1:
                        raise AssertionError("Missing, dropped or failed replay export")
                    results[session_id] = transport
                    await shard.release_group(session_id)
                    if (
                        session.forest is not None
                        or session.resources is not None
                        or session.live_irs
                        or session.queued_irs
                        or session.resp_state_hash_by_request_id
                    ):
                        raise AssertionError("Finalized Session retains resources")
                    if not resources.process_wait.done() or process.terminate_calls != 1:
                        raise AssertionError("Process capability was not joined exactly once")
                    client_times.append(loop.time() - window["start"])
                finally:
                    active_clients -= 1

        probe_task = asyncio.create_task(probe())
        await asyncio.sleep(interval * 2)
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        window["start"] = loop.time()
        clients = [asyncio.create_task(client(session_id, case)) for session_id, case in case_map.items()]
        await asyncio.gather(*clients)
        window["end"] = loop.time()
        wall_seconds, cpu_seconds = time.perf_counter() - wall_start, time.process_time() - cpu_start
        peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        done.set()
        await probe_task
        if len(results) != len(cases) or len(export_times) != len(cases) or backend.calls != expected_calls:
            raise AssertionError("Timed run lost declared work")
        if peak_clients != args.concurrency or backend.peak_timed_inflight != args.concurrency or builder_peak != 1:
            raise AssertionError("Declared concurrency did not become effective")
        if backend.timed_calls != len(cases) or any(
            cohort.backend_arrivals != cohort.size or cohort.ready_arrivals != cohort.size for cohort in cohorts
        ):
            raise AssertionError("Incomplete backend/export-ready cohort")
        if len(backend.closed) != len(cases) or set(backend.closed) != set(case_map):
            raise AssertionError("Each Session must close exactly once")
        await shard.shutdown()
        if (
            shard._groups
            or shard._session_records
            or shard._permit_cleanup_tasks
            or shard._sglang_request_limiter.owned
        ):
            raise AssertionError("Shard retains owned resources after shutdown")
        if not probes:
            raise AssertionError("No event-loop observations; do not report a meaningless P99")

        def p99(values: list[float]) -> float:
            return float(np.percentile(values, 99, method="linear"))

        return {
            "window": "last request + canonical-prefix audit + mock backend + finalization + result take/release",
            "complete_chain_warmup": warmup,
            "completed_exports": len(results),
            "resident_sessions_at_start": len(cases),
            "prepare_backend_calls": prepare_calls,
            "timed_backend_calls": backend.timed_calls,
            "total_backend_calls": backend.calls,
            "peak_clients": peak_clients,
            "peak_backend_inflight": backend.peak_timed_inflight,
            "peak_sync_builder_active": builder_peak,
            "cohorts": [
                {
                    "size": c.size,
                    "backend_arrivals": c.backend_arrivals,
                    "export_ready_arrivals": c.ready_arrivals,
                    "backend_session_order": c.backend_session_order,
                    "export_ready_source_rows": [p.output.metadata["source_row"] for p in c.processes],
                }
                for c in cohorts
            ],
            "declared_submission_session_order": list(case_map),
            "export_session_order": export_session_order,
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "exports_per_second": len(cases) / wall_seconds,
            "loop_p99_seconds": p99([row["start"] - row["due"] for row in probes]),
            "short_request_p99_seconds": p99([row["end"] - row["due"] for row in probes]),
            "client_p99_seconds": p99(client_times),
            "builder_p99_seconds": p99(export_times),
            "client_seconds": client_times,
            "builder_seconds": export_times,
            "probes": probes,
            "probe_count": len(probes),
            "measurement_window": {"start": window["start"], "end": window["end"], "interval_seconds": interval},
            "peak_rss_kib_including_imports_and_preparation": peak_rss_kib,
            "training_sha256_by_session": {key: training_digest(value) for key, value in results.items()},
            "cleanup_complete": True,
            "mock_processes_are_os_processes": False,
        }
    finally:
        for client_task in clients:
            if not client_task.done():
                client_task.cancel()
        await asyncio.gather(*clients, return_exceptions=True)
        if probe_task is not None and not probe_task.done():
            probe_task.cancel()
            await asyncio.gather(probe_task, return_exceptions=True)
        try:
            await shard.shutdown()
        finally:
            # Keep the executor teardown even when native cleanup raises. Python
            # preserves the original and cleanup exceptions in their context chain.
            executor.shutdown(wait=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True, help="Pinned tokenizer directory, no model weights")
    parser.add_argument("--length", choices=("short", "long"), required=True)
    parser.add_argument("--exports", type=int, default=16)
    parser.add_argument("--concurrency", type=int, choices=(1, 4), required=True)
    parser.add_argument("--probe-interval-ms", type=float, default=5.0)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.exports < args.concurrency or args.probe_interval_ms <= 0:
        parser.error("Need exports >= concurrency and a positive probe interval")
    from transformers import AutoTokenizer

    tokenizer_source = json.loads((args.tokenizer / "tokenizer-source.json").read_text())
    if (
        tokenizer_source["id"] != "Qwen/Qwen3-0.6B"
        or tokenizer_source["revision"] != "c1899de289a04d12100db370d81485cdf75e47ca"
    ):
        raise ValueError("Unexpected tokenizer repository/revision")
    files = tokenizer_source["files"]
    if not files or set(files) != {
        path.name for path in args.tokenizer.iterdir() if path.name != "tokenizer-source.json"
    }:
        raise ValueError("Tokenizer directory differs from its declared file set")
    for name, expected_hash in files.items():
        if (
            Path(name).name != name
            or hashlib.sha256((args.tokenizer / name).read_bytes()).hexdigest() != expected_hash
        ):
            raise ValueError("Tokenizer bytes differ from the pinned cloud preparation manifest")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    cases, selection = load_cases(args, tokenizer)
    case_set = sorted((case.describe() for case in cases), key=lambda value: value["source_row"])
    random.Random(args.order_seed).shuffle(cases)
    selection = {
        **selection,
        "cases": [case.describe() for case in cases],
        "order_seed": args.order_seed,
        "case_set_sha256": hashlib.sha256(json.dumps(case_set, sort_keys=True).encode()).hexdigest(),
    }
    result = asyncio.run(measure(args, cases, tokenizer))
    result.update(
        selection=selection,
        length=args.length,
        concurrency=args.concurrency,
        tokenizer_source=tokenizer_source,
        token_ids="reconstructed; not original trajectory token IDs",
        logprobs="synthetic constant -0.125",
        inference="mock backend, no model weights or real model inference",
        formal_evaluation=False,
        order_seed=args.order_seed,
        rss_scope="process high-water including imports, complete-chain warmup and preparation",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
