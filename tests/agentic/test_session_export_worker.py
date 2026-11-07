# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import copy
import threading
import time
from argparse import Namespace
from typing import Any

import numpy as np
import pytest

from relax.agentic.pipeline import RuntimeGroupError, SessionExportTransport, TrainingFieldArtifact
from relax.agentic.session.export import SessionExportWorker
from relax.agentic.session.service import AgenticChatRequestError, SessionPhase
from scripts.benchmarks import benchmark_session_export as benchmark
from scripts.benchmarks.benchmark_session_export import make_session, make_shard


def assert_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_equal(left, right)
    else:
        assert actual == expected


async def wait_started(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, 5), "Materializer never started"


@pytest.mark.parametrize("rich", [False, True])
@pytest.mark.parametrize("kind", ["implicit", "observation", "explicit-branches"])
async def test_background_export_preserves_all_training_fields(monkeypatch, rich, kind) -> None:
    # Exact comparison includes all trace metadata when wall clocks are fixed.
    monkeypatch.setattr("time.time", lambda: 1234.0)
    results = []
    for mode in ("inline", "thread"):
        shard = make_shard(mode)
        session = await make_session(shard, 0, 512, turns=2, rich=rich)
        forest = session.forest
        leaf_hash = next(iter(forest.leaf_state_hashes))
        if kind == "observation":
            leaf_hash = forest.append_obs(
                parent_state_hash=leaf_hash,
                rollout_id=3,
                abort_count=0,
                messages_delta=[{"role": "user", "content": "truncated tool output"}],
                train_token_delta=[1, 2],
                rollout_token_delta=[1, 2],
            ).state_hash
        if kind == "explicit-branches":
            parent = forest.nodes_by_hash[leaf_hash].parent_state_hash
            branch = forest.append_resp(
                parent_state_hash=parent,
                rollout_id=3,
                abort_count=0,
                messages_delta=[{"role": "assistant", "content": "alternative"}],
                token_delta=[3, 4],
                logprob_delta=[-1.0, -2.0],
                topk_token_ids_delta=np.array([[1, 2, 3, 4]] * 2, np.int32) if rich else None,
                topk_logprob_delta=np.full((2, 4), -0.5, np.float32) if rich else None,
                rollout_routed_experts=np.zeros((forest.train_token_count(parent) + 1, 2, 2), np.int32)
                if rich
                else None,
            )
            session.resources.process.output.records = tuple(
                {
                    "name": f"branch-{i}",
                    "messages": forest.full_messages(state_hash),
                    "metadata": {"branch": i},
                    "reward": i + 0.25,
                }
                for i, state_hash in enumerate((leaf_hash, branch.state_hash))
            )
        if rich and kind == "implicit":
            shard.args.use_rollout_routing_replay = True
        await shard._finalize_terminal_session(session)
        assert isinstance(session.result_cell.payload, SessionExportTransport)
        results.append(session.result_cell.payload.exports)
        assert session.forest is None and session.resources is None
        assert not shard._session_records
        assert shard._generation_backend.closed_sessions == ["session-0"]
        # Reconstruction also retains arrays/top-k, mask, rewards and status.
        for export in results[-1]:
            sample = TrainingFieldArtifact(export["sample_payload"]).to_sample()
            before = copy.deepcopy(export["sample_payload"])
            sample.tokens[0] = -100
            sample.metadata["fixture"]["seed"] = -100
            if sample.rollout_topk_token_ids is not None:
                sample.rollout_topk_token_ids[0, 0] = -100
            assert_equal(export["sample_payload"], before)
        # Mutating a retained Forest reference cannot change the transport.
        before = copy.deepcopy(results[-1])
        forest.static_metadata["fixture"]["seed"] = -10
        for node in forest.nodes_by_hash.values():
            node.train_token_delta[:] = [-10]
            node.export_metadata_patch.clear()
        assert_equal(results[-1], before)
        await shard.shutdown()
    assert_equal(results[1], results[0])


async def test_worker_bounds_actual_execution_across_repeated_cancellation() -> None:
    worker = SessionExportWorker()
    started, release = threading.Event(), threading.Event()
    executions = []

    def blocked() -> int:
        executions.append(0)
        started.set()
        assert release.wait(5)
        return 0

    active = asyncio.create_task(worker.run(blocked))
    try:
        await wait_started(started)
        queued = [asyncio.create_task(worker.run(lambda n=n: executions.append(n) or n)) for n in range(1, 21)]
        await asyncio.sleep(0)
        for task in queued[:10]:
            task.cancel()
        active.cancel()
        await asyncio.sleep(0)
        active.cancel()
        await asyncio.sleep(0)
        assert executions == [0]
        assert not active.done()
        release.set()
        outcomes = await asyncio.gather(active, *queued, return_exceptions=True)
        assert all(isinstance(outcome, asyncio.CancelledError) for outcome in outcomes[:11])
        assert outcomes[11:] == list(range(11, 21))
        assert executions == [0, *range(11, 21)]
    finally:
        release.set()
        await worker.shutdown()
    assert not any(t.name.startswith("session-export") for t in threading.enumerate())
    with pytest.raises(RuntimeError, match="closed"):
        await worker.run(lambda: 0)


async def test_worker_exception_releases_capacity() -> None:
    worker = SessionExportWorker()
    try:

        def fail() -> None:
            raise ValueError("materialization failed")

        with pytest.raises(ValueError, match="materialization failed"):
            await worker.run(fail)
        assert await worker.run(lambda: 7) == 7
    finally:
        await worker.shutdown()


@pytest.mark.parametrize("error", [ValueError("assembly failed"), RuntimeGroupError("backend failed")])
async def test_export_error_is_published_and_next_export_succeeds(error) -> None:
    shard = make_shard()
    session = await make_session(shard, 0, 256, turns=1)
    original = shard._build_session_export_transport

    def fail(**kwargs: Any) -> None:
        raise error

    shard._build_session_export_transport = fail
    await shard._finalize_terminal_session(session)
    assert isinstance(session.result_cell.payload, RuntimeGroupError)
    assert str(error) in str(session.result_cell.payload)
    shard._build_session_export_transport = original
    next_session = await make_session(shard, 1, 256, turns=1)
    await shard._finalize_terminal_session(next_session)
    assert isinstance(next_session.result_cell.payload, SessionExportTransport)
    assert shard._generation_backend.closed_sessions == ["session-0", "session-1"]
    await shard.shutdown()


async def test_invalid_explicit_export_is_controlled_drop() -> None:
    shard = make_shard()
    session = await make_session(shard, 0, 256, turns=1)
    session.resources.process.output.records = (
        {"name": "invalid", "messages": [{"role": "assistant", "content": "uncommitted"}], "metadata": {}},
    )
    await shard._finalize_terminal_session(session)
    assert session.result_cell.dropped
    assert session.forest is None and not shard._session_records
    await shard.shutdown()


async def test_terminal_lock_serializes_late_trace_and_chat_but_other_session_responds() -> None:
    shard = make_shard()
    session = await make_session(shard, 0, 256, turns=1)
    other = await make_session(shard, 1, 256, turns=1)
    for item in (session, other):
        item.resp_state_hash_by_request_id["request"] = next(iter(item.forest.leaf_state_hashes))
    original = shard._build_session_export_transport
    started, release = threading.Event(), threading.Event()

    def blocked(**kwargs: Any) -> Any:
        started.set()
        assert release.wait(5)
        return original(**kwargs)

    shard._build_session_export_transport = blocked
    finalizer = asyncio.create_task(shard._finalize_terminal_session(session))
    late_tasks = []
    try:
        await wait_started(started)
        assert session.phase is SessionPhase.FINALIZING and session.lock.locked()

        async def mark(item: Any) -> bool:
            return await shard.mark_chat_service_response_ready(
                session_id=item.session_id,
                request_id="request",
                remote_return_at=1,
                response_ready_at=2,
                http_return_at=3,
            )

        late_tasks.append(asyncio.create_task(mark(session)))
        late_tasks.append(
            asyncio.create_task(
                shard._chat(
                    session=session,
                    messages=[],
                    tools=[],
                    chat_template_kwargs={},
                    max_completion_tokens=None,
                    stop=None,
                    seed=None,
                    logprobs=False,
                )
            )
        )
        await asyncio.sleep(0)
        assert not any(task.done() for task in late_tasks)
        assert await asyncio.wait_for(mark(other), 1)
        release.set()
        await finalizer
        outcomes = await asyncio.gather(*late_tasks, return_exceptions=True)
        assert isinstance(outcomes[1], AgenticChatRequestError)
        payload = session.result_cell.payload.exports[0]["sample_payload"]
        turns = payload["metadata"]["agentic_trace"]["turns"]
        assert "chat_service_http_return_at" not in turns[-1]["events"]
    finally:
        release.set()
        await asyncio.gather(finalizer, *late_tasks, return_exceptions=True)
        await shard.shutdown()


@pytest.mark.parametrize("operation", ["cancel", "drop", "shutdown"])
async def test_finalization_drains_active_and_queued_work_before_cleanup(operation) -> None:
    shard = make_shard()
    sessions = [await make_session(shard, n, 256, turns=1) for n in range(2)]
    processes = [session.resources.process for session in sessions]
    original = shard._build_session_export_transport
    started, release = threading.Event(), threading.Event()
    executed = []

    def blocked(**kwargs: Any) -> Any:
        assert kwargs["session"].forest is not None
        executed.append(kwargs["session"].session_id)
        started.set()
        assert release.wait(5)
        assert kwargs["session"].forest is not None
        return original(**kwargs)

    shard._build_session_export_transport = blocked
    finalizers = [asyncio.create_task(shard._finalize_terminal_session(session)) for session in sessions]
    for session, finalizer in zip(sessions, finalizers, strict=True):
        session.resources.watcher_task = finalizer
    cleanup = None
    try:
        await wait_started(started)
        await asyncio.sleep(0)
        assert all(session.lock.locked() for session in sessions)
        assert len(executed) == 1
        if operation == "cancel":
            for finalizer in finalizers:
                finalizer.cancel()
            await asyncio.sleep(0)
            for finalizer in finalizers:
                finalizer.cancel()
        elif operation == "drop":
            cleanup = asyncio.ensure_future(
                asyncio.gather(*(shard.drop_group(session.group.group_id) for session in sessions))
            )
        else:
            cleanup = asyncio.create_task(shard.shutdown())
            await asyncio.sleep(0)
            cleanup.cancel()
        await asyncio.sleep(0)
        assert all(session.forest is not None and not session.result_cell.terminal for session in sessions)
        release.set()
        await asyncio.gather(*finalizers, *([cleanup] if cleanup is not None else []), return_exceptions=True)
        assert len(executed) == 2
        assert all(session.forest is None and session.resources is None for session in sessions)
        assert all(session.result_cell.terminal for session in sessions)
        assert sorted(shard._generation_backend.closed_sessions) == ["session-0", "session-1"]
        assert all(process.terminated == 1 for process in processes)
        assert not shard._session_records
    finally:
        release.set()
        await asyncio.gather(*finalizers, return_exceptions=True)
        await shard.shutdown()
    assert not any(t.name.startswith("session-export") for t in threading.enumerate())


async def test_shutdown_closes_admission_and_joins_running_worker() -> None:
    worker = SessionExportWorker()
    started, release = threading.Event(), threading.Event()

    def blocked() -> int:
        started.set()
        assert release.wait(5)
        return 1

    active = asyncio.create_task(worker.run(blocked))
    await wait_started(started)
    queued = asyncio.create_task(worker.run(lambda: 2))
    closing = asyncio.create_task(worker.shutdown())
    try:
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        assert await active == 1
        with pytest.raises(RuntimeError, match="closed"):
            await queued
        await closing
        await worker.shutdown()
    finally:
        release.set()
        await asyncio.gather(active, queued, closing, return_exceptions=True)
        await worker.shutdown()
    assert not any(t.name.startswith("session-export") for t in threading.enumerate())


async def test_replay_drains_every_scheduled_probe_after_long_block(monkeypatch) -> None:
    original_factory = benchmark.make_shard

    def slow_shard(mode: str) -> Any:
        shard = original_factory(mode)
        original = shard._build_session_export_transport

        def slow_export(**kwargs: Any) -> Any:
            time.sleep(0.04)
            return original(**kwargs)

        shard._build_session_export_transport = slow_export
        return shard

    monkeypatch.setattr(benchmark, "make_shard", slow_shard)
    result = await benchmark.run_case(
        Namespace(
            mode="inline",
            tokens=128,
            rich=False,
            exports=4,
            concurrency=4,
            min_window=0.02,
            interval=0.005,
            profile=False,
        )
    )
    # Export ends after the minimum window. All deadlines through that end
    # must still be issued and observed, even when the producer was blocked.
    assert result["offered_short_requests"] >= 40
    assert result["short_request"]["count"] == result["event_loop"]["count"] == result["offered_short_requests"]
    assert result["completed_exports"] == 4


def test_replay_digest_preserves_training_metadata_named_events() -> None:
    payload = {"metadata": {"events": {"reward": 1}, "agentic_trace": {"events": {"time": 1}}}}
    expected = benchmark.content_digest(payload)
    payload["metadata"]["agentic_trace"]["events"]["time"] = 2
    assert benchmark.content_digest(payload) == expected
    payload["metadata"]["events"]["reward"] = 2
    assert benchmark.content_digest(payload) != expected
