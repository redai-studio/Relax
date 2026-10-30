# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only ownership and data-equivalence checks for Session export work."""

from __future__ import annotations

import asyncio
import gc
import subprocess
import sys
import textwrap
import threading
import weakref
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch

from relax.agentic.pipeline.runtime import finish_before_cancellation
from relax.agentic.runner import SessionOutput
from relax.agentic.session.export_executor import SessionExportExecutor
from relax.agentic.session.service import (
    AgenticSessionShard,
    ResidentGroup,
    RuntimeGroupError,
    _SessionRecord,
    _SessionResultCell,
)
from relax.agentic.session.state import SessionForest


async def _wait_set(event: threading.Event) -> None:
    async def poll() -> None:
        while not event.is_set():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), timeout=5)


def _blocking_work(started: threading.Event, release: threading.Event) -> threading.Thread:
    started.set()
    if not release.wait(timeout=10):
        raise TimeoutError("test did not release the export worker")
    return threading.current_thread()


async def test_export_executor_serializes_work_without_blocking_event_loop() -> None:
    executor = SessionExportExecutor()
    started, release = threading.Event(), threading.Event()
    calls: list[int] = []
    first = asyncio.create_task(executor.run(_blocking_work, started, release))
    second = asyncio.create_task(executor.run(lambda *, value: calls.append(value), value=7))
    try:
        await _wait_set(started)
        assert await asyncio.wait_for(asyncio.sleep(0, result="responsive"), timeout=1) == "responsive"
        assert not second.done()
        assert calls == []
        release.set()
        worker = await first
        await second
        assert calls == [7]
        await executor.close()
        assert not worker.is_alive()
    finally:
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        await executor.close()


async def test_export_executor_cancels_waiting_work_before_submission() -> None:
    executor = SessionExportExecutor()
    started, release = threading.Event(), threading.Event()
    calls: list[str] = []
    running = asyncio.create_task(executor.run(_blocking_work, started, release))
    try:
        await _wait_set(started)
        queued = asyncio.create_task(executor.run(lambda: calls.append("cancelled")))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        await running
        assert await executor.run(lambda: "next") == "next"
        assert calls == []
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)
        await executor.close()


@pytest.mark.parametrize("worker_raises", [False, True])
async def test_export_executor_repeated_cancellation_drains_running_work(worker_raises: bool) -> None:
    executor = SessionExportExecutor()
    started, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []

    def work() -> None:
        workers.append(_blocking_work(started, release))
        if worker_raises:
            raise ValueError("worker failed after cancellation")

    running = asyncio.create_task(executor.run(work))
    try:
        await _wait_set(started)
        running.cancel()
        await asyncio.sleep(0)
        running.cancel()
        await asyncio.sleep(0)
        assert not running.done()
        next_work = asyncio.create_task(executor.run(lambda: "next"))
        await asyncio.sleep(0)
        assert not next_work.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert await next_work == "next"
        await executor.close()
        assert len(workers) == 1
        assert not workers[0].is_alive()
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)
        await executor.close()


async def test_export_executor_propagates_failure_and_remains_usable() -> None:
    executor = SessionExportExecutor()

    def fail() -> None:
        raise ValueError("bad sample")

    try:
        with pytest.raises(ValueError, match="bad sample"):
            await executor.run(fail)
        assert await executor.run(lambda x: x + 1, 2) == 3
    finally:
        await executor.close()


async def test_export_executor_close_rejects_waiters_and_joins_after_repeated_cancellation() -> None:
    executor = SessionExportExecutor()
    started, release = threading.Event(), threading.Event()
    calls: list[str] = []
    running = asyncio.create_task(executor.run(_blocking_work, started, release))
    try:
        await _wait_set(started)
        queued = asyncio.create_task(executor.run(lambda: calls.append("queued")))
        await asyncio.sleep(0)
        close = asyncio.create_task(executor.close())
        await asyncio.sleep(0)
        close.cancel()
        await asyncio.sleep(0)
        close.cancel()
        await asyncio.sleep(0)
        assert not close.done()
        with pytest.raises(RuntimeError, match="closed"):
            await executor.run(lambda: calls.append("new"))
        release.set()
        worker = await running
        with pytest.raises(RuntimeError, match="closed"):
            await queued
        with pytest.raises(asyncio.CancelledError):
            await close
        await executor.close()
        assert not worker.is_alive()
        assert calls == []
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)
        await executor.close()


class _Tokenizer:
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return ":".join(map(str, token_ids))


def _new_shard() -> Any:
    shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = SimpleNamespace(
        agentic_session_lifecycle=False,
        use_rollout_routing_replay=True,
        num_layers=2,
        moe_router_topk=2,
    )
    shard._export_executor = SessionExportExecutor()
    shard._generation_backend = SimpleNamespace(
        tokenizer=_Tokenizer(),
        compiler=SimpleNamespace(apply_chat_template_kwargs={}),
        abort_request=AsyncMock(),
        shutdown=AsyncMock(),
    )
    shard._groups = {}
    shard._session_records = {}
    shard._permit_cleanup_tasks = set()
    shard._notify_state_change = lambda group=None: None
    shard._train_generation_open = True
    return shard


def _new_session(shard: Any, session_id: str = "session") -> tuple[_SessionRecord, str]:
    forest = SessionForest.create_empty(
        session_id=session_id,
        group_index=2,
        index=3,
        label="label",
        train_metadata={"loss": "grpo"},
        metadata={"source": "test"},
    )
    observation = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=4,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "question"}],
        train_token_delta=[11, 12],
        rollout_token_delta=[11],
        multimodal_train_inputs_delta={"pixels": torch.tensor([[1.0, 2.0]])},
    )
    response = forest.append_resp(
        parent_state_hash=observation.state_hash,
        rollout_id=4,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": "tool call"}],
        token_delta=[13, 14],
        logprob_delta=[-0.125, -0.25],
        topk_token_ids_delta=np.array([[13, 20], [14, 21]], dtype=np.int32),
        topk_logprob_delta=np.array([[-0.125, -1.0], [-0.25, -2.0]], dtype=np.float32),
        rollout_routed_experts=np.arange(12, dtype=np.int32).reshape(3, 2, 2),
        weight_version_delta=["v4"],
        export_metadata_patch={"request_id": "request", "agentic_trace": {"events": {"chat_end_at": 1.0}}},
    )
    observation = forest.append_obs(
        parent_state_hash=response.state_hash,
        rollout_id=4,
        abort_count=0,
        messages_delta=[{"role": "tool", "content": "observation", "tool_call_id": "call"}],
        train_token_delta=[15],
        rollout_token_delta=[15],
    )
    leaf = forest.append_resp(
        parent_state_hash=observation.state_hash,
        rollout_id=5,
        abort_count=1,
        messages_delta=[{"role": "assistant", "content": "answer"}],
        token_delta=[16],
        logprob_delta=[-0.5],
        topk_token_ids_delta=np.array([[16, 22]], dtype=np.int32),
        topk_logprob_delta=np.array([[-0.5, -3.0]], dtype=np.float32),
        rollout_routed_experts=np.arange(20, dtype=np.int32).reshape(5, 2, 2),
        weight_version_delta=["v5"],
    )
    cell = _SessionResultCell()
    group = ResidentGroup(
        rollout_mode="train", group_id=f"group-{session_id}", result_cells={session_id: cell}, sessions=[]
    )
    process = SimpleNamespace(
        output=SessionOutput(metadata={"result": "success"}, reward={"score": 1.0}),
        terminate_and_join=AsyncMock(),
    )
    session = _SessionRecord(
        group=group,
        session_id=session_id,
        session_sampling_params={},
        result_cell=cell,
        forest=forest,
        resp_state_hash_by_request_id={"request": response.state_hash},
        resources=SimpleNamespace(
            process=process,
            process_wait=asyncio.create_task(asyncio.sleep(0)),
            watcher_task=asyncio.create_task(asyncio.sleep(0)),
        ),
    )
    group.sessions.append(session)
    shard._groups[group.group_id] = group
    shard._session_records[session_id] = session
    return session, leaf.state_hash


def _assert_same(left: Any, right: Any) -> None:
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("explicit", [False, True])
async def test_session_export_matches_inline_training_payload(monkeypatch: Any, explicit: bool) -> None:
    shard = _new_shard()
    session, leaf_hash = _new_session(shard)
    monkeypatch.setattr("time.time", lambda: 100.0)
    if explicit:
        forest = session.forest
        assert forest is not None
        session.resources.process.output = SessionOutput.from_records(
            [
                {
                    "name": "trajectory",
                    "messages": forest.full_messages(leaf_hash),
                    "metadata": {"context": "main"},
                    "reward": 2.0,
                }
            ]
        )
    output = session.resources.process.output
    expected = shard._build_session_export_transport(
        session=session,
        reward=output.reward,
        metadata=output.metadata,
        output_records=output.records,
        finalize_events={
            "finalize_arrive_at": 100.0,
            "finalize_lock_acquired_at": 100.0,
            "finalize_start_at": 100.0,
        },
    )
    try:
        await shard._finalize_terminal_session(session)
        actual = session.result_cell.take()
        _assert_same(expected.exports, actual.exports)
        assert session.forest is None
        assert session.resources is None
        assert session.session_id not in shard._session_records
    finally:
        await shard._shutdown()


def _gate_builder(
    shard: Any, *, worker_raises: bool = False
) -> tuple[threading.Event, threading.Event, list[threading.Thread]]:
    started, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []
    original = shard._build_session_export_transport

    def build(**kwargs: Any) -> Any:
        workers.append(_blocking_work(started, release))
        if worker_raises:
            raise ValueError("builder failed after cancellation")
        return original(**kwargs)

    shard._build_session_export_transport = build
    return started, release, workers


async def test_session_export_holds_own_lock_while_health_remains_responsive() -> None:
    shard = _new_shard()
    session, _ = _new_session(shard)
    started, release, workers = _gate_builder(shard)
    finalize = asyncio.create_task(shard._finalize_terminal_session(session))
    trace = None
    try:
        await _wait_set(started)
        trace = asyncio.create_task(
            shard.mark_chat_service_response_ready(
                session_id=session.session_id,
                request_id="request",
                remote_return_at=2.0,
                response_ready_at=3.0,
                http_return_at=4.0,
            )
        )
        await asyncio.sleep(0)
        assert session.lock.locked()
        assert not trace.done()
        assert (await asyncio.wait_for(shard.health(), timeout=1))["active_sessions"] == 1
        release.set()
        await asyncio.gather(finalize, trace)
        assert session.result_cell.terminal
    finally:
        release.set()
        await asyncio.gather(finalize, *([trace] if trace is not None else []), return_exceptions=True)
        await shard._shutdown()
        assert all(not worker.is_alive() for worker in workers)


@pytest.mark.parametrize("worker_raises", [False, True])
async def test_cancelled_session_export_finishes_worker_before_clearing_state(worker_raises: bool) -> None:
    shard = _new_shard()
    session, _ = _new_session(shard)
    process = session.resources.process
    started, release, workers = _gate_builder(shard, worker_raises=worker_raises)
    finalize = asyncio.create_task(shard._finalize_terminal_session(session))
    try:
        await _wait_set(started)
        finalize.cancel()
        await asyncio.sleep(0)
        finalize.cancel()
        await asyncio.sleep(0)
        assert not finalize.done()
        assert session.lock.locked()
        assert session.forest is not None
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await finalize
        assert session.result_cell.take() is None
        assert session.forest is None
        assert session.resources is None
        assert session.session_id not in shard._session_records
        process.terminate_and_join.assert_awaited_once()
    finally:
        release.set()
        await asyncio.gather(finalize, return_exceptions=True)
        await shard._shutdown()
        assert all(not worker.is_alive() for worker in workers)


async def test_cancelled_queued_session_cleans_up_without_running_builder_or_affecting_active_export() -> None:
    shard = _new_shard()
    first, _ = _new_session(shard, "active")
    queued, _ = _new_session(shard, "queued")
    queued_process = queued.resources.process
    started, release, workers = _gate_builder(shard)
    entered_queue = asyncio.Event()
    native_run = shard._export_executor.run
    builder = shard._build_session_export_transport
    built: list[str] = []

    async def observed_run(work: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("session") is queued:
            entered_queue.set()
        return await native_run(work, *args, **kwargs)

    def observed_builder(**kwargs: Any) -> Any:
        built.append(kwargs["session"].session_id)
        return builder(**kwargs)

    shard._export_executor.run = observed_run
    shard._build_session_export_transport = observed_builder
    active_task = asyncio.create_task(shard._finalize_terminal_session(first))
    queued_task = None
    try:
        await _wait_set(started)
        queued_task = asyncio.create_task(shard._finalize_terminal_session(queued))
        await asyncio.wait_for(entered_queue.wait(), timeout=1)
        assert queued.lock.locked() and queued.phase.name == "FINALIZING"
        queued_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(queued_task, timeout=1)
        assert not release.is_set() and not active_task.done()
        assert first.lock.locked() and first.forest is not None
        assert not first.result_cell.terminal
        assert built == [first.session_id]
        assert queued.result_cell.take() is None
        assert queued.forest is None and queued.resources is None
        assert not queued.live_irs and not queued.queued_irs and not queued.resp_state_hash_by_request_id
        assert queued.session_id not in shard._session_records
        queued_process.terminate_and_join.assert_awaited_once()
        await shard.drop_group(queued.group.group_id)
        queued_process.terminate_and_join.assert_awaited_once()
        release.set()
        await active_task
        assert first.result_cell.take() is not None
        assert built == [first.session_id]
    finally:
        release.set()
        await asyncio.gather(active_task, *([queued_task] if queued_task is not None else []), return_exceptions=True)
        await shard._shutdown()
        assert all(not worker.is_alive() for worker in workers)


async def test_public_chat_waits_for_terminal_export_then_rejects_without_committing(monkeypatch: Any) -> None:
    shard = _new_shard()
    session, _ = _new_session(shard)
    forest = session.forest
    assert forest is not None
    original_nodes = set(forest.nodes_by_hash)
    output = session.resources.process.output
    monkeypatch.setattr("time.time", lambda: 100.0)
    expected = shard._build_session_export_transport(
        session=session,
        reward=output.reward,
        metadata=output.metadata,
        output_records=output.records,
        finalize_events={"finalize_arrive_at": 100.0, "finalize_lock_acquired_at": 100.0, "finalize_start_at": 100.0},
    )
    shard._match_parent_state = Mock(side_effect=AssertionError("terminal chat must not match/commit a parent"))
    shard._append_observation = AsyncMock(side_effect=AssertionError("terminal chat must not append an observation"))
    shard._dispatch_queued_irs_locked = Mock(side_effect=AssertionError("terminal chat must not dispatch an IR"))
    started, release, workers = _gate_builder(shard)
    finalize = asyncio.create_task(shard._finalize_terminal_session(session))
    entered_chat = asyncio.Event()
    chat_task = None

    async def chat() -> dict[str, Any]:
        entered_chat.set()
        return await shard.chat(
            session_id=session.session_id,
            messages=[{"role": "user", "content": "late request"}],
            tools=[],
            chat_template_kwargs={},
            max_completion_tokens=1,
            stop=None,
            seed=None,
        )

    try:
        await _wait_set(started)
        chat_task = asyncio.create_task(chat())
        await asyncio.wait_for(entered_chat.wait(), timeout=1)
        assert not chat_task.done()
        assert session.lock.locked() and session.forest is forest
        assert set(forest.nodes_by_hash) == original_nodes
        assert not session.live_irs and not session.queued_irs
        assert (await asyncio.wait_for(shard.health(), timeout=1))["active_sessions"] == 1
        release.set()
        await finalize
        response = await chat_task
        assert response["_http_status"] == 404
        assert response["error"]["code"] == "session_discarded"
        later = await chat()
        assert later["_http_status"] == 404 and later["error"]["code"] == "session_discarded"
        _assert_same(expected.exports, session.result_cell.take().exports)
        assert set(forest.nodes_by_hash) == original_nodes
        shard._match_parent_state.assert_not_called()
        shard._append_observation.assert_not_awaited()
        shard._dispatch_queued_irs_locked.assert_not_called()
        assert not session.live_irs and not session.queued_irs
        assert session.forest is None and session.resources is None
    finally:
        release.set()
        await asyncio.gather(finalize, *([chat_task] if chat_task is not None else []), return_exceptions=True)
        await shard._shutdown()
        assert all(not worker.is_alive() for worker in workers)


async def test_session_export_exception_publishes_error_and_cleans_up() -> None:
    shard = _new_shard()
    session, _ = _new_session(shard)

    def fail(**kwargs: Any) -> None:
        del kwargs
        raise ValueError("invalid trajectory")

    shard._build_session_export_transport = fail
    try:
        await shard._finalize_terminal_session(session)
        result = session.result_cell.take()
        assert isinstance(result, RuntimeGroupError)
        assert "invalid trajectory" in str(result)
        assert session.forest is None
        assert session.resources is None
        assert session.session_id not in shard._session_records
    finally:
        await shard._shutdown()


async def test_drop_group_during_export_publishes_once_and_releases_resources() -> None:
    shard = _new_shard()
    session, _ = _new_session(shard)
    started, release, workers = _gate_builder(shard)
    watcher = asyncio.create_task(
        finish_before_cancellation(shard._finalize_terminal_session(session), "test-session-finalize")
    )
    session.resources.watcher_task = watcher
    drop = None
    try:
        await _wait_set(started)
        drop = asyncio.create_task(shard.drop_group(session.group.group_id))
        await asyncio.sleep(0)
        assert not drop.done()
        release.set()
        await drop
        await asyncio.gather(watcher, return_exceptions=True)
        assert session.result_cell.terminal
        assert session.forest is None
        assert session.resources is None
        assert session.group.group_id not in shard._groups
        assert session.session_id not in shard._session_records
        assert session.group.sessions == []
        await shard.drop_group(session.group.group_id)
    finally:
        release.set()
        await asyncio.gather(watcher, *([drop] if drop is not None else []), return_exceptions=True)
        await shard._shutdown()
        assert all(not worker.is_alive() for worker in workers)


async def test_shutdown_drains_active_export_rejects_queued_work_and_joins_worker() -> None:
    shard = _new_shard()
    first, _ = _new_session(shard, "first")
    second, _ = _new_session(shard, "second")
    started, release, workers = _gate_builder(shard)
    first_finalize = asyncio.create_task(shard._finalize_terminal_session(first))
    second_finalize = None
    shutdown = None
    backend_closed_after_join: list[bool] = []

    async def close_backend() -> None:
        backend_closed_after_join.append(all(not worker.is_alive() for worker in workers))

    shard._generation_backend.shutdown = AsyncMock(side_effect=close_backend)
    try:
        await _wait_set(started)
        second_finalize = asyncio.create_task(shard._finalize_terminal_session(second))
        await asyncio.sleep(0)
        shutdown = asyncio.create_task(shard.shutdown())
        await asyncio.sleep(0)
        assert not shutdown.done()
        assert first.forest is not None
        release.set()
        await asyncio.gather(first_finalize, second_finalize, shutdown)
        assert len(workers) == 1
        assert backend_closed_after_join == [True]
        assert shard._session_records == {}
        assert shard._groups == {}
        assert first.result_cell.terminal and second.result_cell.terminal
        assert isinstance(second.result_cell.payload, RuntimeGroupError)
        with pytest.raises(RuntimeError, match="closed"):
            await shard._export_executor.run(lambda: None)
    finally:
        release.set()
        pending = [
            first_finalize,
            *([second_finalize] if second_finalize is not None else []),
            *([shutdown] if shutdown is not None else []),
        ]
        await asyncio.gather(*pending, return_exceptions=True)
        await shard._export_executor.close()
        assert all(not worker.is_alive() for worker in workers)


async def test_repeated_session_exports_release_forests_and_keep_one_worker() -> None:
    shard = _new_shard()
    forests: list[weakref.ReferenceType[SessionForest]] = []
    workers: set[threading.Thread] = set()
    original = shard._build_session_export_transport

    def build(**kwargs: Any) -> Any:
        workers.add(threading.current_thread())
        return original(**kwargs)

    shard._build_session_export_transport = build
    try:
        for index in range(20):
            session, _ = _new_session(shard, f"repeat-{index}")
            assert session.forest is not None
            forests.append(weakref.ref(session.forest))
            await shard._finalize_terminal_session(session)
            result = session.result_cell.take()
            assert result is not None and len(result.exports) == 1
            await shard.release_group(session.group.group_id)
            assert session.forest is None
            assert session.resources is None
            assert shard._session_records == {}
            assert shard._groups == {}
            assert len(workers) == 1
            assert sum(thread.name.startswith("session-export") for thread in threading.enumerate()) <= 1
        del result, session
        await asyncio.sleep(0)
        gc.collect()
        assert all(reference() is None for reference in forests)
    finally:
        await shard._shutdown()
        assert all(not worker.is_alive() for worker in workers)


def test_export_executor_handles_worker_cancelled_error_without_spinning() -> None:
    # An event-loop timeout cannot interrupt a busy loop over a done Future.
    # Isolate this regression so its timeout also kills a broken worker owner.
    program = textwrap.dedent(
        """
        import asyncio
        from relax.agentic.session.export_executor import SessionExportExecutor

        def cancelled_work():
            raise asyncio.CancelledError("worker cancelled itself")

        async def main():
            executor = SessionExportExecutor()
            try:
                try:
                    await executor.run(cancelled_work)
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError("worker cancellation was not propagated")
                assert await executor.run(lambda: 42) == 42
            finally:
                await executor.close()

        asyncio.run(main())
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=10, check=False
    )
    assert completed.returncode == 0, completed.stderr
