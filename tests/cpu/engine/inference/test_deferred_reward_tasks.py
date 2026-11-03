# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
from types import SimpleNamespace

import pytest

from relax.engine import rewards
from relax.engine.inference import deferred
from relax.utils import utils as relax_utils
from tests.cpu.engine.inference.test_deferred_scoring import _args, _coordinator, _DataSystem, _groups


@pytest.mark.parametrize("group_rm", [False, True])
async def test_deferred_nested_reward_failure_settles_before_offload_and_publish(monkeypatch, group_rm):
    args = _args(group_rm=group_rm)
    events, published = [], _DataSystem()
    slow_started, cleaning, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    tasks = []

    async def reward(args, sample):
        tasks.append(asyncio.current_task())
        if sample.index == 0:
            await slow_started.wait()
            raise ValueError("fast failure")
        if sample.index == 1:
            slow_started.set()
            try:
                await asyncio.Event().wait()  # HTTP/retry suspended until cancellation
            finally:
                cleaning.set()
                await finish.wait()
                events.append("slow settled")
        return 1.0

    monkeypatch.setattr(rewards, "async_rm", reward)
    monkeypatch.setattr(deferred, "build_local_coordinator", lambda args: _coordinator(args, events))
    monkeypatch.setattr(
        relax_utils,
        "convert_samples_to_train_data",
        lambda args, samples: SimpleNamespace(samples=list(samples), numel=lambda: len(samples)),
    )
    monkeypatch.setattr(relax_utils, "build_rollout_custom_meta", lambda batch: {})
    monkeypatch.setattr(relax_utils, "CURRENT_ROLLOUT_BATCH", [])
    # Both group_rm and multi-sample generation reach batched_async_rm.
    groups = _groups(2) if group_rm else [[_groups(2)[0]]]
    batch = asyncio.create_task(relax_utils.transfer_batch_to_data_system(args, groups, 1, 0, published))
    try:
        await asyncio.wait_for(cleaning.wait(), 2)
        assert "genrm offload" not in events
        assert not batch.done() and published.published == []
        finish.set()
        with pytest.raises(ValueError, match="fast failure"):
            await batch
        assert events.index("slow settled") < events.index("genrm offload")
        assert all(task.done() for task in tasks)
        assert published.published == [] and relax_utils.CURRENT_ROLLOUT_BATCH == []
        next_groups = _groups(1)
        next_groups[0][0].index = 2
        await relax_utils.transfer_batch_to_data_system(args, next_groups, 1, 1, published)
        assert [s.index for group in published.published for s in group] == [2]
    finally:
        finish.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(batch, *tasks, return_exceptions=True)


async def test_sync_reward_cancellation_waits_for_remote_terminal_state(monkeypatch):
    loop = asyncio.get_running_loop()
    running, waiting = asyncio.Event(), asyncio.Event()
    terminal = threading.Event()
    events = []

    class Ref:
        def __await__(self):
            async def wait():
                running.set()
                await asyncio.Event().wait()

            return wait().__await__()

    ref = Ref()
    worker = SimpleNamespace(compute_custom=SimpleNamespace(remote=lambda *args: ref))
    executor = rewards.RewardExecutor(num_workers=1)
    executor._workers = [worker]
    monkeypatch.setattr(rewards, "load_function", lambda path: lambda *args: None)
    monkeypatch.setattr(rewards.ray, "cancel", lambda ref, force: events.append("cancel"))

    def get(ref, *, timeout):
        assert timeout > 0
        loop.call_soon_threadsafe(waiting.set)
        assert terminal.wait(5)
        events.append("terminal")

    monkeypatch.setattr(rewards.ray, "get", get)
    task = asyncio.create_task(executor._execute_custom(_args(custom_rm_path="custom"), _groups(1)[0][0], {}))
    try:
        await asyncio.wait_for(running.wait(), 2)
        task.cancel()
        await asyncio.wait_for(waiting.wait(), 2)
        task.cancel()  # cleanup must survive a second cancellation
        assert not task.done()
        terminal.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == ["cancel", "terminal"]
    finally:
        terminal.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("confirm_kill", [False, True])
async def test_sync_reward_timeout_retires_only_affected_worker_and_can_recover(monkeypatch, confirm_kill):
    worker, healthy, ref = object(), object(), object()
    executor = rewards.RewardExecutor(num_workers=2)
    executor._workers = [worker, healthy]
    events = []
    terminal = False

    def cancel(value, *, force):
        assert value is ref and force is False
        events.append("cancel")

    def get(value, *, timeout):
        assert value is ref and timeout == rewards._REWARD_CANCEL_TIMEOUT_S
        if terminal or (confirm_kill and "kill" in events):
            events.append("terminal")
            return None
        raise rewards.ray.exceptions.GetTimeoutError("not terminal")

    def kill(value, *, no_restart):
        assert value is worker and no_restart is True
        events.append("kill")

    monkeypatch.setattr(rewards.ray, "cancel", cancel)
    monkeypatch.setattr(rewards.ray, "get", get)
    monkeypatch.setattr(rewards.ray, "kill", kill)
    if confirm_kill:
        await executor._cancel_and_wait(ref, worker=worker, custom_rm_path="custom")
        assert events == ["cancel", "kill", "terminal"]
    else:
        with pytest.raises(rewards.ray.exceptions.GetTimeoutError):
            await executor._cancel_and_wait(ref, worker=worker, custom_rm_path="custom")
        assert ref in executor._cancelled_worker_calls
        assert executor._workers == [healthy]
        terminal = True
        await executor.settle_cancelled_calls()
        assert events == ["cancel", "kill", "cancel", "terminal"]
    assert executor._cancelled_worker_calls == {}
    assert executor._workers == [healthy]


async def test_deferred_cancel_during_submission_waits_before_gpu_handoff(monkeypatch):
    args = _args(custom_rm_path="custom")
    events, published = [], _DataSystem()
    loop = asyncio.get_running_loop()
    submitting, waiting, settling = asyncio.Event(), asyncio.Event(), asyncio.Event()
    allow_submit, terminal = threading.Event(), threading.Event()
    ref = object()

    def submit(*args):
        loop.call_soon_threadsafe(submitting.set)
        assert allow_submit.wait(5)
        return ref

    def get(value, *, timeout):
        assert value is ref and timeout > 0
        loop.call_soon_threadsafe(waiting.set)
        assert terminal.wait(5)
        events.append("remote terminal")

    executor = rewards.RewardExecutor(num_workers=1)
    executor._workers = [SimpleNamespace(compute_custom=SimpleNamespace(remote=submit))]
    settle = executor.settle_cancelled_calls

    async def settle_observed():
        if executor._submission_cleanup_tasks:
            settling.set()
        await settle()

    monkeypatch.setattr(executor, "settle_cancelled_calls", settle_observed)
    monkeypatch.setattr(rewards.RewardExecutor, "_instance", executor)
    monkeypatch.setattr(rewards, "load_function", lambda path: lambda *args: None)
    monkeypatch.setattr(rewards.ray, "cancel", lambda ref, force: events.append("cancel"))
    monkeypatch.setattr(rewards.ray, "get", get)
    monkeypatch.setattr(deferred, "build_local_coordinator", lambda args: _coordinator(args, events))
    batch = asyncio.create_task(relax_utils.transfer_batch_to_data_system(args, _groups(1), 1, 0, published))
    try:
        await asyncio.wait_for(submitting.wait(), 2)
        batch.cancel()
        await asyncio.wait_for(settling.wait(), 2)
        assert not batch.done() and "genrm offload" not in events
        allow_submit.set()
        await asyncio.wait_for(waiting.wait(), 2)
        batch.cancel()
        assert not batch.done() and "genrm offload" not in events
        terminal.set()
        with pytest.raises(asyncio.CancelledError):
            await batch
        assert events.index("remote terminal") < events.index("genrm offload")
        assert published.published == []
        assert executor._submission_cleanup_tasks == set()
        assert executor._cancelled_worker_calls == {}
    finally:
        allow_submit.set()
        terminal.set()
        await asyncio.gather(batch, return_exceptions=True)
