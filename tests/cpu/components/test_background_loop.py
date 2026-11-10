# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
from unittest.mock import Mock

import pytest

from relax.components.background_loop import BackgroundLoopRunner


@pytest.mark.parametrize("fail", [False, True])
async def test_concurrent_waiters_share_thread_and_result(fail):
    runner = BackgroundLoopRunner()
    release = threading.Event()
    error = ValueError("worker failed")
    calls = []
    main_thread = threading.get_ident()
    preparation = Mock(side_effect=lambda: calls.append(threading.get_ident()))

    def work():
        calls.append(threading.get_ident())
        assert release.wait(5)
        if fail:
            raise error

    tasks = [asyncio.create_task(runner.run(work, before_start=preparation)) for _ in range(3)]
    try:
        await asyncio.sleep(0)
        assert not any(task.done() for task in tasks)
    finally:
        release.set()
    results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
    assert results == ([error] * 3 if fail else [None] * 3)
    preparation.assert_called_once_with()
    assert len(calls) == 2
    assert calls[0] == main_thread
    assert calls[1] != main_thread
    assert await runner.join(1)


async def test_cancelled_waiter_and_join_timeout_do_not_release_live_thread():
    runner = BackgroundLoopRunner()
    release = threading.Event()
    work = Mock(side_effect=lambda: release.wait(5))
    first = asyncio.create_task(runner.run(work))
    second = asyncio.create_task(runner.run(work))
    try:
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not await runner.join(0)
        third = asyncio.create_task(runner.run(work))
        await asyncio.sleep(0)
        assert not second.done()
        assert not third.done()
    finally:
        release.set()
    await asyncio.wait_for(asyncio.gather(second, third), 5)
    work.assert_called_once_with()
    assert await runner.join(1)


async def test_join_allows_event_loop_to_release_worker():
    runner = BackgroundLoopRunner()
    release = threading.Event()
    task = asyncio.create_task(runner.run(lambda: release.wait(5)))
    await asyncio.sleep(0)
    asyncio.get_running_loop().call_later(0.01, release.set)
    try:
        # A synchronous join would time out before the callback could run.
        assert await runner.join(1)
        await asyncio.wait_for(task, 1)
    finally:
        release.set()


async def test_previous_waiter_retains_error_when_next_run_starts():
    runner = BackgroundLoopRunner()
    error = ValueError("first run")

    def fail():
        raise error

    first = asyncio.create_task(runner.run(fail))
    await asyncio.sleep(0)
    # Hold the event loop while the short-lived worker exits. Its completion
    # callback is still queued, so the next generation starts before it resumes.
    runner._state.thread.join(1)
    assert not runner._state.thread.is_alive()
    assert not first.done()
    await runner.run(lambda: None)
    with pytest.raises(ValueError, match="first run") as exc:
        await first
    assert exc.value is error
    assert await runner.join(1)


async def test_preparation_failure_does_not_start_worker_and_can_retry():
    runner = BackgroundLoopRunner()
    work = Mock()
    with pytest.raises(ValueError, match="prepare"):
        await runner.run(work, before_start=Mock(side_effect=ValueError("prepare")))
    work.assert_not_called()
    assert await runner.join(0)
    await runner.run(work)
    work.assert_called_once_with()
    assert await runner.join(1)
