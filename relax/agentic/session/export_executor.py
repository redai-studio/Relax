# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded CPU export work owned by one Session Shard."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, TypeVar


_Result = TypeVar("_Result")


async def _join_before_cancellation(work: asyncio.Future[_Result]) -> _Result:
    """Keep ownership of running work through repeated caller cancellation.

    Cancelling an asyncio wrapper does not stop its Python worker thread. The
    caller must retain its Session lock and the admission slot until the real
    work has finished, including when that work raises an exception.
    """
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(work)
        except asyncio.CancelledError:
            if work.done():
                # A callable may itself raise CancelledError without marking
                # the executor Future cancelled. Consume that terminal state
                # instead of repeatedly shielding a completed failure.
                if not work.cancelled():
                    work.exception()
                raise
            cancelled = True
        except BaseException:
            if cancelled:
                raise asyncio.CancelledError from None
            raise
        else:
            if cancelled:
                raise asyncio.CancelledError
            return result


class SessionExportExecutor:
    """Run at most one export per Shard without blocking its event loop.

    Waiters stay in asyncio, before submission, so the executor never owns an
    unbounded queue of export inputs. Threads start lazily on the first export.
    This improves request responsiveness; Python work still competes for the
    GIL and does not necessarily use less CPU or finish faster.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="session-export")
        self._slot = asyncio.Lock()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def run(self, work: Callable[..., _Result], *args: Any, **kwargs: Any) -> _Result:
        """Submit after admission and drain running work before
        cancellation."""
        if self._closed:
            raise RuntimeError("Session export executor is closed")
        async with self._slot:
            if self._closed:
                raise RuntimeError("Session export executor is closed")
            future = asyncio.get_running_loop().run_in_executor(self._executor, partial(work, *args, **kwargs))
            return await _join_before_cancellation(future)

    def stop_accepting(self) -> None:
        """Reject new submissions before Session shutdown starts."""
        self._closed = True

    async def close(self) -> None:
        """Drain the active export and join the owned thread exactly once."""
        self.stop_accepting()
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(), name="session-export-close")
        await _join_before_cancellation(self._close_task)

    async def _close(self) -> None:
        async with self._slot:
            # The slot guarantees there is no submitted work left. Join off
            # the event loop as the final worker teardown can still take time.
            await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
