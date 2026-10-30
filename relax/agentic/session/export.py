# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bound CPU materialization without queueing heavy inputs in an executor."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

from relax.agentic.pipeline.runtime import finish_before_cancellation


_Result = TypeVar("_Result")


class SessionExportWorker:
    """One materializer per Shard, including during caller cancellation.

    Waiting callers retain their existing Session state; they do not copy it or
    submit work to ThreadPoolExecutor's unbounded queue. The Session owner must
    hold its lock until run() returns, including its cancellation drain.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="session-export")
        self._slot = asyncio.Semaphore(1)
        self._closed = False

    async def run(self, materialize: Callable[[], _Result]) -> _Result:
        async with self._slot:
            if self._closed:
                raise RuntimeError("Session export worker is closed")
            return await finish_before_cancellation(
                asyncio.get_running_loop().run_in_executor(self._executor, materialize),
                "session-export-materialize",
            )

    async def shutdown(self) -> None:
        self._closed = True
        async with self._slot:
            # All submitted work has completed before this idle executor is
            # joined. No CPU materialization runs on the event loop here.
            self._executor.shutdown(wait=True, cancel_futures=True)
