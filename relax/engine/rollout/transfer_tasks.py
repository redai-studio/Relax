# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Observe dispatched training writes without transferring cancellation to
them."""

import asyncio
from collections.abc import Coroutine
from typing import Any


class RolloutTransfers:
    def __init__(self) -> None:
        self.tasks: list[asyncio.Task] = []
        self._first_failure = asyncio.get_running_loop().create_future()

    def submit(self, write: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.create_task(write)
        self.tasks.append(task)
        task.add_done_callback(self._observe)

    def _observe(self, task: asyncio.Task) -> None:
        if (task.cancelled() or task.exception() is not None) and not self._first_failure.done():
            self._first_failure.set_result(task)

    async def wait_first(self, tasks: set[asyncio.Future]) -> set[asyncio.Future]:
        if not tasks:
            raise ValueError("No rollout work to wait for")
        done, _ = await asyncio.wait(tasks | {self._first_failure}, return_when=asyncio.FIRST_COMPLETED)
        if self._first_failure.done():
            self._first_failure.result().result()
        return done

    async def finish(self) -> None:
        pending = set(self.tasks)
        while pending:
            done = await self.wait_first(pending)
            # A write can finish before its completion callback is dispatched.
            for task in done:
                task.result()
            pending.difference_update(done)
