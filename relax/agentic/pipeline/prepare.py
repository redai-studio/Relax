# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""DataSource-to-Prepare conversion for complete Groups."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Callable, Deque, Optional, Tuple, cast

from relax.agentic.pipeline import GroupInput
from relax.agentic.pipeline.runtime import RuntimeDomain, RuntimeGroupStream
from relax.utils.types import Sample


class PrepareDomain:
    """Own source fetches and the FIFO of Groups crossing into Runtime."""

    def __init__(
        self,
        scope_id: str,
        data_source: Any,
        runtime_domain: RuntimeDomain,
    ) -> None:
        self.scope_id = scope_id
        self.data_source = data_source
        self.runtime_domain = runtime_domain
        self._prelaunch_next_step = False
        self._next_group_sequence = 0
        self._prepare_fetch_task: Optional[asyncio.Task[None]] = None
        self._pending_group_inputs: Deque[GroupInput] = deque()
        self._warming_prepare_tasks: dict[
            asyncio.Task[Optional[RuntimeGroupStream]], Tuple[GroupInput, asyncio.Future[bool]]
        ] = {}
        self._ready_prepare_tasks: Deque[asyncio.Task[RuntimeGroupStream]] = deque()
        self._progress_callback: Callable[[], None] = lambda: None

    @property
    def counts(self) -> tuple[int, int]:
        warming = sum(
            not launch_decision.done() or launch_decision.result()
            for _group_input, launch_decision in self._warming_prepare_tasks.values()
        )
        return warming, self.ready_group_count

    @property
    def ready_group_count(self) -> int:
        return len(self._ready_prepare_tasks)

    def set_progress_callback(self, callback: Callable[[], None]) -> None:
        self._progress_callback = callback

    def open_step(self, *, prelaunch_next_step: bool) -> None:
        self._prelaunch_next_step = prelaunch_next_step

    def fill(self, demand: int = 0) -> None:
        target = self.runtime_domain.concurrency if self._prelaunch_next_step else demand
        gap = target - sum(self.counts)
        while gap > 0 and self._pending_group_inputs:
            self._start_prepare(self._pending_group_inputs.popleft())
            gap -= 1

        if self._prepare_fetch_task is None and gap > 0:
            self._prepare_fetch_task = asyncio.create_task(self._fetch_group_inputs(gap), name="agentic-prepare-fetch")
            self._prepare_fetch_task.add_done_callback(lambda _task: self._progress_callback())

    def collect(self) -> None:
        if self._prepare_fetch_task is not None and self._prepare_fetch_task.done():
            prepare_fetch_task = self._prepare_fetch_task
            self._prepare_fetch_task = None
            prepare_fetch_task.result()
        for task in tuple(self._ready_prepare_tasks):
            error = task.exception()
            if error is not None:
                self._ready_prepare_tasks.remove(task)
                raise error

    def take_ready(self, limit: int) -> Tuple[RuntimeGroupStream, ...]:
        count = min(max(limit, 0), len(self._ready_prepare_tasks))
        return tuple(self._ready_prepare_tasks.popleft().result() for _ in range(count))

    def close_step(self) -> None:
        if self._prelaunch_next_step:
            return
        deferred_inputs = []
        for group_input, launch_decision in self._warming_prepare_tasks.values():
            if not launch_decision.done():
                launch_decision.set_result(False)
                deferred_inputs.append(group_input)
        self._pending_group_inputs.extendleft(reversed(deferred_inputs))

    async def shutdown(self) -> None:
        prepare_fetch_task = self._prepare_fetch_task
        self._prepare_fetch_task = None
        self._pending_group_inputs.clear()
        tasks = list(self._warming_prepare_tasks)
        tasks.extend(self._ready_prepare_tasks)
        if prepare_fetch_task is not None:
            tasks.append(prepare_fetch_task)
        for task in tasks:
            task.cancel()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        self._warming_prepare_tasks.clear()
        self._ready_prepare_tasks.clear()
        for outcome in outcomes:
            if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
                raise outcome

    async def _fetch_group_inputs(self, requested_group_count: int) -> None:
        sample_groups = await self.data_source.get_samples.remote(requested_group_count)
        if not sample_groups:
            raise RuntimeError("data source returned no sample groups")
        if len(sample_groups) > requested_group_count:
            raise RuntimeError("data source returned more groups than requested")
        for sample_group in sample_groups:
            for sample in sample_group:
                if sample.status == Sample.Status.ABORTED:
                    raise RuntimeError(
                        "Agentic rollout no longer accepts aborted samples from the data source. Partial rollout state "
                        "must remain internal to runtime."
                    )
            sequence = self._next_group_sequence
            self._next_group_sequence += 1
            self._pending_group_inputs.append(
                GroupInput(group_id=f"{self.scope_id}:prepare_group_{sequence}", samples=sample_group)
            )
        if self._prelaunch_next_step:
            self.fill()

    def _start_prepare(self, group_input: GroupInput) -> None:
        launch_decision = asyncio.get_running_loop().create_future()
        prepare_task = asyncio.create_task(
            self.runtime_domain.prepare_group(group_input, launch_decision),
            name=f"agentic-prepare:{group_input.group_id}",
        )
        self._warming_prepare_tasks[prepare_task] = (group_input, launch_decision)
        prepare_task.add_done_callback(self._on_prepare_finished)

    def _on_prepare_finished(self, task: asyncio.Task[Optional[RuntimeGroupStream]]) -> None:
        self._warming_prepare_tasks.pop(task)
        if not task.cancelled() and (task.exception() is not None or task.result() is not None):
            self._ready_prepare_tasks.append(cast(asyncio.Task[RuntimeGroupStream], task))
        self._progress_callback()
