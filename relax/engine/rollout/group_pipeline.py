# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded CPU processing with a round publication barrier."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


class GroupPipeline:
    """One preparation owner and one scorer owner, with ordered backpressure.

    A pending slot is held until BOTH preparation and scoring finish. Thus a
    slow judge cannot accumulate an unbounded queue of images/trajectories. No
    worker waits on work scheduled to its own executor.
    """

    def __init__(self, prepare: Callable[..., Any], score: Callable[[Any], Any], limit: int = 2) -> None:
        if type(limit) is not int or limit < 1:
            raise ValueError("overlap_pending_groups must be a positive integer")
        self._prepare = prepare
        self._score = score
        self._limit = limit
        self._pending: deque[Future] = deque()
        self._writers = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout-prepare")
        self._scorers = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout-reward")

    def _process(self, *args: Any) -> tuple[Any, Future]:
        group = self._prepare(*args)
        return group, self._scorers.submit(self._score, group)

    def reserve(self, slots: int = 1) -> None:
        """Reserve response capacity before generation."""
        if type(slots) is not int or not 1 <= slots <= self._limit:
            raise ValueError("Invalid group capacity reservation")
        while len(self._pending) + slots > self._limit:
            self._finish_oldest()
        for future in self._pending:
            if future.done():
                # Surface a writer failure even before this slot is drained.
                _, reward = future.result()
                if reward.done():
                    reward.result()

    def submit(self, *args: Any) -> None:
        if len(self._pending) >= self._limit:
            raise RuntimeError("GroupPipeline.reserve() is required before submitting")
        self._pending.append(self._writers.submit(self._process, *args))

    def _finish_oldest(self) -> None:
        _, reward = self._pending.popleft().result()
        reward.result()

    def finish(self) -> None:
        while self._pending:
            self._finish_oldest()

    def close(self) -> None:
        # Writers can still submit a reward task; join them FIRST. On failure
        # callers discard this round, but must not leave background mutations.
        self._writers.shutdown(wait=True, cancel_futures=True)
        self._scorers.shutdown(wait=True, cancel_futures=True)
