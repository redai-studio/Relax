# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded ordered CPU work with deadline and traffic context propagation."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Callable, Iterable, Iterator, TypeVar


_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


def ordered_map(function: Callable[[_Input], _Output], values: Iterable[_Input], workers: int) -> Iterator[_Output]:
    """Keep at most ``workers`` results in flight; writers remain on the
    caller."""
    if type(workers) is not int or workers < 1:
        raise ValueError("CPU worker count must be a positive integer")
    if workers == 1:
        yield from map(function, values)
        return
    iterator = iter(values)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="weight-block-io") as pool:
        pending = deque()
        try:
            for _ in range(workers):
                try:
                    value = next(iterator)
                except StopIteration:
                    break
                pending.append(pool.submit(copy_context().run, function, value))
            while pending:
                yield pending.popleft().result()
                try:
                    value = next(iterator)
                except StopIteration:
                    continue
                pending.append(pool.submit(copy_context().run, function, value))
        finally:
            for future in pending:
                future.cancel()
