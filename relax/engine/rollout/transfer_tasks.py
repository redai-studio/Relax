# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Watch asynchronous transfers without cancelling dispatched storage
writes."""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from relax.distributed.settlement import UnsafeInferenceError


_T = TypeVar("_T")


def _record_cleanup_error(error: BaseException, result: BaseException) -> None:
    if isinstance(result, asyncio.CancelledError) or result is error:
        return
    if isinstance(error, asyncio.CancelledError):
        raise result
    note = f"rollout task cleanup also failed: {type(result).__name__}"
    try:
        add_note = getattr(error, "add_note", None)
        if callable(add_note) and note not in getattr(error, "__notes__", ()):
            add_note(note)
    except Exception:
        # Diagnostic metadata must never replace the original failure.
        pass


async def _cancel_after_error(error: BaseException, tasks: list[asyncio.Future]) -> None:
    """Cancel local requests, wait through repeated cancellation, and preserve
    the primary error.

    Storage transfers must use the settlement path below.
    """
    for task in tasks:
        if not task.done():
            task.cancel()
    drained = asyncio.gather(*tasks, return_exceptions=True)
    while not drained.done():
        try:
            await asyncio.shield(drained)
        except asyncio.CancelledError:
            continue
    for result in drained.result():
        if isinstance(result, BaseException):
            _record_cleanup_error(error, result)


async def _await_with_transfers(awaitable: Awaitable[_T], transfers: list[asyncio.Task]) -> _T:
    """Surface a failed write even while generation or input remains
    blocked."""
    foreground = asyncio.ensure_future(awaitable)
    try:
        while True:
            pending = []
            for task in transfers:
                if task.done():
                    task.result()
                else:
                    pending.append(task)
            if foreground.done():
                return foreground.result()
            await asyncio.wait([foreground, *pending], return_when=asyncio.FIRST_COMPLETED)
    except BaseException as error:
        await _cancel_after_error(error, [foreground])
        raise


async def _settle_after_error(
    error: BaseException,
    requests: list[asyncio.Task],
    transfers: list[asyncio.Task],
    *,
    recovery: bool,
    timeout: float = 60.0,
) -> None:
    """Drain local requests and settle storage writes before reporting failure.

    Cancelling a dispatched write makes its storage effects uncertain. Keep the
    existing recovery timeout and unsafe-write classification, including when
    the caller is cancelled repeatedly during settlement.
    """
    for task in requests:
        if not task.done():
            task.cancel()
    pending = [*requests, *transfers]
    if not pending:
        return
    waiter = asyncio.create_task(asyncio.wait(pending, timeout=timeout))
    while not waiter.done():
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError:
            continue
    done, unfinished = waiter.result()
    failures = [task.exception() for task in done if not task.cancelled() and task.exception() is not None]
    if unfinished:
        raise UnsafeInferenceError("Old generation work is still running; require global restart") from error
    if recovery and any(task.cancelled() or task.exception() is not None for task in transfers):
        raise UnsafeInferenceError("Old storage writes failed; their effects are uncertain") from error
    for result in failures:
        _record_cleanup_error(error, result)
