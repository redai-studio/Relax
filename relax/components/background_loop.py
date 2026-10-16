# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Async waiting for component-owned blocking loops, without deployment
dependencies."""

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class _RunState:
    done: asyncio.Event = field(default_factory=asyncio.Event)
    thread: Optional[threading.Thread] = None
    error: Optional[Exception] = None


class BackgroundLoopRunner:
    """Share one live thread among callers on the same event loop.

    Each run owns its completion and error, even if a later run starts before
    its waiters resume. Cancelling a waiter does not stop the thread. Stop
    signals, health reporting and restart decisions belong to the component.
    """

    def __init__(self) -> None:
        self._state: Optional[_RunState] = None

    async def run(self, target: Callable[[], None], *, before_start: Optional[Callable[[], None]] = None) -> None:
        state = self._state
        if state is None or not state.thread.is_alive():
            if before_start is not None:
                before_start()
            loop = asyncio.get_running_loop()
            state = _RunState()

            def thread_target() -> None:
                try:
                    target()
                except Exception as exc:
                    state.error = exc
                finally:
                    loop.call_soon_threadsafe(state.done.set)

            state.thread = threading.Thread(target=thread_target, daemon=True)
            state.thread.start()
            self._state = state
        await state.done.wait()
        if state.error is not None:
            raise state.error

    async def join(self, timeout: float) -> bool:
        """Wait without blocking the event loop; False means the thread is
        still alive.

        A timeout never cancels computation or releases ownership of the
        thread. Worker exceptions are delivered by run(), not by shutdown
        waiting.
        """
        state = self._state
        if state is None:
            return True
        await asyncio.get_running_loop().run_in_executor(None, state.thread.join, timeout)
        return not state.thread.is_alive()
