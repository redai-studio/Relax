# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


__all__ = ["get_async_loop", "run"]


# Create a background event loop thread
class AsyncLoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro):
        # Schedule a coroutine onto the loop and block until it's done
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


# Create one global instance
async_loop = None


def get_async_loop():
    global async_loop
    if async_loop is None:
        async_loop = AsyncLoopThread()
    return async_loop


def shutdown_async_loop(timeout: float = 5.0):
    """Request loop shutdown and join its thread when called externally.

    Must be called before ``ray.shutdown()`` during global restart.  The call
    waits up to ``timeout`` for the event-loop thread when called externally. A
    call from the loop thread only requests shutdown; the loop stops after the
    current callback returns. The next call to :func:`run` lazily creates a
    fresh loop.
    """
    global async_loop
    if async_loop is None:
        return
    inst = async_loop
    async_loop = None
    loop = inst.loop
    loop.call_soon_threadsafe(loop.stop)
    if inst._thread is threading.current_thread():
        # A shutdown callback running on this loop cannot join its own thread.
        logger.warning("shutdown_async_loop() called from the async loop thread; skipping join")
    else:
        inst._thread.join(timeout=timeout)


def run(coro):
    """Run a coroutine in the background event loop."""
    return get_async_loop().run(coro)
