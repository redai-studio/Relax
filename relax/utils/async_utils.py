# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading


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
    """Stop the global async event loop and **block** until its thread exits.

    Must be called before ``ray.shutdown()`` during global restart.  The call
    is blocking: it waits for the event-loop thread to fully terminate so that
    no C++ ObjectRefStream watchers survive into ``ray.shutdown()``. The next
    call to :func:`run` will lazily create a fresh loop.
    """
    global async_loop
    if async_loop is None:
        return
    inst = async_loop
    if inst._thread is threading.current_thread():
        raise RuntimeError("The async loop must be shut down from another thread")
    loop = inst.loop

    async def drain() -> None:
        tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # Complete cancellation callbacks before stopping, including the Future
    # awaited by run() in the training thread.
    asyncio.run_coroutine_threadsafe(drain(), loop).result(timeout=timeout)
    loop.call_soon_threadsafe(loop.stop)
    inst._thread.join(timeout=timeout)
    if inst._thread.is_alive():
        raise TimeoutError("The async loop did not stop before Ray shutdown")
    loop.close()
    async_loop = None


def run(coro):
    """Run a coroutine in the background event loop."""
    return get_async_loop().run(coro)
