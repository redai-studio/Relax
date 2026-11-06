# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Quiesce driver-owned Serve callbacks before destroying the Ray core
worker."""

import asyncio
import concurrent.futures
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class ServeRouterLoop:
    loop: asyncio.AbstractEventLoop
    thread: threading.Thread
    polls: tuple[Any, ...]


def quiesce_serve_routers(timeout: float = 30.0) -> list[ServeRouterLoop]:
    """Stop cached handles' dedicated/shared polls on their owning event loops.

    Ray 2.56's handle shutdown only stops metrics. Use its shared-router
    registry to include live routers whose base handle has been evicted from
    the cache. This is a terminal driver operation, after all application work
    has stopped. Unsupported initialized router layouts fail explicitly instead
    of racing core-worker destruction.
    """
    from ray.serve import context
    from ray.serve._private.router import SharedRouterLongPollClient, SingletonThreadRouter

    client = context._global_client
    grouped: dict[asyncio.AbstractEventLoop, list[Any]] = {}
    handles = [] if client is None else list(client.handle_cache.values())
    for handle in handles:
        wrapper = vars(handle).get("_router")
        if wrapper is None:
            continue
        if not isinstance(wrapper, SingletonThreadRouter):
            raise RuntimeError(f"Unsupported Serve router during shutdown: {type(wrapper).__name__}")
        router = getattr(wrapper, "__dict__", {}).get("_asyncio_router")
        if router is None:
            raise RuntimeError(f"Unsupported Serve router during shutdown: {type(wrapper).__name__}")
        grouped.setdefault(router._event_loop, []).append(router)
    singleton_loop = SingletonThreadRouter._asyncio_loop
    if singleton_loop is not None and singleton_loop.is_running() and singleton_loop not in grouped:
        raise RuntimeError("Running Serve router loop has no cached handle for shutdown")

    def stop_polls(
        loop: asyncio.AbstractEventLoop,
        routers: list[Any],
        completed: concurrent.futures.Future[ServeRouterLoop],
    ) -> None:
        try:
            polls = {}
            for router in routers:
                cache = SharedRouterLongPollClient.get_or_create
                misses = cache.cache_info().misses
                # The positional form must match Ray's original lru_cache key.
                shared = cache(router._controller_handle, loop)
                if cache.cache_info().misses != misses:
                    shared.long_poll_client.stop()
                    raise RuntimeError("Serve shared poll registry was not found during shutdown")
                polls[id(shared.long_poll_client)] = shared.long_poll_client
                polls[id(router.long_poll_client)] = router.long_poll_client
                for registered in shared.routers.values():
                    for member in list(registered):
                        polls[id(member.long_poll_client)] = member.long_poll_client
            for poll in polls.values():
                poll.stop()
            completed.set_result(ServeRouterLoop(loop, threading.current_thread(), tuple(polls.values())))
        except Exception as error:
            completed.set_exception(error)

    pending = []
    for loop, routers in grouped.items():
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if loop is current_loop or not loop.is_running():
            raise RuntimeError("Serve shutdown requires a running router loop on another thread")
        completed = concurrent.futures.Future()
        loop.call_soon_threadsafe(stop_polls, loop, routers, completed)
        pending.append(completed)
    return [completed.result(timeout=timeout) for completed in pending]


def shutdown_serve_router_loops(owners: list[ServeRouterLoop], timeout: float = 30.0) -> None:
    """Drain and join router loops after serve.shutdown has closed their
    metrics."""

    async def drain() -> None:
        current = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks() if task is not current]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    for owner in owners:
        if owner.thread is threading.current_thread():
            raise RuntimeError("Cannot join the current Serve router thread")
        future = asyncio.run_coroutine_threadsafe(drain(), owner.loop)
        future.result(timeout=timeout)
        owner.loop.call_soon_threadsafe(owner.loop.stop)
        owner.thread.join(timeout=timeout)
        if owner.thread.is_alive():
            raise TimeoutError("Serve router thread did not stop before Ray shutdown")
        owner.loop.close()
