# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real asyncio-thread ordering tests; no Ray cluster or CUDA
initialization."""

import asyncio
import concurrent.futures
import sys
import threading
from functools import lru_cache
from types import ModuleType, SimpleNamespace

import pytest

from relax.utils import async_utils
from relax.utils.ray_shutdown import quiesce_serve_routers, shutdown_serve_router_loops


@pytest.fixture
def router_state(monkeypatch):
    loops = []
    cache = {}

    class Poll:
        def __init__(self, loop):
            self.loop = loop
            self.is_running = True
            self.stop_thread = None
            self.calls = 0

        def stop(self):
            self.is_running = False
            self.stop_thread = threading.current_thread()

        def poll(self):
            if self.is_running:
                self.calls += 1

    class Shared:
        def __init__(self, controller, loop):
            self.long_poll_client = Poll(loop)
            self.routers = {}

        @classmethod
        @lru_cache(maxsize=None)
        def get_or_create(cls, controller, loop):
            return cls(controller, loop)

    class Singleton:
        _asyncio_loop = None

        def __init__(self, router):
            self._asyncio_router = router

    def new_loop():
        loop = asyncio.new_event_loop()
        started = threading.Event()
        loop.call_soon(started.set)
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        assert started.wait(2)
        loops.append((loop, thread))
        return loop

    def add_router(loop, key, *, cached=True):
        router = SimpleNamespace(_event_loop=loop, _controller_handle="controller", long_poll_client=Poll(loop))
        shared = Shared.get_or_create("controller", loop)
        shared.routers[key] = [router]
        if cached:
            cache[key] = SimpleNamespace(_router=Singleton(router))
        return router

    for name in ("ray", "ray.serve", "ray.serve._private"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    context = ModuleType("ray.serve.context")
    context._global_client = SimpleNamespace(handle_cache=cache)
    router_module = ModuleType("ray.serve._private.router")
    router_module.SharedRouterLongPollClient = Shared
    router_module.SingletonThreadRouter = Singleton
    monkeypatch.setitem(sys.modules, "ray.serve.context", context)
    monkeypatch.setitem(sys.modules, "ray.serve._private.router", router_module)
    yield SimpleNamespace(
        new_loop=new_loop, add_router=add_router, cache=cache, shared=Shared, singleton=Singleton, context=context
    )
    for loop, thread in loops:
        if not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
            thread.join(2)
            assert not thread.is_alive()
            loop.close()


def test_quiesce_includes_evicted_routers_and_joins_all_owned_threads(router_state):
    state = router_state
    first, second = state.new_loop(), state.new_loop()
    state.add_router(first, "cached")
    evicted = state.add_router(first, "evicted", cached=False)
    state.add_router(second, "other-loop")
    owners = quiesce_serve_routers(timeout=2)
    assert len(owners) == 2 and sum(len(owner.polls) for owner in owners) == 5
    assert not evicted.long_poll_client.is_running
    finalized = threading.Event()
    started = threading.Event()

    async def pending_watcher():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            finalized.set()

    asyncio.run_coroutine_threadsafe(pending_watcher(), first)
    assert started.wait(2)
    for owner in owners:
        done = concurrent.futures.Future()

        def run_late_callbacks(owner=owner, done=done):
            for poll in owner.polls:
                assert poll.stop_thread is threading.current_thread()
                poll.poll()
                assert poll.calls == 0
            done.set_result(True)

        # The loop remains usable for Serve's metrics shutdown after quiescence.
        owner.loop.call_soon_threadsafe(run_late_callbacks)
        assert done.result(timeout=2)
    shutdown_serve_router_loops(owners, timeout=2)
    assert finalized.is_set()
    assert all(not owner.thread.is_alive() and owner.loop.is_closed() for owner in owners)


def test_unsupported_router_fails_instead_of_silently_skipping_it(router_state):
    router_state.cache["unsupported"] = SimpleNamespace(_router=object())
    with pytest.raises(RuntimeError, match="Unsupported Serve router"):
        quiesce_serve_routers(timeout=2)


def test_current_loop_router_does_not_transfer_ownership_of_application_loop(router_state):
    loop = router_state.new_loop()
    router = router_state.add_router(loop, "app")
    router_state.cache["app"] = SimpleNamespace(_router=SimpleNamespace(_asyncio_router=router))
    with pytest.raises(RuntimeError, match="Unsupported Serve router"):
        quiesce_serve_routers(timeout=2)
    assert loop.is_running() and router.long_poll_client.is_running


def test_missing_shared_registry_fails_without_leaving_a_new_poll_running(router_state):
    loop = router_state.new_loop()
    router_state.add_router(loop, "cached")
    router_state.shared.get_or_create.cache_clear()
    with pytest.raises(RuntimeError, match="registry was not found"):
        quiesce_serve_routers(timeout=2)
    assert not router_state.shared.get_or_create("controller", loop).long_poll_client.is_running


@pytest.mark.parametrize("client_missing", [False, True])
def test_running_router_loop_without_cached_owner_cannot_be_silently_skipped(router_state, client_missing):
    router_state.singleton._asyncio_loop = router_state.new_loop()
    if client_missing:
        router_state.context._global_client = None
    with pytest.raises(RuntimeError, match="no cached handle"):
        quiesce_serve_routers(timeout=2)


def test_relax_async_loop_drains_watchers_and_can_restart():
    finalized = threading.Event()
    started = threading.Event()
    inst = async_utils.get_async_loop()

    async def watcher():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            finalized.set()

    asyncio.run_coroutine_threadsafe(watcher(), inst.loop)
    assert started.wait(2)
    async_utils.shutdown_async_loop(timeout=2)
    assert finalized.is_set() and not inst._thread.is_alive()
    assert async_utils.async_loop is None
    replacement = async_utils.get_async_loop()
    assert replacement is not inst
    async_utils.shutdown_async_loop(timeout=2)
