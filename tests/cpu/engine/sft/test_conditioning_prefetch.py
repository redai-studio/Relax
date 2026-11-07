# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU read-ahead ownership, byte admission and cancellation coverage."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from relax.engine.sft.conditioning_prefetch import ConditioningPrefetch


class Lease:
    def __init__(self, key: str, nbytes: int) -> None:
        self.key, self.nbytes = key, nbytes
        self.closed = False
        self.unchanged = True
        self.data = {"value": key}

    def copy_to(self, device):
        assert not self.closed
        return {"value": self.key, "device": device}, self.nbytes

    def is_unchanged(self):
        return self.unchanged

    def close(self):
        self.closed = True


class Cache:
    def __init__(self, sizes):
        self.sizes = sizes
        self.calls = []
        self.leases = []
        self.entered = threading.Event()
        self.proceed = threading.Event()
        self.proceed.set()

    def lookup(self, key, *, max_bytes):
        self.calls.append((key, max_bytes, threading.get_ident()))
        self.entered.set()
        assert self.proceed.wait(5), "Test failed to release its controlled storage read"
        if key not in self.sizes or self.sizes[key] > max_bytes:
            return None
        lease = Lease(key, self.sizes[key])
        self.leases.append(lease)
        return lease


def drain(prefetch):
    """Fence the single executor without depending on scheduler sleeps."""
    prefetch._pool.submit(lambda: None).result(timeout=5)


def test_read_ahead_runs_storage_work_outside_the_training_thread():
    cache = Cache({"a": 8})
    cache.proceed.clear()
    prefetch = ConditioningPrefetch(cache, depth=2, max_bytes=16)
    try:
        assert prefetch.submit("a")
        assert cache.entered.wait(5)
        assert cache.calls[0][2] != threading.get_ident()
        cache.proceed.set()
        with prefetch.take("a") as lease:
            assert lease.copy_to("cpu") == ({"value": "a", "device": "cpu"}, 8)
            assert prefetch.stats()["bytes"] == 8
        assert cache.leases[0].closed
        assert prefetch.stats()["bytes"] == 0
    finally:
        cache.proceed.set()
        prefetch.close()


def test_byte_budget_includes_consumer_borrow_until_copy_finishes():
    cache = Cache({"a": 3, "b": 4})
    prefetch = ConditioningPrefetch(cache, depth=2, max_bytes=5)
    try:
        assert prefetch.submit("a")
        borrowed = prefetch.take("a")
        assert borrowed is not None
        assert prefetch.submit("b")
        drain(prefetch)
        assert prefetch.take("b") is None
        assert cache.calls[-1][1] == 2
        assert prefetch.stats()["bytes"] == 3
        borrowed.close()
        assert prefetch.submit("b"), "A budget miss must not become a permanent negative cache"
        with prefetch.take("b"):
            assert prefetch.stats()["bytes"] == 4
        assert prefetch.stats()["peak_bytes"] <= 5
    finally:
        prefetch.close()


def test_changed_future_window_discards_running_reads_without_leaking():
    cache = Cache({"old": 4, "new": 4})
    cache.proceed.clear()
    prefetch = ConditioningPrefetch(cache, depth=1, max_bytes=4)
    try:
        assert prefetch.set_window(["old"]) == 1
        assert cache.entered.wait(5)
        assert prefetch.set_window(["new"]) == 0, "An outstanding read still consumes the bounded slot"
        cache.proceed.set()
        drain(prefetch)
        assert cache.leases[0].closed
        assert prefetch.take("old") is None
        assert prefetch.set_window(["new"]) == 1
        with prefetch.take("new") as lease:
            assert lease.data == {"value": "new"}
        assert prefetch.stats()["bytes"] == 0
    finally:
        cache.proceed.set()
        prefetch.close()


def test_missing_entry_is_retried_after_online_publication():
    cache = Cache({})
    prefetch = ConditioningPrefetch(cache, depth=1, max_bytes=8)
    try:
        assert prefetch.submit("a")
        assert prefetch.take("a") is None
        cache.sizes["a"] = 8
        assert prefetch.submit("a")
        with prefetch.take("a"):
            pass
        assert prefetch.stats()["misses"] == 1
        assert prefetch.stats()["consumed"] == 1
    finally:
        prefetch.close()


def test_close_drains_inflight_storage_before_cache_can_close():
    cache = Cache({"a": 8})
    cache.proceed.clear()
    prefetch = ConditioningPrefetch(cache, depth=1, max_bytes=8)
    assert prefetch.submit("a")
    assert cache.entered.wait(5)
    started = threading.Event()

    def close():
        started.set()
        prefetch.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        done = pool.submit(close)
        assert started.wait(5)
        assert not done.done()
        cache.proceed.set()
        done.result(timeout=5)
    assert cache.leases[0].closed
    assert prefetch.stats()["bytes"] == 0
    assert prefetch.stats()["outstanding"] == 0
    assert not prefetch.submit("a")
    prefetch.close()


def test_close_does_not_invalidate_a_consumer_borrow():
    cache = Cache({"a": 8})
    prefetch = ConditioningPrefetch(cache, depth=1, max_bytes=8)
    assert prefetch.submit("a")
    lease = prefetch.take("a")
    prefetch.close()
    assert not cache.leases[0].closed
    assert prefetch.stats()["bytes"] == 8
    lease.close()
    assert cache.leases[0].closed
    assert prefetch.stats()["bytes"] == 0


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("read_finished", [False, True])
def test_interrupted_take_discards_popped_ticket_until_read_finishes(monkeypatch, error_type, read_finished):
    cache = Cache({"a": 8})
    cache.proceed.clear()
    prefetch = ConditioningPrefetch(cache, depth=1, max_bytes=8)
    try:
        assert prefetch.submit("a")
        assert cache.entered.wait(5)
        ticket = prefetch._pending["a"]
        result = ticket.future.result

        def interrupt_wait():
            if read_finished:
                cache.proceed.set()
                result(timeout=5)
            raise error_type("consumer cancelled")

        monkeypatch.setattr(ticket.future, "result", interrupt_wait)
        with pytest.raises(error_type, match="consumer cancelled"):
            prefetch.take("a")
        assert ticket.discarded
        if not read_finished:
            assert prefetch.stats()["outstanding"] == 1, "Do not free the slot of an active lookup"
            assert not prefetch.submit("another")
        cache.proceed.set()
        drain(prefetch)
        assert cache.leases[0].closed
        assert prefetch.stats()["bytes"] == prefetch.stats()["outstanding"] == 0
        prefetch.close()
        assert prefetch.stats()["drained"]
    finally:
        cache.proceed.set()
        prefetch.close()


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_interrupted_identity_check_releases_unreturned_lease(monkeypatch, error_type):
    cache = Cache({"a": 8})
    prefetch = ConditioningPrefetch(cache, depth=1, max_bytes=8)
    try:
        assert prefetch.submit("a")
        drain(prefetch)

        def interrupt_check():
            raise error_type("identity check cancelled")

        monkeypatch.setattr(cache.leases[0], "is_unchanged", interrupt_check)
        with pytest.raises(error_type, match="identity check cancelled"):
            prefetch.take("a")
        assert cache.leases[0].closed
        assert prefetch.stats()["bytes"] == prefetch.stats()["outstanding"] == 0
    finally:
        prefetch.close()


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_interrupted_close_can_retry_and_drain_the_running_reader(monkeypatch, error_type):
    cache = Cache({"a": 8})
    cache.proceed.clear()
    prefetch = ConditioningPrefetch(cache, depth=1, max_bytes=8)
    assert prefetch.submit("a")
    assert cache.entered.wait(5)
    shutdown = prefetch._pool.shutdown
    calls = []

    def interrupt_once(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise error_type("drain interrupted")
        return shutdown(**kwargs)

    monkeypatch.setattr(prefetch._pool, "shutdown", interrupt_once)
    with pytest.raises(error_type, match="drain interrupted"):
        prefetch.close()
    assert prefetch.stats()["closed"] and not prefetch.stats()["drained"]
    assert not prefetch.submit("another")
    retry_started = threading.Event()

    def retry():
        retry_started.set()
        prefetch.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        completed = pool.submit(retry)
        assert retry_started.wait(5)
        assert not completed.done(), "Retry must still wait for the running reader"
        cache.proceed.set()
        completed.result(timeout=5)
    assert len(calls) == 2
    assert cache.leases[0].closed
    assert prefetch.stats()["drained"]
    assert prefetch.stats()["bytes"] == prefetch.stats()["outstanding"] == 0
    prefetch.close()
    assert len(calls) == 2


def test_changed_file_identity_rejects_prefetched_lease():
    cache = Cache({"a": 8})
    prefetch = ConditioningPrefetch(cache, depth=1, max_bytes=8)
    try:
        assert prefetch.submit("a")
        drain(prefetch)
        cache.leases[0].unchanged = False
        assert prefetch.take("a") is None
        assert cache.leases[0].closed
        assert prefetch.stats()["stale"] == 1
        assert prefetch.stats()["bytes"] == 0
    finally:
        prefetch.close()


def test_storage_failure_falls_back_and_does_not_poison_future_window():
    cache = Cache({"a": 8})
    normal_lookup = cache.lookup

    def fail(*args, **kwargs):
        raise OSError("controlled read failure")

    cache.lookup = fail
    prefetch = ConditioningPrefetch(cache, depth=1, max_bytes=8)
    try:
        assert prefetch.submit("a")
        assert prefetch.take("a") is None
        assert prefetch.stats()["errors"] == 1
        cache.lookup = normal_lookup
        assert prefetch.submit("a")
        with prefetch.take("a"):
            pass
    finally:
        prefetch.close()


@pytest.mark.parametrize("depth,max_bytes", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_prefetch_rejects_invalid_capacity(depth, max_bytes):
    with pytest.raises(ValueError):
        ConditioningPrefetch(Cache({}), depth=depth, max_bytes=max_bytes)
