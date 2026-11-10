# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.distributed.checkpoint_service.backends.shard_p2p.remote import transport


class Clock:
    def __init__(self):
        self.now = 0.0

    def sleep(self, duration):
        self.now += duration


class Agent:
    def __init__(self, states, failure=None):
        self.states = {peer: list(values) for peer, values in states.items()}
        self.failure = failure
        self.events = []
        self.posted = set()

    def event(self, operation, peer):
        self.events.append((operation, peer))
        if (operation, peer) == self.failure:
            raise RuntimeError(f"injected {operation} {peer}")

    def initialize_xfer(self, operation, source, target, peer):
        assert operation == "WRITE"
        self.event("initialize", peer)
        return peer

    def transfer(self, handle):
        self.event("post", handle)
        self.posted.add(handle)
        return self.states[handle].pop(0)

    def check_xfer_state(self, handle):
        assert self.posted == self.states.keys(), "Polling started before every peer was posted"
        self.event("poll", handle)
        return self.states[handle].pop(0)

    def release_xfer_handle(self, handle):
        self.event("release", handle)

    def get_xfer_descs(self, values, memory):
        assert memory == "VRAM"
        return values


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(transport.time, "monotonic", lambda: value.now)
    monkeypatch.setattr(transport.time, "sleep", value.sleep)
    monkeypatch.setattr(transport.torch.cuda, "set_device", Mock())
    monkeypatch.setattr(transport, "_FAILED_WRITERS", [])
    return value


def writer_for(agent):
    writer = transport.BufferedWriter.__new__(transport.BufferedWriter)
    writer.agent = agent
    writer.failed = False
    writer.profile = {}
    writer.timeout = 1.0
    writer._active_range_batch = None
    writer._retained_failure = False
    writer.buffers = [object(), object()]
    writer.registrations = [object(), object()]
    writer.pending = [None, None]
    writer.next_index = 0
    return writer


def requests_for(agent):
    return [(object(), object(), peer) for peer in agent.states]


@pytest.mark.parametrize("peers", [2, 4, 8])
def test_every_peer_is_posted_before_any_wait_and_all_complete_before_release(clock, peers):
    agent = Agent({index: ["PROC"] * (peers - index) + ["DONE"] for index in range(peers)})
    writer = writer_for(agent)
    writer._transfer_many(requests_for(agent), [object()], 1.0)
    operations = [event[0] for event in agent.events]
    assert operations[:peers] == ["initialize"] * peers
    assert operations[peers : 2 * peers] == ["post"] * peers
    assert operations[-peers:] == ["release"] * peers
    assert writer.profile["max_parallel_inflight"] == peers
    assert writer.profile["parallel_batches"] == 1
    assert writer.profile["parallel_batch_s"] == clock.now
    assert not writer.failed and writer._active_range_batch is None
    assert transport._FAILED_WRITERS == []


def test_immediate_done_peer_does_not_release_slow_peer_or_its_buffer(clock):
    agent = Agent({0: ["DONE"], 1: ["PROC", "PROC", "DONE"]})
    writer = writer_for(agent)
    writer._transfer_many(requests_for(agent), [object()], 1.0)
    assert agent.events[-2:] == [("release", 0), ("release", 1)]
    assert writer.profile["max_parallel_inflight"] == 1
    assert writer.profile["parallel_peer_0_s"] == 0
    assert writer.profile["parallel_peer_1_s"] > 0


@pytest.mark.parametrize("operation", ["initialize", "post", "poll"])
def test_second_peer_failure_keeps_unconfirmed_handles_and_sources_alive(clock, operation):
    agent = Agent({0: ["PROC", "PROC"], 1: ["PROC", "DONE"]}, failure=(operation, 1))
    writer = writer_for(agent)
    sources = [object()]
    requests = requests_for(agent)
    with pytest.raises(RuntimeError, match=f"injected {operation} 1"):
        writer._transfer_many(requests, sources, 1.0)
    assert writer.failed and writer._active_range_batch["sources"] is sources
    assert writer._active_range_batch["requests"] is requests
    assert not any(event[0] == "release" for event in agent.events)
    retained = weakref.ref(writer)
    del writer
    gc.collect()
    assert retained() is transport._FAILED_WRITERS[0]
    assert retained().agent is agent


@pytest.mark.parametrize("bad_state", ["ERR", "unexpected"])
def test_poll_error_releases_only_done_peer_and_retains_failed_handle(clock, bad_state):
    agent = Agent({0: ["DONE"], 1: ["PROC", bad_state]})
    writer = writer_for(agent)
    with pytest.raises(RuntimeError, match="did not complete"):
        writer._transfer_many(requests_for(agent), [object()], 1.0)
    assert ("release", 0) in agent.events and ("release", 1) not in agent.events
    assert writer.failed and len(transport._FAILED_WRITERS) == 1
    assert writer._active_range_batch["records"][1]["state"] == bad_state


def test_timeout_is_one_batch_deadline_and_never_releases_active_handles(clock):
    agent = Agent({0: ["PROC"] * 20, 1: ["PROC"] * 20})
    writer = writer_for(agent)
    with pytest.raises(TimeoutError, match="total deadline"):
        writer._transfer_many(requests_for(agent), [object()], 0.001)
    assert clock.now == 0.001
    assert not any(event[0] == "release" for event in agent.events)
    assert writer.failed and len(writer._active_range_batch["records"]) == 2


@pytest.mark.parametrize("stage", ["before_initialize", "after_initialize", "after_first_post"])
def test_deadline_is_not_reset_between_native_calls(clock, stage):
    agent = Agent({0: ["PROC", "DONE"], 1: ["PROC", "DONE"]})
    writer = writer_for(agent)
    original = agent.event

    def advance(operation, peer):
        original(operation, peer)
        if (stage == "after_initialize" and operation == "initialize" and peer == 0) or (
            stage == "after_first_post" and operation == "post" and peer == 0
        ):
            clock.now = 2.0

    agent.event = advance
    if stage == "before_initialize":
        clock.now = 2.0
    with pytest.raises(TimeoutError, match="total deadline"):
        writer._transfer_many(requests_for(agent), [object()], 1.0)
    assert ("post", 1) not in agent.events
    assert not any(event[0] == "release" for event in agent.events)
    assert writer.failed


def test_release_failure_still_cleans_every_done_handle_and_retains_failed_release(clock):
    agent = Agent({0: ["DONE"], 1: ["DONE"]}, failure=("release", 0))
    writer = writer_for(agent)
    with pytest.raises(RuntimeError, match="injected release 0"):
        writer._transfer_many(requests_for(agent), [object()], 1.0)
    assert agent.events[-2:] == [("release", 0), ("release", 1)]
    assert writer.failed
    assert [row["released"] for row in writer._active_range_batch["records"]] == [False, True]


def test_release_exception_does_not_replace_original_write_error(clock):
    agent = Agent({0: ["DONE"], 1: ["ERR"]}, failure=("release", 0))
    writer = writer_for(agent)
    with pytest.raises(RuntimeError, match="did not complete: ERR"):
        writer._transfer_many(requests_for(agent), [object()], 1.0)
    assert ("release", 1) not in agent.events
    assert writer.failed


def test_host_future_timeout_poisons_both_buffers_and_queued_tasks(clock):
    writer = writer_for(Agent({}))
    future = Mock()
    future.result.side_effect = TimeoutError("host wait expired")
    writer.pending[0] = future
    with pytest.raises(TimeoutError, match="host wait expired"):
        writer.acquire()
    assert writer.pending[0] is future and writer.failed
    with pytest.raises(RuntimeError, match="Cannot reuse"):
        writer.acquire()
    ready = Mock()
    with pytest.raises(RuntimeError, match="queued write"):
        writer._write_ranges([], {}, ready)
    ready.synchronize.assert_not_called()
    with pytest.raises(RuntimeError, match="queued write"):
        writer._write({}, {}, ready)
    assert len(transport._FAILED_WRITERS) == 1


@pytest.mark.parametrize("failure", ["initialize", "post", "poll", "timeout", "ERR", "unexpected"])
def test_single_peer_failure_retains_native_state_and_actual_source(clock, failure):
    native_failure = (failure, 0) if failure in ("initialize", "post", "poll") else None
    states = (
        ["PROC"] * 10 if failure == "timeout" else ["PROC", failure if failure in ("ERR", "unexpected") else "DONE"]
    )
    agent = Agent({0: states}, failure=native_failure)
    writer = writer_for(agent)
    source = transport.torch.ones(4)
    source_ref = weakref.ref(source)
    descriptions = (object(), object())
    with pytest.raises((RuntimeError, TimeoutError)):
        writer._transfer(*descriptions, 0, deadline=0.001, sources=source)
    assert writer.failed
    assert not any(event[0] == "release" for event in agent.events)
    assert writer._active_range_batch["requests"] == [(*descriptions, 0)]
    assert writer._active_range_batch["sources"] is source
    with pytest.raises(RuntimeError, match="Cannot reuse"):
        writer.acquire()
    with pytest.raises(RuntimeError, match="unconfirmed write"):
        writer.submit(0, {}, {})
    writer_ref = weakref.ref(writer)
    del writer, source
    gc.collect()
    assert writer_ref() is transport._FAILED_WRITERS[0]
    assert source_ref() is writer_ref()._active_range_batch["sources"]


def test_single_peer_done_releases_once_and_does_not_retain_sources(clock):
    agent = Agent({0: ["PROC", "DONE"]})
    writer = writer_for(agent)
    source = transport.torch.ones(4)
    source_ref = weakref.ref(source)
    writer._transfer(object(), object(), 0, sources=source)
    assert agent.events.count(("release", 0)) == 1
    assert not writer.failed and writer._active_range_batch is None
    assert transport._FAILED_WRITERS == []
    del source
    gc.collect()
    assert source_ref() is None


def test_real_worker_timeout_during_native_post_never_posts_next_peer(clock):
    agent = Agent({0: ["PROC", "DONE"], 1: ["PROC", "DONE"]})
    writer = writer_for(agent)
    writer.timeout = 0.0  # _wait still allows one second of host-side grace.
    entered, resume = threading.Event(), threading.Event()
    post = agent.transfer
    sources = [object()]

    def blocking_post(handle):
        state = post(handle)
        if handle == 0:
            entered.set()
            assert resume.wait(timeout=5)
        return state

    agent.transfer = blocking_post
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(writer._transfer_many, requests_for(agent), sources, 10.0)
        writer.pending[0] = future
        assert entered.wait(timeout=3)
        queued = pool.submit(writer._write, {}, {})
        try:
            with pytest.raises(TimeoutError):
                writer._wait(0)
            assert writer.pending[0] is future and writer.failed
            assert writer._active_range_batch["sources"] is sources
        finally:
            resume.set()
        with pytest.raises(RuntimeError, match="failed while"):
            future.result(timeout=3)
        with pytest.raises(RuntimeError, match="queued write"):
            queued.result(timeout=3)
    assert ("post", 0) in agent.events and ("post", 1) not in agent.events
    assert not any(event[0] == "release" for event in agent.events)
    assert writer is transport._FAILED_WRITERS[0]


@pytest.mark.parametrize("peers", [1, 2])
def test_range_dispatch_uses_actual_single_or_multiple_peers(clock, peers):
    writer = writer_for(Agent({}))
    writer._transfer = Mock()
    writer._transfer_many = Mock()
    writer.targets = {str(i): (f"peer-{i}", "generation") for i in range(peers)}
    tensor = SimpleNamespace(device=SimpleNamespace(index=0), data_ptr=lambda: 1024)
    writes = [
        (dict(target=str(i), param="weight", dst_offset=4, rows=2, row_bytes=8, dst_stride=16), tensor)
        for i in range(peers)
    ]
    manifests = {str(i): dict(gpu_id=i, params={"weight": dict(address=2048)}) for i in range(peers)}
    writer._write_ranges(writes, manifests, Mock())
    if peers > 1:
        writer._transfer.assert_not_called()
        requests, sources, deadline = writer._transfer_many.call_args.args
        assert sources is writes and deadline == 1.0 and len(requests) == peers
        assert requests[0][0] == [(1024, 8, 0), (1032, 8, 0)]
        assert requests[0][1] == [(2052, 8, 0), (2068, 8, 0)]
    else:
        assert writer._transfer.call_count == peers
        writer._transfer_many.assert_not_called()


def test_ready_wait_consumes_the_same_deadline_before_any_post(clock):
    writer = writer_for(Agent({0: ["DONE"], 1: ["DONE"]}))
    writer.targets = {str(i): (i, "generation") for i in range(2)}
    tensor = SimpleNamespace(device=SimpleNamespace(index=0), data_ptr=lambda: 1024)
    writes = [
        (dict(target=str(i), param="weight", dst_offset=0, rows=1, row_bytes=8, dst_stride=8), tensor)
        for i in range(2)
    ]
    manifests = {str(i): dict(gpu_id=i, params={"weight": dict(address=2048)}) for i in range(2)}
    ready = SimpleNamespace(synchronize=lambda: clock.sleep(2.0))
    with pytest.raises(TimeoutError, match="total deadline"):
        writer._write_ranges(writes, manifests, ready)
    assert writer.agent.events == [] and writer.failed
