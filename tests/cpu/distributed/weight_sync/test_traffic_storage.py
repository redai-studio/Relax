# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""File writes and asynchronous payload counters retain separate scopes."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest

from relax.distributed.weight_sync import traffic
from relax.distributed.weight_sync.store import atomic_write


class Recorder:
    def __init__(self):
        self.events = []
        self.missing = []

    def transferred(self, stage, size, **fields):
        self.events.append({"stage": stage, "bytes": size, **fields})

    def unmeasured(self, stage, reason):
        self.missing.append((stage, reason))


def test_real_file_writes_preserve_context_scope_and_attempt(tmp_path):
    outer, inner = Recorder(), Recorder()
    with traffic.capture(outer, attempt=2), ThreadPoolExecutor(1) as pool:
        pool.submit(copy_context().run, atomic_write, tmp_path / "a", b"123").result()
        with traffic.capture(inner, attempt=3):
            atomic_write(tmp_path / "b", b"12345")
        atomic_write(tmp_path / "c", b"7")
    atomic_write(tmp_path / "outside", b"0000")
    assert [(e["bytes"], e["scope"], e["attempt"]) for e in outer.events] == [
        (3, "file_write", 2),
        (1, "file_write", 2),
    ]
    assert [(e["bytes"], e["scope"], e["attempt"]) for e in inner.events] == [(5, "file_write", 3)]
    assert (tmp_path / "a").read_bytes() == b"123"
    assert traffic.current() is None


def test_async_completion_records_once_after_success():
    class Work:
        results = iter([False, True, True])

        def wait(self):
            return next(self.results)

    sink = Recorder()
    with traffic.capture(sink, attempt=4):
        work = traffic._complete(Work(), 123, "rank0-rank1", True)
    assert work.wait() is False
    assert sink.events == []
    assert work.wait() is True
    assert work.wait() is True
    assert sink.events == [
        {"stage": "G", "bytes": 123, "link": "rank0-rank1", "scope": "collective_payload", "attempt": 4}
    ]


def test_async_error_preserves_original_failure_and_partial_coverage():
    class Work:
        def wait(self):
            raise OSError("transport interrupted")

    sink = Recorder()
    with traffic.capture(sink):
        work = traffic._complete(Work(), 128, "rank0-rank1", True)
    with pytest.raises(OSError, match="transport interrupted"):
        work.wait()
    assert sink.events == []
    assert sink.missing[0][0] == "G"
    assert "rank0-rank1" in sink.missing[0][1]


def test_network_counter_delta_retains_shared_link_attribution():
    before = {"lo": {"tx_bytes": 5, "rx_bytes": 8}}
    after = {"lo": {"tx_bytes": 20, "rx_bytes": 23}}
    observed = traffic.interface_delta(before, after)
    assert observed["interfaces"]["lo"] == {"tx_bytes": 15, "rx_bytes": 15}
    assert observed["attribution"] == "includes_concurrent_traffic"
    with pytest.raises(ValueError, match="reset"):
        traffic.interface_delta(after, before)
    with pytest.raises(ValueError, match="set changed"):
        traffic.interface_delta(before, {})
