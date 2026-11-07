"""Reusable source I/O workers preserve context, failure draining and no-op
ranks."""

import contextvars
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from relax.distributed.weight_sync import source_io


def test_reusable_worker_retains_no_context_from_previous_capture():
    capture = contextvars.ContextVar("source-capture")
    threads = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        for value in range(3):
            token = capture.set(value)
            try:

                def read():
                    threads.append(threading.get_ident())
                    return capture.get()

                assert source_io.coordinated_io(read, lambda state: [state], stage="read", executor=executor) == value
            finally:
                capture.reset(token)
    assert len(set(threads)) == 1


def test_no_local_work_still_agrees_without_creating_a_worker(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("No local file work needs no thread pool")

    monkeypatch.setattr(source_io, "ThreadPoolExecutor", forbidden)
    states = []

    def agree(state):
        states.append(state)
        return [state]

    assert source_io.coordinated_io(None, agree, stage="non-owner") is None
    assert states == [{"done": True, "error": None, "expired": False}]


def test_collective_failure_drains_reusable_worker_before_releasing_inputs():
    finished = threading.Event()

    def slow_write():
        time.sleep(0.1)
        finished.set()

    def fail_agreement(state):
        raise RuntimeError("collective failure")

    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match="collective failure"):
            source_io.coordinated_io(slow_write, fail_agreement, stage="write", executor=executor)
        assert finished.is_set()
        assert (
            source_io.coordinated_io(lambda: "retry", lambda state: [state], stage="retry", executor=executor)
            == "retry"
        )
