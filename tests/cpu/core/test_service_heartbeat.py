# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import threading
from types import SimpleNamespace

import pytest

from relax.core.service import Service


@pytest.mark.parametrize("in_flight", [False, True])
def test_stop_heartbeat_joins_idle_and_in_flight_submissions(in_flight):
    service = object.__new__(Service)
    service.role = "actor"
    service._heartbeat_thread = None
    service._stop_heartbeat = threading.Event()
    entered, finish_submission, stopped = (threading.Event() for _ in range(3))
    completed = []
    errors = []

    def submit(*args):
        entered.set()
        assert finish_submission.wait(5)
        assert not stopped.is_set()
        completed.append(args)

    service.healthy = SimpleNamespace(update_heartbeat=SimpleNamespace(remote=submit))
    if not in_flight:
        finish_submission.set()
    service._start_heartbeat()
    thread = service._heartbeat_thread
    assert entered.wait(5)

    def stop():
        try:
            service._stop_heartbeat_thread()
            stopped.set()
        except BaseException as error:
            errors.append(error)

    # Shutdown can overlap the health checker's service restart callback.
    closers = [threading.Thread(target=stop, daemon=True) for _ in range(2 if in_flight else 1)]
    for closer in closers:
        closer.start()
    try:
        assert service._stop_heartbeat.wait(5)
        if in_flight:
            assert not stopped.is_set() and all(closer.is_alive() for closer in closers)
        finish_submission.set()
        for closer in closers:
            closer.join(5)
        assert all(not closer.is_alive() for closer in closers) and not thread.is_alive()
        assert not errors and stopped.is_set()
        assert completed == [("actor", 0)]
        assert service._heartbeat_thread is None
        service._stop_heartbeat_thread()  # Repeated shutdown is harmless.
    finally:
        finish_submission.set()
        service._stop_heartbeat.set()
        thread.join(5)
        for closer in closers:
            closer.join(5)
