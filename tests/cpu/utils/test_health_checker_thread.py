# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import threading
from types import SimpleNamespace
from unittest.mock import Mock

from relax.utils import async_utils, health_system


def test_health_checker_recovery_can_stop_shared_loop(monkeypatch):
    shared = async_utils.get_async_loop()
    status = SimpleNamespace(
        get_unhealthy_services=SimpleNamespace(remote=Mock(return_value=["actor"])),
        get_service_health=SimpleNamespace(remote=Mock(return_value={"error": "training failed"})),
        get_stale_services=SimpleNamespace(remote=Mock(return_value=[])),
    )
    monkeypatch.setattr(health_system.ray, "get", lambda result: result)
    completed = threading.Event()
    observed = []

    def recover(role: str) -> None:
        observed.append((role, threading.current_thread()))
        async_utils.shutdown_async_loop()
        checker.stop()
        completed.set()

    checker = health_system.HealthChecker(status, recover, check_interval=0.01)
    try:
        checker.start()
        assert completed.wait(timeout=5), "Recovery callback could not shut down the shared loop"
        checker_thread = checker._thread
        checker_thread.join(timeout=2)
        assert not checker_thread.is_alive()
        assert not shared._thread.is_alive()
        assert observed == [("actor", checker_thread)]
        assert checker_thread is not shared._thread
        status.get_stale_services.remote.assert_not_called()
    finally:
        checker.stop()
        async_utils.shutdown_async_loop()
