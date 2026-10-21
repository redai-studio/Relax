# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
from concurrent.futures import CancelledError
from unittest.mock import MagicMock

from relax.utils import async_utils, health_system


def test_health_checker_restart_drains_training_loop(monkeypatch):
    monkeypatch.setattr(async_utils, "async_loop", None)
    monkeypatch.setattr(health_system.ray, "get", lambda value: value)
    status = MagicMock()
    status.get_unhealthy_services.remote.return_value = ["rollout"]
    status.get_service_health.remote.return_value = {"error": "engine died"}
    entered = threading.Event()
    cancelled = threading.Event()
    restarted = threading.Event()
    cleaned_up = threading.Event()

    async def pending_training():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up.set()

    def train():
        try:
            async_utils.run(pending_training())
        except CancelledError:
            cancelled.set()

    def restart(role):
        assert role == "rollout"
        checker.stop()
        async_utils.shutdown_async_loop()
        restarted.set()

    checker = health_system.HealthChecker(status, restart)
    training = threading.Thread(target=train, daemon=True)
    training.start()
    try:
        assert entered.wait(5)
        checker.start()
        assert restarted.wait(5)
        training.join(5)
        assert not training.is_alive()
        assert cancelled.is_set()
        assert cleaned_up.is_set()
        assert async_utils.async_loop is None
    finally:
        checker.stop()
        if async_utils.async_loop is not None:
            async_utils.shutdown_async_loop()

    async def resume():
        return 7

    try:
        assert async_utils.run(resume()) == 7
    finally:
        async_utils.shutdown_async_loop()
