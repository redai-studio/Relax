# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

from relax.entrypoints import train


def _prepare_shutdown(monkeypatch, shutdown_error):
    controller = SimpleNamespace(shutdown=Mock(side_effect=shutdown_error))
    hard_exit = Mock()
    is_initialized = Mock(return_value=True)
    serve_shutdown = Mock()
    ray_shutdown = Mock()
    monkeypatch.setattr(train, "_ctrl", controller)
    monkeypatch.setattr(train, "_shutdown_done", False)
    monkeypatch.setattr(train, "_kernel_cache_heartbeat", None)
    monkeypatch.setattr(train, "_hard_exit", hard_exit)
    monkeypatch.setattr(train.ray, "is_initialized", is_initialized)
    monkeypatch.setattr(train.serve, "shutdown", serve_shutdown)
    monkeypatch.setattr(train.ray, "shutdown", ray_shutdown)
    return hard_exit, is_initialized, serve_shutdown, ray_shutdown


def test_ordinary_controller_shutdown_error_keeps_existing_cleanup(monkeypatch):
    calls = _prepare_shutdown(monkeypatch, RuntimeError("ordinary cleanup error"))
    hard_exit, is_initialized, serve_shutdown, ray_shutdown = calls

    train._graceful_shutdown(exit_code=1)

    is_initialized.assert_called_once_with()
    serve_shutdown.assert_called_once_with()
    ray_shutdown.assert_called_once_with()
    hard_exit.assert_called_once_with(1)
