# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.distributed.ray import model_router as module


@pytest.mark.parametrize("registers", [True, False])
def test_model_router_admission_requires_confirmed_head_registration(monkeypatch, registers):
    router = module.ModelRouter()
    router.endpoint = ("localhost", 8000)
    head = SimpleNamespace(configure_model_router=SimpleNamespace(remote=Mock(return_value=registers)))
    response = Mock()
    monkeypatch.setattr(module.requests, "get", Mock(return_value=response))
    monkeypatch.setattr(module.ray, "get", lambda ref, **kwargs: ref)
    if registers:
        router.activate([head, None])
        assert router.ready
    else:
        with pytest.raises(RuntimeError, match="registration is unconfirmed"):
            router.activate([head])
        assert not router.ready
    head.configure_model_router.remote.assert_called_once_with("localhost", 8000)


def test_model_router_transport_failure_closes_admission(monkeypatch):
    router = module.ModelRouter()
    router.endpoint = ("localhost", 8000)
    router.ready = True
    get = Mock(side_effect=ConnectionError("router unavailable"))
    monkeypatch.setattr(module.requests, "get", get)
    head = SimpleNamespace(configure_model_router=SimpleNamespace(remote=Mock()))
    with pytest.raises(ConnectionError):
        router.activate([head])
    assert not router.ready
    head.configure_model_router.remote.assert_not_called()


def test_model_router_retains_process_until_exit_confirmed():
    router = module.ModelRouter()
    process = Mock()
    process.is_alive.return_value = True
    router.processes.append(process)
    with pytest.raises(RuntimeError, match="exit is unconfirmed"):
        router.close()
    assert router.processes == [process]
    process.is_alive.return_value = False
    router.close()
    assert router.processes == []
