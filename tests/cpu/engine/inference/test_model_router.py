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


@pytest.mark.parametrize("launch_fails", [False, True])
def test_static_router_owns_only_its_process_and_does_not_inherit_policy_routing(monkeypatch, launch_fails):
    from relax.distributed.ray import policy_engine_pool as policy

    router = module.ModelRouter()
    other_process, child = Mock(), Mock()
    monkeypatch.setattr(policy, "_LAUNCHED_ROUTER_PROCESSES", [other_process])
    monkeypatch.setattr(policy.multiprocessing, "Process", Mock(return_value=child))
    monkeypatch.setattr(policy.time, "sleep", lambda _: None)
    monkeypatch.setattr(policy, "find_available_port", lambda start: 8000)
    monkeypatch.setattr(policy, "get_host_info", lambda: ("local", "127.0.0.1"))
    child.is_alive.return_value = not launch_fails
    args = SimpleNamespace(
        sglang_router_ip=None,
        sglang_router_port=None,
        use_slime_router=True,
        router_dp_aware=True,
        slime_router_middleware_paths=["policy.middleware"],
    )
    if launch_fails:
        with pytest.raises(AssertionError):
            router.start(args)
    else:
        router.start(args)
        assert router.endpoint[1] == 8000
    (child_config,) = policy.multiprocessing.Process.call_args.kwargs["args"]
    assert not child_config.router_dp_aware and child_config.slime_router_middleware_paths == []
    assert args.router_dp_aware and args.slime_router_middleware_paths == ["policy.middleware"]
    assert router.processes == [child]
    assert policy._LAUNCHED_ROUTER_PROCESSES == [other_process]


def test_model_router_can_register_first_slime_worker(monkeypatch):
    from fastapi.testclient import TestClient

    from relax.engine.router.router import SlimeRouter

    slime = SlimeRouter(SimpleNamespace(slime_router_max_connections=4, slime_router_middleware_paths=[]))
    router = module.ModelRouter()
    router.endpoint = ("localhost", 8000)
    with TestClient(slime.app, raise_server_exceptions=False) as client:
        assert client.get("/health").status_code == 500

        def register(*args):
            return client.post("/add_worker", params={"url": "http://engine:30000"}).is_success

        monkeypatch.setattr(module.requests, "get", lambda url, **kwargs: client.get(url))
        monkeypatch.setattr(module.ray, "get", lambda ref, **kwargs: ref)
        router.activate([SimpleNamespace(configure_model_router=SimpleNamespace(remote=register))])
        assert router.ready
        assert client.get("/workers").json() == {"workers": [{"url": "http://engine:30000"}]}
