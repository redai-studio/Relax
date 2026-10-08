# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The Controller gives the managed OPD teacher a ``/teacher`` gateway:

deployed right after the teacher starts, never without one, and removed before
the teacher's engines are shut down.
"""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import ray
import transfer_queue
from ray import serve

from relax.utils import utils as relax_utils
from tests.core.controller_test_utils import load_controller_with_stubbed_dependencies


if not hasattr(transfer_queue, "StreamingTokenBudgetSampler"):
    pytest.skip(
        "controller tests require a TransferQueue build with StreamingTokenBudgetSampler",
        allow_module_level=True,
    )

controller = load_controller_with_stubbed_dependencies("_test_controller_teacher_gateway_controller")

SERVE_URL = "http://192.0.2.10:8000"


class _StartupWentOn(Exception):
    """Raised by the first step after the teacher block, to stop startup
    there."""


class _Manager:
    def __init__(self, events, name="teacher"):
        self.shutdown = SimpleNamespace(remote=lambda: events.append(f"shutdown {name}"))


def _stub_startup(monkeypatch, events, teacher_manager):
    def fake_start_teacher(config, *, runtime_env=None):
        events.append("start teacher")
        return None, teacher_manager

    def fake_serve_run(application, *, name, route_prefix):
        events.append(f"deploy {name} at {route_prefix}")

    def stop_startup(config):
        raise _StartupWentOn

    monkeypatch.setattr(controller, "validate_ppo_config", lambda config: None)
    monkeypatch.setattr(controller, "maybe_start_managed_opd_teacher", fake_start_teacher)
    monkeypatch.setattr(controller, "resolve_sft_algo_key", stop_startup)
    monkeypatch.setattr(serve, "run", fake_serve_run)
    monkeypatch.setattr(relax_utils, "get_serve_url", lambda route_prefix="": f"{SERVE_URL}{route_prefix}")


def _register_all_serve(config):
    owner = SimpleNamespace(config=config, runtime_env=None, _teacher_manager=None)
    with pytest.raises(_StartupWentOn):
        controller.Controller.register_all_serve(owner)
    return owner


def test_controller_deploys_teacher_gateway_after_teacher_start(monkeypatch):
    events = []
    manager = _Manager(events)
    _stub_startup(monkeypatch, events, manager)
    config = Namespace(colocate=False, resource={"actor": [1, 4], "rollout": [1, 4]})

    owner = _register_all_serve(config)

    assert events == ["start teacher", "deploy teacher at /teacher"]
    assert owner._teacher_manager is manager
    # Services are created after this point and pickle the config, so they all
    # learn where the teacher's topology can be discovered.
    assert config.opd_teacher_discovery_url == f"{SERVE_URL}/teacher/engines?schema_version=2"


def test_controller_deploys_one_teacher_gateway_for_all_mopd_teachers(monkeypatch):
    events = []
    _stub_startup(monkeypatch, events, [_Manager(events, "math"), _Manager(events, "code")])
    config = Namespace(
        colocate=False,
        resource={"actor": [1, 4], "rollout": [1, 4]},
        opd_teacher_routes='{"math": "/ckpt/math", "code": "/ckpt/code"}',
    )

    _register_all_serve(config)

    assert events == ["start teacher", "deploy teacher at /teacher"]


def test_controller_skips_teacher_gateway_without_managed_teacher(monkeypatch):
    events = []
    _stub_startup(monkeypatch, events, None)
    config = Namespace(colocate=False, resource={"actor": [1, 4], "rollout": [1, 4]})

    _register_all_serve(config)

    assert events == ["start teacher"]
    assert not hasattr(config, "opd_teacher_discovery_url")


def _shutdown(monkeypatch, events, teacher_manager):
    monkeypatch.setattr(serve, "delete", lambda name: events.append(f"delete {name}"))
    monkeypatch.setattr(ray, "get", lambda ref, timeout=None: ref)
    owner = SimpleNamespace(
        serve_dict={},
        _teacher_manager=teacher_manager,
        stop_health_check=lambda: None,
        _shutdown_agentic_rollout_services=lambda: None,
        _cleanup_s3_model_weights_after_init=lambda force=False: None,
    )
    controller.Controller.shutdown(owner)


def test_controller_removes_teacher_gateway_on_shutdown(monkeypatch):
    events = []

    _shutdown(monkeypatch, events, _Manager(events))

    # Gateway first: it must never route to engines that are being shut down.
    assert events == ["delete teacher", "shutdown teacher"]


def test_controller_shutdown_without_teacher_leaves_serve_alone(monkeypatch):
    events = []

    _shutdown(monkeypatch, events, None)

    assert events == []
