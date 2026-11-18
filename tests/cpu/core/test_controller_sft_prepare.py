# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""First-step evaluation must not wait on an idle SFT consumer."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.cpu.core.controller_test_utils import load_controller_with_stubbed_dependencies


@pytest.fixture
def controller_module():
    return load_controller_with_stubbed_dependencies("_test_controller_sft_prepare")


def test_first_step_eval_keeps_normal_startup_to_avoid_waiting_for_an_idle_consumer(controller_module, monkeypatch):
    monkeypatch.setenv("RELAX_SFT_EARLY_PREPARE", "1")
    controller = controller_module.Controller.__new__(controller_module.Controller)
    controller.config = SimpleNamespace(
        loss_type="sft",
        start_rollout_id=0,
        load="hf",
        hf_checkpoint="hf",
        eval_prompt_data=["eval"],
        eval_interval=1,
    )
    service = SimpleNamespace(handle=Mock())
    controller._maybe_prepare_sft_data("sft", service)
    service.handle.prepare.remote.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_rollout_id": 4},
        {"load": "checkpoint"},
        {"sft_predict_interval": 1},
        {"generation_task": "i2i"},
    ],
)
def test_sft_early_prepare_preserves_resume_and_first_step_consumers(controller_module, monkeypatch, overrides):
    monkeypatch.setenv("RELAX_SFT_EARLY_PREPARE", "1")
    controller = controller_module.Controller.__new__(controller_module.Controller)
    controller.config = SimpleNamespace(
        **({"loss_type": "sft", "start_rollout_id": 0, "load": "hf", "hf_checkpoint": "hf"} | overrides)
    )
    service = SimpleNamespace(handle=Mock())

    controller._maybe_prepare_sft_data("sft", service)

    service.handle.prepare.remote.assert_not_called()


@pytest.mark.parametrize("enabled,role,expected", [(True, "sft", True), (False, "sft", False), (True, "actor", False)])
def test_sft_early_prepare_runs_once_only_for_enabled_fresh_producer(
    controller_module, monkeypatch, enabled, role, expected
):
    monkeypatch.setenv("RELAX_SFT_EARLY_PREPARE", str(int(enabled)))
    controller = controller_module.Controller.__new__(controller_module.Controller)
    controller.config = SimpleNamespace(loss_type="sft", start_rollout_id=0, load="hf", hf_checkpoint="hf")
    service = SimpleNamespace(handle=Mock())

    controller._maybe_prepare_sft_data(role, service)
    controller._maybe_prepare_sft_data(role, service)

    if expected:
        service.handle.prepare.remote.assert_called_once_with(prefetch_first_batch=True)
        assert service._sft_prepare_response is service.handle.prepare.remote.return_value
    else:
        service.handle.prepare.remote.assert_not_called()


@pytest.mark.parametrize("configured,expected", [("true", True), ("0", False)])
def test_sft_prepare_uses_runtime_environment_before_driver(controller_module, monkeypatch, configured, expected):
    monkeypatch.setenv("RELAX_SFT_EARLY_PREPARE", "0" if expected else "1")
    controller = controller_module.Controller.__new__(controller_module.Controller)
    controller.config = SimpleNamespace(loss_type="sft", start_rollout_id=0, load="hf", hf_checkpoint="hf")
    controller.runtime_env = {"env_vars": {"RELAX_SFT_EARLY_PREPARE": configured}}
    service = SimpleNamespace(handle=Mock())

    controller._maybe_prepare_sft_data("sft", service)

    assert service.handle.prepare.remote.called is expected


@pytest.mark.parametrize("failure_stage", ["services", "health"])
def test_constructor_failure_stops_prepared_producer_before_closing_queue(
    controller_module, monkeypatch, failure_stage
):
    events = []
    error = RuntimeError("later startup failed")

    async def stop():
        assert "queue-closed" not in events
        events.extend(["pending-put-finished", "producer-stopped"])

    service = SimpleNamespace(
        _sft_prepare_response=object(),
        handle=SimpleNamespace(stop=SimpleNamespace(remote=AsyncMock(side_effect=stop))),
    )

    def register(controller):
        controller.serve_dict["sft"] = service
        if failure_stage == "services":
            raise error

    monkeypatch.setattr(controller_module, "require_control_plane_resource", lambda *_args: None)
    monkeypatch.setattr(controller_module, "resolve_sft_num_rollout", lambda _config: None)
    monkeypatch.setattr(
        controller_module, "HealthManager", lambda **_kwargs: SimpleNamespace(start=Mock(side_effect=error))
    )
    monkeypatch.setattr(controller_module, "create_dcs_deployment", lambda **_kwargs: (None, None))
    monkeypatch.setattr(controller_module.Controller, "_remove_stale_model_caches_before_startup", lambda _self: None)
    monkeypatch.setattr(controller_module.Controller, "_initialize_data_system", lambda _self: None)
    monkeypatch.setattr(controller_module.Controller, "register_all_serve", register)
    monkeypatch.setattr(controller_module.Controller, "_report_error_to_metrics_service", lambda *_args: None)
    monkeypatch.setattr(
        controller_module.Controller, "_close_data_system", lambda _self: events.append("queue-closed")
    )
    config = SimpleNamespace(use_health_check=failure_stage == "health", use_agentic_rollout=False)

    with pytest.raises(RuntimeError) as actual:
        controller_module.Controller(config)

    assert actual.value is error
    assert events == ["pending-put-finished", "producer-stopped", "queue-closed"]
