# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""First-step evaluation must not wait on an idle SFT consumer."""

from types import SimpleNamespace
from unittest.mock import Mock

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
