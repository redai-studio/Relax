# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.core import controller as inference_recovery
from relax.core.registry import process_role
from relax.utils.env import Envs, known_env_names, validate_env


def configure_recovery(config):
    controller = object.__new__(inference_recovery.Controller)
    controller.config = config
    return controller._create_inference_recovery_state()


def role_config(*, hybrid=False):
    return SimpleNamespace(
        debug_rollout_only=False,
        debug_train_only=False,
        loss_type="policy_loss",
        advantage_estimator="grpo",
        fully_async=False,
        true_on_policy_mode=True,
        hybrid=hybrid,
        use_fault_tolerance=True,
        train_backend="megatron",
        kl_coef=0,
        use_kl_loss=False,
        colocate=hybrid,
        offload_train=False,
        offload_rollout=False,
        enable_weights_backuper=True,
        rollout_function_path="relax.engine.rollout.sglang_rollout.generate_rollout",
    )


@pytest.fixture
def constructor(monkeypatch):
    constructor = Mock()
    monkeypatch.delenv("RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS", raising=False)
    monkeypatch.setattr(inference_recovery, "InferenceRecoveryState", constructor)
    return constructor


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "fully_async,hybrid,colocate",
    [(True, False, False), (True, True, True), (False, True, True), (False, False, True)],
)
def test_fault_tolerance_selects_component_recovery_without_changing_roles(
    constructor, enabled, fully_async, hybrid, colocate
):
    config = role_config(hybrid=hybrid)
    config.use_fault_tolerance = enabled
    config.fully_async, config.colocate = fully_async, colocate
    config.true_on_policy_mode = False
    if colocate and not hybrid:
        config.offload_train = config.offload_rollout = True
    roles_before = list(process_role(config))

    state = configure_recovery(config)

    assert config.use_fault_tolerance is enabled
    assert config.inference_recovery is enabled
    assert state is config.inference_recovery_state
    if enabled:
        assert state is constructor.options.return_value.remote.return_value
        constructor.options.return_value.remote.assert_called_once_with(fully_async and not hybrid, max_attempts=3)
    else:
        assert state is None
        constructor.options.assert_not_called()
    assert list(process_role(config)) == roles_before


def test_true_on_policy_recovery_does_not_add_forward_receiver(constructor):
    config = role_config()
    config.fully_async = True

    assert configure_recovery(config) is constructor.options.return_value.remote.return_value
    constructor.options.return_value.remote.assert_called_once_with(False, max_attempts=3)


@pytest.mark.parametrize("loss_type", ["sft", "dpo", "rm"])
def test_offline_training_does_not_create_inference_recovery_state(constructor, loss_type):
    config = role_config()
    config.loss_type = loss_type

    assert configure_recovery(config) is None
    assert config.inference_recovery is False
    assert config.inference_recovery_state is None
    assert config.use_fault_tolerance is True
    constructor.options.assert_not_called()


@pytest.mark.parametrize("budget,expected", [("", 3), (" ", 3), ("0", 0), ("1", 1), ("5", 5)])
def test_recovery_budget_env_is_passed_to_state(monkeypatch, constructor, budget, expected):
    monkeypatch.setenv("RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS", budget)
    config = role_config(hybrid=True)

    configure_recovery(config)

    constructor.options.return_value.remote.assert_called_once_with(False, max_attempts=expected)


@pytest.mark.parametrize("budget", ["-1", "1.5", "bad"])
def test_invalid_recovery_budget_fails_before_creating_state(monkeypatch, constructor, budget):
    monkeypatch.setenv("RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS", budget)
    config = role_config(hybrid=True)

    with pytest.raises(ValueError, match="RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS"):
        configure_recovery(config)

    constructor.options.assert_not_called()


@pytest.mark.parametrize(
    "name,default,raw,expected",
    [
        ("RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS", 3, "5", 5),
        ("RELAX_INFERENCE_RECOVERY_DRAIN_TIMEOUT", 180.0, "2.5", 2.5),
    ],
)
def test_recovery_env_registered_with_typed_defaults(monkeypatch, name, default, raw, expected):
    assert name in known_env_names()
    monkeypatch.delenv(name, raising=False)
    assert getattr(Envs, name) == default
    monkeypatch.setenv(name, " ")
    assert getattr(Envs, name) == default
    monkeypatch.setenv(name, raw)
    assert getattr(Envs, name) == expected


def test_startup_validation_rejects_invalid_budget_even_when_recovery_disabled():
    with pytest.raises(ValueError, match="RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS"):
        validate_env({"RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS": "bad"})


@pytest.mark.parametrize("enabled", [None, False])
def test_disabled_fault_tolerance_ignores_budget_env(monkeypatch, constructor, enabled):
    monkeypatch.setenv("RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS", "bad")
    config = role_config(hybrid=True)
    if enabled is None:
        del config.use_fault_tolerance
    else:
        config.use_fault_tolerance = enabled

    assert configure_recovery(config) is None
    constructor.options.assert_not_called()


@pytest.mark.parametrize("offload", ["offload_train", "offload_rollout"])
def test_hybrid_offload_keeps_engine_recovery(constructor, offload):
    config = role_config(hybrid=True)
    config.fully_async = True
    setattr(config, offload, True)

    assert configure_recovery(config) is None
    constructor.options.assert_not_called()
    assert config.use_fault_tolerance is True
    assert config.inference_recovery is False
    assert config.inference_recovery_state is None


@pytest.mark.parametrize(
    "conflict,value", [("train_backend", "fsdp"), ("rollout_external", True), ("use_routing_replay", True)]
)
def test_unsupported_component_recovery_keeps_engine_recovery(constructor, conflict, value):
    config = role_config()
    config.fully_async = True
    setattr(config, conflict, value)

    assert configure_recovery(config) is None
    assert config.use_fault_tolerance is True
    assert config.inference_recovery is False
    assert config.inference_recovery_state is None
    constructor.options.assert_not_called()


def test_colocate_without_offload_or_backup_keeps_engine_recovery(constructor):
    config = role_config()
    config.colocate, config.enable_weights_backuper = True, False

    assert configure_recovery(config) is None
    assert config.use_fault_tolerance is True
    assert config.inference_recovery is False
    constructor.options.assert_not_called()
