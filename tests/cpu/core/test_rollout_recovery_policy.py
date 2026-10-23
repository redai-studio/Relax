# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.core import inference_recovery
from relax.core.inference_recovery import configure_recovery
from relax.core.registry import process_role
from relax.utils.env import Envs, known_env_names, validate_env


def role_config(*, hybrid=False):
    return SimpleNamespace(
        debug_rollout_only=False,
        debug_train_only=False,
        loss_type="policy_loss",
        advantage_estimator="grpo",
        fully_async=False,
        true_on_policy_mode=True,
        hybrid=hybrid,
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


@pytest.mark.parametrize("flag", [None, "", " ", "0", "1", "true", "false", "TRUE", " FALSE ", "on", "off"])
@pytest.mark.parametrize(
    "fully_async,hybrid,colocate",
    [(True, False, False), (True, True, True), (False, True, True), (False, False, True)],
)
def test_recovery_mode_defaults_and_overrides_keep_original_roles(
    monkeypatch, constructor, flag, fully_async, hybrid, colocate
):
    if flag is None:
        monkeypatch.delenv("RELAX_ENABLE_INFERENCE_RECOVERY", raising=False)
    else:
        monkeypatch.setenv("RELAX_ENABLE_INFERENCE_RECOVERY", flag)
    config = role_config(hybrid=hybrid)
    config.fully_async, config.colocate = fully_async, colocate
    config.true_on_policy_mode = False
    if colocate and not hybrid:
        config.offload_train = config.offload_rollout = True
    roles_before = list(process_role(config))
    enabled = flag.strip().lower() in ("1", "true", "on") if flag and flag.strip() else fully_async or hybrid

    state = configure_recovery(config)

    assert config.inference_recovery is enabled
    assert state is config.inference_recovery_state
    if enabled:
        assert state is constructor.options.return_value.remote.return_value
        constructor.options.return_value.remote.assert_called_once_with(fully_async and not hybrid, max_attempts=3)
    else:
        assert state is None
        constructor.options.assert_not_called()
    assert list(process_role(config)) == roles_before


def test_default_true_on_policy_recovery_does_not_add_forward_receiver(monkeypatch, constructor):
    monkeypatch.delenv("RELAX_ENABLE_INFERENCE_RECOVERY", raising=False)
    config = role_config()
    config.fully_async = True

    assert configure_recovery(config) is constructor.options.return_value.remote.return_value
    constructor.options.return_value.remote.assert_called_once_with(False, max_attempts=3)


@pytest.mark.parametrize("budget,expected", [("", 3), (" ", 3), ("0", 0), ("1", 1), ("5", 5)])
def test_recovery_budget_env_is_passed_to_state(monkeypatch, constructor, budget, expected):
    monkeypatch.setenv("RELAX_ENABLE_INFERENCE_RECOVERY", "1")
    monkeypatch.setenv("RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS", budget)
    config = role_config(hybrid=True)

    configure_recovery(config)

    constructor.options.return_value.remote.assert_called_once_with(False, max_attempts=expected)


@pytest.mark.parametrize("budget", ["-1", "1.5", "bad"])
def test_invalid_recovery_budget_fails_before_creating_state(monkeypatch, constructor, budget):
    monkeypatch.setenv("RELAX_ENABLE_INFERENCE_RECOVERY", "1")
    monkeypatch.setenv("RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS", budget)
    config = role_config(hybrid=True)

    with pytest.raises(ValueError, match="RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS"):
        configure_recovery(config)

    constructor.options.assert_not_called()


@pytest.mark.parametrize("flag", ["tru", "2"])
def test_invalid_recovery_switch_fails_before_creating_state(monkeypatch, constructor, flag):
    monkeypatch.setenv("RELAX_ENABLE_INFERENCE_RECOVERY", flag)
    with pytest.raises(ValueError, match="RELAX_ENABLE_INFERENCE_RECOVERY"):
        configure_recovery(role_config(hybrid=True))
    constructor.options.assert_not_called()


@pytest.mark.parametrize(
    "name,default,raw,expected",
    [
        ("RELAX_ENABLE_INFERENCE_RECOVERY", None, "true", True),
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
        validate_env({"RELAX_ENABLE_INFERENCE_RECOVERY": "false", "RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS": "bad"})


@pytest.mark.parametrize("flag", ["0", "false"])
def test_disabled_recovery_ignores_budget_env(monkeypatch, constructor, flag):
    monkeypatch.setenv("RELAX_ENABLE_INFERENCE_RECOVERY", flag)
    monkeypatch.setenv("RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS", "bad")

    assert configure_recovery(role_config(hybrid=True)) is None
    constructor.options.assert_not_called()


@pytest.mark.parametrize("offload", ["offload_train", "offload_rollout"])
@pytest.mark.parametrize("flag", ["1", "true"])
def test_hybrid_recovery_rejects_offload_before_creating_state(monkeypatch, constructor, offload, flag):
    monkeypatch.setenv("RELAX_ENABLE_INFERENCE_RECOVERY", flag)
    config = role_config(hybrid=True)
    config.fully_async = True
    setattr(config, offload, True)
    with pytest.raises(ValueError, match=f"{offload}=True.*requires disabled when hybrid=True"):
        configure_recovery(config)
    constructor.options.assert_not_called()
    assert config.inference_recovery is False
    assert config.inference_recovery_state is None


@pytest.mark.parametrize(
    "conflict,value", [("train_backend", "fsdp"), ("rollout_external", True), ("use_routing_replay", True)]
)
def test_default_unsupported_config_keeps_original_path(monkeypatch, constructor, conflict, value):
    monkeypatch.delenv("RELAX_ENABLE_INFERENCE_RECOVERY", raising=False)
    config = role_config()
    config.fully_async = True
    setattr(config, conflict, value)

    assert configure_recovery(config) is None
    assert config.inference_recovery is False
    assert config.inference_recovery_state is None
    constructor.options.assert_not_called()


@pytest.mark.parametrize("flag", ["1", "true", " TRUE "])
def test_explicit_recovery_reports_all_actual_conflicts_before_creating_state(monkeypatch, constructor, flag):
    monkeypatch.setenv("RELAX_ENABLE_INFERENCE_RECOVERY", flag)
    config = role_config()
    config.fully_async = True
    config.train_backend, config.kl_coef, config.use_opd = "fsdp", 0.1, True
    config.use_routing_replay = True

    with pytest.raises(ValueError) as error:
        configure_recovery(config)

    message = str(error.value)
    for expected in (
        "train_backend='fsdp' (requires 'megatron')",
        "kl_coef=0.1 (requires 0)",
        "use_opd=True (requires disabled or empty)",
        "use_routing_replay=True (requires disabled when fully_async=True)",
    ):
        assert expected in message
    constructor.options.assert_not_called()
    assert config.inference_recovery is False
    assert config.inference_recovery_state is None


def test_explicit_colocate_recovery_reports_missing_offload_and_backup(monkeypatch, constructor):
    monkeypatch.setenv("RELAX_ENABLE_INFERENCE_RECOVERY", "1")
    config = role_config()
    config.colocate, config.enable_weights_backuper = True, False

    with pytest.raises(ValueError) as error:
        configure_recovery(config)

    for name in ("offload_train", "offload_rollout", "enable_weights_backuper"):
        assert f"{name}=False (requires enabled when colocate=True)" in str(error.value)
    constructor.options.assert_not_called()
