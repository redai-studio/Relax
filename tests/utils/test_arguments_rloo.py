# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for RLOO validation through the production argument path."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from tests.utils.test_arguments_opd_teacher_colocate import (
    _opd_args,
)
from tests.utils.test_arguments_opd_teacher_colocate import (
    arguments_module as _arguments_module_fixture,
)


arguments_module = _arguments_module_fixture


def _rloo_args(**overrides) -> SimpleNamespace:
    """Return a complete ``slime_validate_args`` input with valid RLOO
    defaults."""
    args = _opd_args()
    defaults = {
        "advantage_estimator": "rloo",
        "n_samples_per_prompt": 8,
        "rollout_batch_size": 16,
        "global_batch_size": 128,
        "num_steps_per_rollout": None,
        "fully_async": False,
        "hybrid": False,
        "rewards_normalization": True,
        "normalize_advantages": False,
        "partial_rollout": False,
        "use_dynamic_global_batch_size": False,
        "calculate_per_token_loss": True,
        "kl_coef": 0.0,
        "max_staleness": 0,
        "grpo_std_normalization": False,
    }
    defaults.update(overrides)
    for name, value in defaults.items():
        setattr(args, name, value)
    return args


def _validate(arguments_module, **overrides) -> SimpleNamespace:
    args = _rloo_args(**overrides)
    arguments_module.slime_validate_args(args)
    return args


def test_rloo_valid_config_passes(arguments_module):
    args = _validate(arguments_module)
    assert args.rollout_batch_size == 16
    assert args.global_batch_size == 128


def test_rloo_derives_rollout_batch_size_before_validation(arguments_module):
    args = _validate(arguments_module, rollout_batch_size=None, global_batch_size=128)
    assert args.rollout_batch_size == 16


def test_rloo_derives_batch_before_rejecting_fully_async(arguments_module):
    with pytest.raises(ValueError, match="synchronous"):
        _validate(arguments_module, rollout_batch_size=None, global_batch_size=128, fully_async=True)


def test_batch_derivation_requires_exact_divisibility(arguments_module):
    with pytest.raises(ValueError, match="must be divisible"):
        _validate(arguments_module, rollout_batch_size=None, global_batch_size=127)


def test_fully_async_non_rloo_can_derive_rollout_batch_size(arguments_module):
    args = _validate(
        arguments_module,
        advantage_estimator="grpo",
        rollout_batch_size=None,
        global_batch_size=127,
        fully_async=True,
        resource={"actor": [0], "rollout": [0], "actor_fwd": [0], "teacher": [0]},
    )
    assert args.rollout_batch_size == 15
    assert args.true_on_policy_mode is False


def test_rloo_requires_n_samples_ge_2(arguments_module):
    with pytest.raises(ValueError, match="n-samples-per-prompt >= 2"):
        _validate(arguments_module, n_samples_per_prompt=1, rollout_batch_size=16, global_batch_size=16)


@pytest.mark.parametrize("mode", ["fully_async", "hybrid"])
def test_rloo_rejects_async_modes(arguments_module, mode):
    with pytest.raises(ValueError, match="synchronous"):
        _validate(arguments_module, **{mode: True})


def test_rloo_requires_rewards_normalization(arguments_module):
    with pytest.raises(ValueError, match="rewards normalization"):
        _validate(arguments_module, rewards_normalization=False)


def test_rloo_rejects_normalize_advantages(arguments_module):
    with pytest.raises(ValueError, match="normalize-advantages"):
        _validate(arguments_module, normalize_advantages=True)


def test_rloo_requires_global_token_loss_reduction(arguments_module):
    with pytest.raises(ValueError, match="calculate-per-token-loss"):
        _validate(arguments_module, calculate_per_token_loss=False)


def test_rloo_rejects_reward_side_kl(arguments_module):
    with pytest.raises(ValueError, match="does not support nonzero --kl-coef"):
        _validate(arguments_module, kl_coef=0.01)


def test_rloo_rejects_stale_rollouts(arguments_module):
    with pytest.raises(ValueError, match="max-staleness 0"):
        _validate(arguments_module, max_staleness=1)


def test_rloo_allows_direct_kl_loss(arguments_module):
    args = _validate(arguments_module, use_kl_loss=True, kl_loss_coef=0.01, ref_load="/")
    assert args.use_kl_loss is True
    assert args.kl_loss_coef == 0.01


def test_rloo_requires_matching_global_batch_size(arguments_module):
    with pytest.raises(ValueError, match="one optimizer update per rollout"):
        _validate(arguments_module, global_batch_size=64)


def test_rloo_rejects_multiple_steps_per_rollout(arguments_module):
    with pytest.raises(ValueError, match="num-steps-per-rollout 1"):
        _validate(arguments_module, num_steps_per_rollout=2, global_batch_size=64)


@pytest.mark.parametrize("mode", ["partial_rollout", "use_dynamic_global_batch_size"])
def test_rloo_rejects_dynamic_batch_modes(arguments_module, mode):
    with pytest.raises(ValueError, match="effective batch size to drift"):
        _validate(arguments_module, **{mode: True})


def test_non_rloo_path_does_not_apply_rloo_guards(arguments_module):
    args = _validate(
        arguments_module,
        advantage_estimator="grpo",
        n_samples_per_prompt=1,
        rollout_batch_size=16,
        global_batch_size=16,
        rewards_normalization=False,
        normalize_advantages=True,
        partial_rollout=True,
        use_dynamic_global_batch_size=True,
        calculate_per_token_loss=False,
        kl_coef=0.01,
        max_staleness=1,
        ref_load="/",
    )
    assert args.advantage_estimator == "grpo"


def test_advantage_estimator_choices_include_rloo(arguments_module, monkeypatch):
    monkeypatch.setattr(
        arguments_module,
        "RouterArgs",
        SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser),
    )
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    action = next(action for action in parser._actions if action.dest == "advantage_estimator")
    assert "rloo" in action.choices
