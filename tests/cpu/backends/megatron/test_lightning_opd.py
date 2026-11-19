# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest
import torch

from relax.engine.sft.runtime import validate_lightning_opd_args


def test_advantage_uses_this_updates_student_forward():
    from relax.backends.megatron.loss import set_lightning_opd_advantages

    data = {
        "teacher_log_probs": [torch.tensor([-1.5, -3.0])],
        "log_probs": [torch.tensor([-2.0, -2.0], requires_grad=True)],
        "rollout_log_probs": [torch.tensor([-5.0, -5.0])],
    }
    set_lightning_opd_advantages(data)
    torch.testing.assert_close(data["advantages"][0], torch.tensor([0.5, -1.0]))
    assert not data["advantages"][0].requires_grad
    data["log_probs"] = [torch.tensor([-1.75, -2.5])]
    set_lightning_opd_advantages(data)
    torch.testing.assert_close(data["advantages"][0], torch.tensor([0.25, -0.5]))


@pytest.mark.parametrize(
    "option",
    [
        "true_on_policy_mode",
        "use_rollout_logprobs",
        "use_opd",
        "calculate_per_token_loss",
        "use_routing_replay",
        "use_rollout_routing_replay",
    ],
)
def test_offline_objective_rejects_incompatible_online_or_reduction_options(option):
    args = Namespace(
        loss_type="lightning_opd",
        train_backend="megatron",
        task_type="causal_lm",
        advantage_estimator="grpo",
        rollout_temperature=1,
        n_samples_per_prompt=1,
        rollout_batch_size=8,
        global_batch_size=8,
        **{option: True},
    )
    with pytest.raises(ValueError, match=option):
        validate_lightning_opd_args(args)


def _lightning_opd_args(**overrides):
    values = dict(
        loss_type="lightning_opd",
        train_backend="megatron",
        task_type="causal_lm",
        advantage_estimator="grpo",
        rollout_temperature=1,
        n_samples_per_prompt=1,
        rollout_batch_size=8,
        global_batch_size=8,
    )
    values.update(overrides)
    return Namespace(**values)


def test_lightning_opd_accepts_global_batch_derived_later_from_rollout_batch():
    # relax_validate_args derives --global-batch-size from --num-steps-per-rollout after this validator runs.
    validate_lightning_opd_args(
        _lightning_opd_args(rollout_batch_size=256, global_batch_size=None, num_steps_per_rollout=1)
    )
    with pytest.raises(ValueError, match="one global batch"):
        validate_lightning_opd_args(_lightning_opd_args(rollout_batch_size=256, global_batch_size=128))


def test_lightning_opd_rejects_oversize_strategies_it_cannot_honour():
    validate_lightning_opd_args(_lightning_opd_args(sft_oversize_strategy="keep"))
    with pytest.raises(ValueError, match="sft-oversize-strategy"):
        validate_lightning_opd_args(_lightning_opd_args(sft_oversize_strategy="skip"))
