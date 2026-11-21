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
        "debug_rollout_only",
        "debug_train_only",
        "load_debug_rollout_data",
        "true_on_policy_mode",
        "use_rollout_logprobs",
        "use_opd",
        "calculate_per_token_loss",
        "use_routing_replay",
        "use_rollout_routing_replay",
    ],
)
def test_offline_objective_rejects_incompatible_options(option):
    value = "rollout.pt" if option == "load_debug_rollout_data" else True
    with pytest.raises(ValueError, match=option):
        validate_lightning_opd_args(_lightning_opd_args(**{option: value}))


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


def test_lightning_opd_loss_averages_clipped_masked_responses(monkeypatch):
    from relax.backends.megatron import loss as loss_module

    args = _lightning_opd_args(
        global_batch_size=2,
        qkv_format="thd",
        calculate_per_token_loss=False,
        recompute_loss_function=False,
        allgather_cp=False,
        use_rollout_logprobs=False,
        use_opsm=False,
        use_tis=False,
        get_mismatch_metrics=False,
        use_kl_loss=False,
        entropy_coef=0.0,
        eps_clip=0.2,
        eps_clip_high=0.2,
        opd_token_selection="student_sampled",
    )
    batch = {
        "response_lengths": [1, 3],
        "total_lengths": [2, 4],
        "unconcat_tokens": [torch.arange(2), torch.arange(4)],
        "loss_masks": [torch.tensor([1]), torch.tensor([1, 0, 1])],
        "log_probs": [torch.full((1,), -2.0), torch.full((3,), -2.0)],
        "teacher_log_probs": [torch.tensor([-1.0]), torch.tensor([-3.0, -0.5, -1.5])],
        "dynamic_cp_size": 1,
        "dynamic_cp_rank": 0,
    }
    loss_module.set_lightning_opd_advantages(batch)
    # The GPU scorer boundary supplies current log-probs with known current/old probability ratios.
    current = (torch.tensor([1.5, 0.5, 1.0, 1.0]).log() - 2.0).requires_grad_()
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *a, **kw: (None, {"log_probs": current.split([1, 3]), "entropy": [torch.zeros_like(current)]}),
    )
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_world_size", lambda **kw: 1)
    loss, _, _ = loss_module.loss_function(args, batch, num_microbatches=1, logits=torch.empty(1, 1, 1))

    # Response losses: -1.2 (upper clip) and (0.8 - 0.5) / 2 (lower clip, middle token masked).
    torch.testing.assert_close(loss, torch.tensor(-0.525))
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([0.0, 0.0, 0.0, -0.125]))
