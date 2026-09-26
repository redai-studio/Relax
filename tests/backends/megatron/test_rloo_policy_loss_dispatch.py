# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Smoke coverage for RLOO dispatch through Megatron's policy loss."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _load_loss_module():
    """Import the production loss module without requiring Megatron in CPU
    CI."""
    try:
        import megatron.core  # noqa: F401
    except ModuleNotFoundError:
        megatron = ModuleType("megatron")
        core = ModuleType("megatron.core")
        mpu = ModuleType("megatron.core.mpu")
        core.mpu = mpu
        sys.modules.update(
            {
                "megatron": megatron,
                "megatron.core": core,
                "megatron.core.mpu": mpu,
            }
        )
        try:
            return importlib.import_module("relax.backends.megatron.loss")
        finally:
            sys.modules.pop("megatron.core.mpu", None)
            sys.modules.pop("megatron.core", None)
            sys.modules.pop("megatron", None)
    return importlib.import_module("relax.backends.megatron.loss")


loss_module = _load_loss_module()


def test_policy_loss_function_dispatches_rloo_objective(monkeypatch):

    log_probs = torch.tensor([-0.7, -0.4, -0.2, -0.1], dtype=torch.float64, requires_grad=True)
    entropy = torch.zeros_like(log_probs)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=torch.float64)

    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (None, {"log_probs": [log_probs], "entropy": [entropy]}),
    )
    monkeypatch.setattr(loss_module, "resolve_opd_gather_topk_token_ids", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        loss_module,
        "compute_policy_opd_loss",
        lambda **_kwargs: (None, {}),
    )

    args = SimpleNamespace(
        advantage_estimator="rloo",
        true_on_policy_mode=False,
        use_rollout_logprobs=False,
        use_opsm=False,
        get_mismatch_metrics=False,
        use_tis=False,
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.0,
        use_kl_loss=False,
    )
    batch = {
        "advantages": advantages,
        "log_probs": [torch.zeros_like(log_probs)],
        "response_lengths": [log_probs.numel()],
        "total_lengths": [log_probs.numel() + 1],
        "unconcat_tokens": [torch.arange(log_probs.numel() + 1)],
        "loss_masks": [torch.ones_like(log_probs)],
    }

    loss, metrics = loss_module.policy_loss_function(
        args,
        batch,
        logits=torch.empty(1, 1, 1),
        sum_of_sample_mean=lambda values: values.sum(),
    )

    expected = -(advantages * log_probs).sum()
    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["pg_loss"], expected.detach())
    assert metrics["pg_clipfrac"].item() == 0.0

    loss.backward()
    assert torch.allclose(log_probs.grad, -advantages)


def test_rloo_unequal_lengths_use_global_token_scalar_and_gradient_oracle(monkeypatch):
    """Exercise the production reducer and returned Megatron token normalizer
    with unequal non-empty responses."""
    response_lengths = [2, 4]
    masks = [
        torch.tensor([1.0, 1.0], dtype=torch.float64),
        torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64),
    ]
    num_tokens = sum(response_lengths)
    log_probs = torch.tensor(
        [-0.2, -0.4, -0.1, -0.3, -0.5, -0.7][:num_tokens],
        dtype=torch.float64,
        requires_grad=True,
    )
    advantages = torch.tensor(
        [2.0] * response_lengths[0] + [-1.0] * response_lengths[1],
        dtype=torch.float64,
    )

    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (
            None,
            {"log_probs": [log_probs], "entropy": [torch.zeros_like(log_probs)]},
        ),
    )
    monkeypatch.setattr(loss_module, "resolve_opd_gather_topk_token_ids", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loss_module, "compute_policy_opd_loss", lambda **_kwargs: (None, {}))

    args = SimpleNamespace(
        loss_type="policy_loss",
        advantage_estimator="rloo",
        calculate_per_token_loss=True,
        qkv_format="thd",
        recompute_loss_function=False,
        allgather_cp=False,
        global_batch_size=2,
        true_on_policy_mode=False,
        use_rollout_logprobs=False,
        use_opsm=False,
        get_mismatch_metrics=False,
        use_tis=False,
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.0,
        use_kl_loss=False,
    )
    batch = {
        "advantages": advantages,
        "log_probs": [torch.zeros_like(log_probs)],
        "response_lengths": response_lengths,
        "total_lengths": [length + 1 for length in response_lengths],
        "unconcat_tokens": [torch.arange(length + 1) for length in response_lengths],
        "loss_masks": masks,
        "dynamic_cp_size": 1,
        "dynamic_cp_rank": 0,
    }

    token_sum_loss, normalizer, _ = loss_module.loss_function(
        args,
        batch,
        num_microbatches=1,
        logits=torch.empty(1, 1, 1, dtype=torch.float64),
    )

    flat_mask = torch.cat(masks)
    expected_token_sum = -(advantages * log_probs * flat_mask).sum()
    expected_num_tokens = flat_mask.sum()
    expected_final_loss = expected_token_sum / expected_num_tokens
    final_loss = token_sum_loss / normalizer

    assert torch.allclose(token_sum_loss, expected_token_sum)
    assert normalizer.dtype == torch.int
    assert torch.allclose(normalizer.to(expected_num_tokens.dtype), expected_num_tokens)
    assert torch.allclose(final_loss, expected_final_loss)

    first_length = response_lengths[0]
    per_sample_token_mean = (
        -(
            (advantages[:first_length] * log_probs[:first_length] * masks[0]).sum()
            / torch.clamp_min(masks[0].sum(), 1)
            + (advantages[first_length:] * log_probs[first_length:] * masks[1]).sum()
            / torch.clamp_min(masks[1].sum(), 1)
        )
        / 2
    )
    assert not torch.allclose(final_loss, per_sample_token_mean)

    final_loss.backward()
    expected_gradient = -(advantages * flat_mask) / expected_num_tokens
    assert torch.allclose(log_probs.grad, expected_gradient)
