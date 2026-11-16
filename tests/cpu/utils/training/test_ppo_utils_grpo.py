# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.utils.training.ppo_utils import compute_approx_kl as _compiled_compute_approx_kl
from relax.utils.training.ppo_utils import get_grpo_returns


compute_approx_kl = torch.compiler.disable(_compiled_compute_approx_kl)


def test_get_grpo_returns_broadcasts_rewards_to_token_shapes():
    torch.manual_seed(0)
    rewards = torch.tensor([1.5, -0.25])
    kl = [torch.randn(3), torch.randn(2, 2, dtype=torch.float64)]

    returns = get_grpo_returns(rewards, kl)

    expected = [torch.full_like(kl[0], 1.5), torch.full_like(kl[1], -0.25)]
    assert len(returns) == len(expected)
    for actual, expected_return in zip(returns, expected):
        assert torch.allclose(actual, expected_return, atol=1e-6)


def test_compute_approx_kl_k1_returns_signed_log_ratio():
    torch.manual_seed(0)
    log_probs = torch.tensor([-0.2, -0.4, -1.0])
    log_probs_base = torch.tensor([-0.3, -0.1, -1.0])

    actual = compute_approx_kl(log_probs, log_probs_base, "k1")
    expected = log_probs.float() - log_probs_base.float()

    assert torch.allclose(actual, expected, atol=1e-6)


def test_compute_approx_kl_k2_matches_formula_and_is_non_negative():
    torch.manual_seed(0)
    log_probs = torch.tensor([-0.7, -0.2, -1.4])
    log_probs_base = torch.tensor([-0.1, -0.8, -1.0])
    log_ratio = log_probs.float() - log_probs_base.float()

    actual = compute_approx_kl(log_probs, log_probs_base, "k2")
    expected = log_ratio.square() / 2.0

    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.all(actual >= 0)


def test_compute_approx_kl_k3_matches_formula_and_is_non_negative():
    torch.manual_seed(0)
    log_probs = torch.tensor([-0.8, -0.2, -1.5])
    log_probs_base = torch.tensor([-0.3, -0.7, -1.0])
    log_ratio = log_probs.float() - log_probs_base.float()

    actual = compute_approx_kl(log_probs, log_probs_base, "k3")
    expected = torch.exp(-log_ratio) - 1.0 + log_ratio

    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.all(actual >= 0)


def test_compute_approx_kl_low_var_matches_formula_and_clamps_large_values():
    torch.manual_seed(0)
    log_probs = torch.tensor([-20.0, -0.5, 0.5])
    log_probs_base = torch.zeros_like(log_probs)
    log_ratio = log_probs.float() - log_probs_base.float()

    actual = compute_approx_kl(log_probs, log_probs_base, "low_var_kl")
    expected = torch.clamp(torch.exp(-log_ratio) - 1.0 + log_ratio, min=-10, max=10)

    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.all(actual >= 0)
    assert torch.all(actual <= 10)


def test_compute_approx_kl_rejects_unknown_estimator():
    torch.manual_seed(0)
    log_probs = torch.tensor([-0.2, -0.4])
    log_probs_base = torch.tensor([-0.3, -0.1])

    with pytest.raises(ValueError, match="Unknown kl_loss_type: unsupported"):
        compute_approx_kl(log_probs, log_probs_base, "unsupported")
