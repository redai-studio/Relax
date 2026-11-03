# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""FlowGRPO math: transition/log-prob parity, loss, advantage normalization."""

from __future__ import annotations

import pytest
import torch

from relax.models import flow_grpo as fg


def _sample_then_replay(eta=0.7, seed=0, sde_type="sde"):
    """Draw x_next from the transition Gaussian, then score it via replay.

    The engine owns sampling, so the module only exposes the moments; this
    mirrors what the engine does (mean + std * noise) and checks the replay
    path scores exactly that Gaussian.
    """
    torch.manual_seed(seed)
    b, c, h, w = 4, 4, 8, 8
    sigma = torch.tensor(0.7).view(1, 1, 1, 1)
    sigma_next = torch.tensor(0.5).view(1, 1, 1, 1)
    x_t = torch.randn(b, c, h, w)
    noise_pred = torch.randn(b, c, h, w)

    mean, std = fg.flow_sde_transition_moments(noise_pred, x_t, sigma, sigma_next, eta=eta, sde_type=sde_type)
    g = torch.Generator().manual_seed(123)
    x_next = mean + std * torch.randn(noise_pred.shape, generator=g)

    logp_sample = fg.flow_sde_log_prob(x_next, mean, std).mean(dim=(1, 2, 3))
    logp_replay = fg.replay_transition_logp(
        noise_pred, x_t, x_next, sigma.flatten(), sigma_next.flatten(), eta=eta, sde_type=sde_type
    )
    return logp_sample, logp_replay


def test_flow_grpo_rollout_replay_logp_parity():
    logp_sample, logp_replay = _sample_then_replay()
    assert torch.allclose(logp_sample, logp_replay, atol=1e-5)


def test_dance_grpo_rollout_replay_logp_parity():
    logp_sample, logp_replay = _sample_then_replay(seed=3, sde_type="dance")
    assert torch.allclose(logp_sample, logp_replay, atol=1e-5)

    sigma = torch.tensor(0.7).view(1, 1, 1, 1)
    sigma_next = torch.tensor(0.5).view(1, 1, 1, 1)
    assert not torch.allclose(
        fg.flow_sde_transition_std(sigma, sigma_next, eta=0.7),
        fg.flow_sde_transition_std(sigma, sigma_next, eta=0.7, sde_type="dance"),
    )


def test_flow_sde_transition_std_matches_moments():
    """The standalone std (actor debug log) is the one the moments use."""
    sigma, sigma_next = torch.tensor(0.7), torch.tensor(0.5)
    x_t = torch.randn(2, 3)
    _, std = fg.flow_sde_transition_moments(torch.randn(2, 3), x_t, sigma, sigma_next, eta=0.7)
    assert torch.allclose(std, fg.flow_sde_transition_std(sigma, sigma_next, eta=0.7))


def test_flow_sde_unsupported_sde_type_raises():
    sigma, sigma_next = torch.tensor(0.7), torch.tensor(0.5)
    with pytest.raises(ValueError):
        fg.flow_sde_transition_moments(torch.zeros(1), torch.zeros(1), sigma, sigma_next, sde_type="ddpm")


def test_flow_grpo_on_policy_ratio_is_one():
    _, logp = _sample_then_replay()
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    loss, metrics = fg.grpo_clip_loss(logp, logp, adv, clip_eps=0.2)
    assert abs(float(metrics["ratio_mean"]) - 1.0) < 1e-6
    assert float(metrics["approx_kl"]) < 1e-9


def test_flow_grpo_clip_loss_clips_high_ratio():
    old = torch.zeros(2)
    new = torch.tensor([2.0, 2.0])  # ratio = e^2 >> 1+eps
    adv = torch.tensor([1.0, 1.0])  # positive advantage → clipped branch active
    loss, metrics = fg.grpo_clip_loss(new, old, adv, clip_eps=0.2)
    # clipped ratio ceiling is 1.2 → loss per sample = -adv * 1.2 = -1.2
    assert torch.allclose(loss, torch.full((2,), -1.2), atol=1e-5)
    assert float(metrics["clip_fraction"]) == 1.0


def test_normalize_grouped_centers_per_group():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    adv = fg.normalize_grouped(rewards, [[0, 1, 2], [3, 4, 5]])
    # each group symmetric around its mean → middle element is 0
    assert abs(float(adv[1])) < 1e-4 and abs(float(adv[4])) < 1e-4
    assert float(adv[0]) < 0 < float(adv[2])


def test_normalize_grouped_divides_by_each_group_own_std():
    """Each group is scaled by its OWN spread, not a shared batch-wide one."""
    rewards = torch.tensor([0.0, 10.0, 0.0, 1.0])
    adv = fg.normalize_grouped(rewards, [[0, 1], [2, 3]])
    # Both groups are 2-element and symmetric, so per-group normalization maps
    # each to the same +/- pair despite their very different raw spreads.
    assert torch.allclose(adv[:2], adv[2:], atol=1e-4)


def test_normalize_grouped_batch_std_mode_uses_one_global_divisor():
    rewards = torch.tensor([0.0, 10.0, 0.0, 1.0])
    adv = fg.normalize_grouped(rewards, [[0, 1], [2, 3]], std_mode="batch")
    # Both groups are centered independently, then divided by the SAME batch std,
    # so the wider group keeps a larger advantage magnitude.
    assert float(adv[1]) > float(adv[3])
    assert float(adv[1] / adv[3]) == pytest.approx(10.0, rel=1e-4)


def test_normalize_grouped_none_mode_centers_without_dividing():
    rewards = torch.tensor([1.0, 3.0, 10.0, 10.2])
    adv = fg.normalize_grouped(rewards, [[0, 1], [2, 3]], std_mode="none")
    assert adv.tolist() == pytest.approx([-1.0, 1.0, -0.1, 0.1], abs=1e-5)


def test_normalize_grouped_unknown_mode_raises():
    with pytest.raises(ValueError, match="std_mode"):
        fg.normalize_grouped(torch.tensor([1.0, 2.0]), [[0, 1]], std_mode="global")


def test_normalize_grouped_singleton_group_is_zero_not_nan():
    """A 1-member group carries no relative signal → 0, never NaN.

    torch's std is unbiased, so a singleton group's std is NaN; dividing by it
    used to emit a NaN advantage that poisoned the whole optimizer step.
    """
    rewards = torch.tensor([1.0, 2.0, 3.0])
    adv = fg.normalize_grouped(rewards, [[0], [1, 2]])
    assert torch.isfinite(adv).all()
    assert float(adv[0]) == 0.0
    assert float(adv[1]) < 0 < float(adv[2])


def test_normalize_grouped_empty_group_is_skipped():
    adv = fg.normalize_grouped(torch.tensor([1.0, 2.0]), [[], [0, 1]])
    assert torch.isfinite(adv).all()


def test_combine_component_advantages_weighted_sum():
    comp = {
        "pickscore": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "aesthetic": torch.tensor([4.0, 3.0, 2.0, 1.0]),
    }
    gi = [[0, 1], [2, 3]]
    adv = fg.combine_component_advantages(comp, {"pickscore": 0.5, "aesthetic": 0.5}, gi)
    assert adv.shape == (4,)
    assert torch.isfinite(adv).all()


def test_combine_component_advantages_passes_std_mode():
    comp = {"pickscore": torch.tensor([0.0, 10.0, 0.0, 1.0])}
    gi = [[0, 1], [2, 3]]
    adv = fg.combine_component_advantages(comp, {"pickscore": 1.0}, gi, std_mode="none")
    assert adv.tolist() == pytest.approx([-5.0, 5.0, -0.5, 0.5], abs=1e-5)


def test_combine_component_advantages_missing_component_raises():
    with pytest.raises(ValueError):
        fg.combine_component_advantages({"aesthetic": torch.zeros(2)}, {"pickscore": 1.0}, [[0, 1]])


def test_flow_grpo_std_dev_t_clamps_only_at_sigma_one():
    """The derived coefficient is exact away from sigma==1 and clamped at it.

    The clamp is the boundary where a replay silently disagrees with the
    sampler: at sigma==1 the true sqrt(sigma/(1-sigma)) diverges, so the value
    is whatever ``sigma_max`` says. Training step 0 is rejected upstream at
    config validation; this only pins that the tensor stays finite.
    """
    eta, smax = 0.7, 0.99
    interior = torch.tensor(0.87958)
    got = fg.flow_sde_transition_std_dev_t(interior, eta, smax)
    want = torch.sqrt(interior / (1 - interior)) * eta
    assert torch.allclose(got, want, atol=1e-6)

    at_one = fg.flow_sde_transition_std_dev_t(torch.tensor(1.0), eta, smax)
    assert torch.isfinite(at_one)
    assert torch.allclose(at_one, torch.tensor((1.0 / (1 - smax)) ** 0.5 * eta), atol=1e-6)
