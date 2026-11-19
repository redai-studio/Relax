# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Independent numeric oracles for the canonical scalar image ART profile."""

import pytest
import torch

from relax.models.diffusion_art import art_scalar_covariance, diffusion_art_loss


def test_complete_group_covariance_matches_hand_calculation_and_is_frozen():
    rewards = torch.tensor([-3.0, -1.0, 1.0, 3.0], requires_grad=True)
    endpoints = torch.stack([rewards, -rewards], 1).detach().requires_grad_(True)
    result = art_scalar_covariance(endpoints, rewards, torch.zeros(4, dtype=torch.long), 1)
    torch.testing.assert_close(result["group_covariance"], torch.tensor([[20 / 3, -20 / 3]]))
    torch.testing.assert_close(result["group_directions"], torch.tensor([[1.0, -1.0]]))
    assert result["group_guidance_magnitudes"].tolist() == [1.0]
    assert result["group_reward_abs_means"].tolist() == [2.0]
    assert result["group_active"].tolist() == [True]
    assert all(not value.requires_grad for value in result.values())


def test_interleaved_groups_do_not_mix_reward_evidence():
    groups = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    r = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    x = torch.stack([r, -r], 1).repeat_interleave(2, dim=0)
    rewards = torch.stack([r, -r], 1).flatten()
    result = art_scalar_covariance(x, rewards, groups, 2)
    torch.testing.assert_close(result["group_directions"], torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))
    assert result["group_counts"].tolist() == [4, 4]


def test_small_nonzero_signal_preserves_historical_epsilon_floor():
    rewards = torch.tensor([-1.0, 1.0])
    x = torch.tensor([[-1e-8, 1e-8], [1e-8, -1e-8]])
    result = art_scalar_covariance(x, rewards, torch.tensor([0, 0]), 1)
    torch.testing.assert_close(result["group_directions"], torch.tensor([[0.002, -0.002]]))
    assert result["group_active"].item()


@pytest.mark.parametrize("kind", ["constant", "zero_covariance", "singleton"])
def test_exact_zero_signal_is_finite_without_a_variance_threshold(kind):
    rewards = torch.full((10,), 0.1) if kind == "constant" else torch.arange(10).float()
    x = torch.ones(10, 2)
    if kind == "constant":
        x[:, 0] = torch.arange(10)
    if kind == "singleton":
        x, rewards = x[:1], rewards[:1]
    result = art_scalar_covariance(x, rewards, torch.zeros(len(x), dtype=torch.long), 2)
    assert not result["group_active"].any()
    assert torch.count_nonzero(result["group_guidance_magnitudes"]) == 0
    assert torch.count_nonzero(result["group_directions"]) == 0
    assert all(torch.isfinite(value).all() for value in result.values())


@pytest.mark.parametrize(
    "alpha,masked", [(0, False), (0.005, False), (0.01, True), (0.25, True), (1, False), (2, False), (-0.25, False)]
)
def test_directional_mask_preserves_overshoot_corrections(alpha, masked):
    direction = torch.tensor([[1.0, -1.0]], requires_grad=True)
    current = (-alpha * direction).detach().requires_grad_(True)
    old = torch.zeros_like(current, requires_grad=True)
    sigma = torch.tensor([0.5], requires_grad=True)
    magnitude = torch.ones(1, requires_grad=True)
    dt = torch.tensor([-0.1], requires_grad=True)
    result = diffusion_art_loss(current, old, sigma, direction, magnitude, ode_velocity_coefficient=dt)
    expected = 0.0 if masked else 0.5 * (1 - alpha) ** 2
    torch.testing.assert_close(result["per_item_loss"], torch.tensor([expected]))
    torch.testing.assert_close(result["target"], torch.tensor([[-1.0, 1.0]]))
    assert result["masked"].item() == masked
    result["per_item_loss"].sum().backward()
    expected_grad = torch.zeros_like(current) if masked else 0.5 * (current.detach() + direction.detach())
    torch.testing.assert_close(current.grad, expected_grad)
    assert all(value.grad is None for value in (old, direction, sigma, magnitude, dt))
    assert not result["target"].requires_grad


def test_initial_sigma_mask_metric_and_disabling_mask_match_equations():
    current = torch.tensor([[-0.25, 0.25]], requires_grad=True)
    kwargs = dict(
        current_velocity=current,
        old_velocity=torch.zeros_like(current),
        sigma=torch.ones(1),
        guidance_direction=torch.tensor([[1.0, -1.0]]),
        guidance_magnitude=torch.ones(1),
        ode_velocity_coefficient=torch.tensor([-0.1]),
    )
    result = diffusion_art_loss(**kwargs)
    torch.testing.assert_close(result["movement_score"], torch.tensor([0.1**2 * 0.25**2 / (2 * 0.8**2)]))
    assert result["masked"].item()
    unmasked = diffusion_art_loss(**kwargs, corrective_mask_enabled=False)
    assert not unmasked["masked"].item()
    torch.testing.assert_close(unmasked["per_item_loss"], torch.tensor([0.75**2]))


def test_masked_and_zero_guidance_rows_stay_in_original_denominator():
    current = torch.tensor([[0.0, 0.0], [-0.25, 0.25], [1.0, -1.0]], requires_grad=True)
    result = diffusion_art_loss(
        current,
        torch.zeros_like(current),
        torch.full((3,), 0.5),
        torch.tensor([[1.0, -1.0]]).expand(3, -1),
        torch.tensor([1.0, 1.0, 0.0]),
        ode_velocity_coefficient=torch.full((3,), -0.1),
    )
    loss = result["per_item_loss"].mean()
    torch.testing.assert_close(loss, torch.tensor(0.5 / 3))
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([[1 / 6, -1 / 6], [0.0, 0.0], [0.0, 0.0]]))


def test_structural_and_scheduler_contracts_fail_explicitly():
    with pytest.raises(ValueError, match="integer"):
        art_scalar_covariance(torch.ones(2, 2), torch.ones(2), torch.ones(2), 1)
    with pytest.raises(RuntimeError, match="group index"):
        art_scalar_covariance(torch.ones(2, 2), torch.ones(2), torch.tensor([0, 1]), 1)
    with pytest.raises(RuntimeError, match="negative"):
        diffusion_art_loss(
            torch.ones(2, 2),
            torch.zeros(2, 2),
            torch.ones(2),
            torch.ones(2, 2),
            torch.ones(2),
            ode_velocity_coefficient=torch.zeros(2),
        )
