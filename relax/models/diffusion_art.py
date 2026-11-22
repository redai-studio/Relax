# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Canonical scalar DiffusionART kernels, independent of Ray and model
families.

Equations and contracts follow vanzll/DiffusionART at
58e2052278f935a90fffbef5624d93e458085c07 (Apache-2.0; see
docs/zh/guide/diffusion-art.md). This first implementation deliberately
supports only the scalar image profile: unbiased covariance, historical
direction normalization, exact constant-reward handling, and sigma-weighted
velocity MSE with the scheduler-reference corrective mask.
"""

from __future__ import annotations

import math
from typing import Dict

import torch


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _vector(value: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 0 or value.shape[0] != reference.shape[0] or value.numel() != reference.shape[0]:
        raise ValueError(f"{name} must contain one value per candidate.")
    return value.detach().to(device=reference.device, dtype=torch.float32).reshape(-1)


@torch.no_grad()
def art_scalar_covariance(
    endpoints: torch.Tensor,
    rewards: torch.Tensor,
    row_group_indices: torch.Tensor,
    group_count: int,
    *,
    guidance_gain: float = 1.0,
    eps: float = 1e-5,
) -> Dict[str, torch.Tensor]:
    """Aggregate complete rollout groups BEFORE slicing candidates or ranks.

    Rewards must already be scalarized; group IDs identify conditioning
    instances, not prompt text. The caller must supply every member of each
    group exactly once. Empty/singleton groups are inactive. No valid-count
    filtering or reward standardization is implicitly inserted here.
    """
    if endpoints.ndim < 2 or endpoints.numel() == 0:
        raise ValueError("endpoints must have nonempty candidate and feature dimensions.")
    if type(group_count) is not int or group_count <= 0:
        raise ValueError("group_count must be a positive integer.")
    count = endpoints.shape[0]
    if rewards.shape != (count,) or row_group_indices.shape != (count,):
        raise ValueError("rewards and row_group_indices must have shape [candidates].")
    if row_group_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("row_group_indices must contain integer indices.")
    eps = _positive(eps, "eps")
    gain = float(guidance_gain)
    if not math.isfinite(gain) or gain < 0:
        raise ValueError("guidance_gain must be finite and nonnegative.")
    x = endpoints.detach().float().flatten(1)
    r = rewards.detach().to(device=x.device, dtype=torch.float32)
    groups = row_group_indices.to(device=x.device, dtype=torch.long)
    # Async assertions avoid a GPU-to-host synchronization in tensor kernels.
    torch._assert_async(((groups >= 0) & (groups < group_count)).all(), "group index out of range")
    if x.device.type == "cpu":
        low, high = torch.aminmax(x)
        finite_endpoints = torch.isfinite(low) & torch.isfinite(high)
    else:
        finite_endpoints = torch.isfinite(x).all()
    torch._assert_async(finite_endpoints & torch.isfinite(r).all(), "endpoints and rewards must be finite")
    counts = torch.bincount(groups, minlength=group_count).float()
    sums = x.new_zeros(group_count).index_add_(0, groups, r)
    squares = x.new_zeros(group_count).index_add_(0, groups, r.square())
    means = sums / counts.clamp_min(1)
    stds = (squares / counts.clamp_min(1) - means.square()).clamp_min(0).sqrt()
    maxima = r.new_full((group_count,), -torch.inf).scatter_reduce_(0, groups, r, reduce="amax")
    minima = r.new_full((group_count,), torch.inf).scatter_reduce_(0, groups, r, reduce="amin")
    constant = maxima == minima
    centered = torch.where(constant[groups], 0.0, r - means[groups])
    means = torch.where(constant, maxima, means)
    stds = torch.where(constant, 0.0, stds)
    mad = x.new_zeros(group_count).index_add_(0, groups, centered.abs()) / counts.clamp_min(1)
    covariance = x.new_zeros(group_count, x.shape[1]).index_add_(0, groups, centered[:, None] * x)
    covariance = covariance / (counts - 1).clamp_min(1)[:, None]
    covariance = torch.where((counts > 1)[:, None], covariance, 0.0)
    signal = covariance / torch.where(mad > 0, mad, 1.0)[:, None]
    signal_rms = signal.square().mean(1).sqrt()
    active = (counts > 1) & (mad > 0) & (signal_rms > 0) & (gain > 0)
    directions = torch.where(active[:, None], signal / signal_rms.clamp_min(eps)[:, None], 0.0)
    shape = (group_count, *endpoints.shape[1:])
    return {
        "group_covariance": covariance.reshape(shape),
        "group_directions": directions.reshape(shape),
        "group_guidance_magnitudes": torch.where(active, torch.full_like(mad, gain), 0.0),
        "group_active": active,
        "group_counts": counts,
        "group_reward_means": means,
        "group_reward_stds": stds,
        "group_reward_abs_means": mad,
        "group_covariance_rms": covariance.square().mean(1).sqrt(),
    }


def diffusion_art_loss(
    current_velocity: torch.Tensor,
    old_velocity: torch.Tensor,
    sigma: torch.Tensor,
    guidance_direction: torch.Tensor,
    guidance_magnitude: torch.Tensor,
    *,
    ode_velocity_coefficient: torch.Tensor,
    corrective_mask_enabled: bool = True,
    q2_reference_sde_noise_level: float = 0.8,
    q2_reference_kl_threshold: float = 1e-5,
    eps: float = 1e-5,
) -> Dict[str, torch.Tensor]:
    """Return per-item losses; only current velocity carries gradients.

    The caller averages over the ORIGINAL global candidate/time count,
    including masked and zero-guidance rows. All velocity tensors must be
    finite and use noise-directed flow coordinates, with negative ODE dt. This
    scalar image profile does not restore zero-guidance rows.
    """
    if current_velocity.ndim < 2 or current_velocity.numel() == 0:
        raise ValueError("current_velocity must have nonempty candidate and feature dimensions.")
    if old_velocity.shape != current_velocity.shape or guidance_direction.shape != current_velocity.shape:
        raise ValueError("current/old velocity and guidance direction must have identical shapes.")
    eps = _positive(eps, "eps")
    noise = _positive(q2_reference_sde_noise_level, "q2_reference_sde_noise_level")
    threshold = _positive(q2_reference_kl_threshold, "q2_reference_kl_threshold")
    v = current_velocity.float()
    old = old_velocity.detach().to(v)
    raw = guidance_direction.detach().to(v)
    s = _vector(sigma, v, "sigma")
    magnitude = _vector(guidance_magnitude, v, "guidance_magnitude")
    dt = _vector(ode_velocity_coefficient, v, "ode_velocity_coefficient")
    torch._assert_async((torch.isfinite(s) & (s >= 0) & (s <= 1)).all(), "sigma must be finite in [0, 1]")
    torch._assert_async((torch.isfinite(dt) & (dt < 0)).all(), "ODE coefficients must be finite and negative")
    torch._assert_async(
        (torch.isfinite(magnitude) & (magnitude >= 0)).all(), "guidance magnitude must be finite and nonnegative"
    )
    broadcast_shape = (-1,) + (1,) * (v.ndim - 1)
    rms = raw.square().flatten(1).mean(1).sqrt()
    direction = raw / rms.clamp_min(eps).reshape(broadcast_shape)
    target = (old - magnitude.reshape(broadcast_shape) * direction).detach()
    residual = target - v
    unmasked = s * residual.square().flatten(1).mean(1)
    movement = v - old
    alignment = (movement * residual.detach()).flatten(1).mean(1)
    initial = torch.isclose(s, torch.ones_like(s), atol=eps, rtol=0)
    std = torch.where(initial, torch.full_like(s, noise), (noise**2 * s / (1 - s).clamp_min(eps) * (-dt)).sqrt())
    coefficient = torch.where(initial, dt, dt * (1 + 0.5 * noise**2))
    energy = coefficient.square() * movement.square().flatten(1).mean(1) / (2 * std.square().clamp_min(eps**2))
    active = (magnitude > 0) & (rms > 0) & (s > 0)
    masked = (active & corrective_mask_enabled & (energy >= threshold) & (alignment > 0)).detach()
    keep = (active & ~masked).detach()
    return {
        "per_item_loss": torch.where(keep, unmasked, torch.zeros_like(unmasked)),
        "per_item_unmasked_loss": unmasked.detach(),
        "target": target,
        "keep_mask": keep,
        "masked": masked,
        "movement_score": energy.detach(),
        "outward_alignment": alignment.detach(),
    }
