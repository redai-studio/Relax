# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Framework-agnostic FlowGRPO math for native generative RL.

This module is the single source of truth for the Flow-SDE transition, its
Gaussian transition log-probability, grouped advantage normalization, and the
clipped PPO objective.

It is deliberately pure ``torch`` with no Relax / SGLang / diffusers imports so
it can be unit-tested on CPU and reused by any model adapter. The transition
math mirrors FlowGRPO (see the design doc section 6) and is the exact formula
implemented by the rollout engine, so rollout-sampling and actor-replay
log-probs match bit-for-bit on the first on-policy update.

Sampling itself lives entirely in the rollout engine — this module only ever
*scores* a stored transition, so nothing here draws noise.

All internal arithmetic runs in float32 regardless of the input dtype: mean,
variance, log-prob and ratio are precision-sensitive (the ``1 / (2 * sigma**2)``
factor amplifies any float64/float32 mismatch), while the model forward and the
stored trajectory use bf16.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Tuple

import torch


__all__ = [
    "ADVANTAGE_STD_MODES",
    "dance_sde_transition_std_dev_t",
    "flow_sde_transition_moments",
    "flow_sde_transition_std",
    "flow_sde_transition_std_dev_t",
    "flow_sde_log_prob",
    "replay_transition_logp",
    "grpo_clip_loss",
    "normalize_grouped",
    "combine_component_advantages",
]

ADVANTAGE_STD_MODES = ("group", "batch", "none")


# ---------------------------------------------------------------------------
# Flow-SDE transition (single source of truth, shared by rollout + replay)
# ---------------------------------------------------------------------------


def flow_sde_transition_std_dev_t(
    sigma: torch.Tensor,
    eta: float,
    sigma_max: float = 0.99,
) -> torch.Tensor:
    """Diffusion coefficient ``std_dev_t = sqrt(sigma / (1 - sigma)) * eta``.

    ``sigma == 1`` (the first denoising step) would divide by zero, so it is
    clamped to ``sigma_max``. The clamped value is arbitrary and does NOT match
    what the sampler used, so a replay of step 0 scores a different Gaussian
    than the one the sample was drawn from (measured against the SGLang
    diffusion engine: std_dev_t 7.0 vs ~2.92, which flips the sign of the
    ``sample`` coefficient in the transition mean and inflates the std 2.4x;
    every later step agrees to 1e-4). Training step 0 is therefore rejected
    upstream by the actor's ``sde_type="sde"`` config validation — the clamp
    only keeps the tensor finite, it is not a fix.
    """
    denom = 1 - torch.where(sigma == 1, torch.full_like(sigma, sigma_max), sigma)
    return torch.sqrt(sigma / denom) * eta


def dance_sde_transition_std_dev_t(
    sigma: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    """DanceGRPO diffusion coefficient: constant ``std_dev_t = eta``."""
    return torch.full_like(sigma, float(eta))


def _transition_std_dev_t(sigma: torch.Tensor, eta: float, sigma_max: float, sde_type: str) -> torch.Tensor:
    """Dispatch the diffusion coefficient for ``sde_type``."""
    if sde_type == "dance":
        return dance_sde_transition_std_dev_t(sigma, eta)
    if sde_type == "sde":
        return flow_sde_transition_std_dev_t(sigma, eta, sigma_max)
    raise ValueError(f"Unsupported flow SDE type: {sde_type!r}")


def flow_sde_transition_std(
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    eta: float,
    sigma_max: float = 0.99,
    sde_type: str = "sde",
) -> torch.Tensor:
    """Std of the per-step transition Gaussian ``N(mean, std**2)``.

    ``std = std_dev_t * sqrt(-dt)``. This is the same normalizer
    :func:`flow_sde_transition_moments` returns, exposed on its own for the
    actor's per-step debug log (which wants the noise scale without a model
    forward).
    """
    dt = sigma_next - sigma
    return _transition_std_dev_t(sigma, eta, sigma_max, sde_type) * torch.sqrt(-dt)


def flow_sde_transition_moments(
    noise_pred: torch.Tensor,
    sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    eta: float = 1.0,
    sigma_max: float = 0.99,
    sde_type: str = "sde",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and std of one Flow-SDE denoising transition ``x_t -> x_next``.

    Returns ``(prev_sample_mean, std)`` of the Gaussian the sampler would have
    drawn ``x_next`` from, evaluated at the *current* model prediction
    ``noise_pred``. Only the mean carries a gradient; the std is a constant
    w.r.t. the model parameters.
    """
    dt = sigma_next - sigma
    std_dev_t = _transition_std_dev_t(sigma, eta, sigma_max, sde_type)

    prev_sample_mean = (
        sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + noise_pred * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )
    return prev_sample_mean, std_dev_t * torch.sqrt(-dt)


def flow_sde_log_prob(
    prev_sample: torch.Tensor,
    prev_sample_mean: torch.Tensor,
    std_var: torch.Tensor,
) -> torch.Tensor:
    """Elementwise Gaussian log-prob of ``prev_sample`` under ``N(mean,
    std_var**2)``."""
    return (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_var**2)
        - torch.log(std_var)
        - 0.5 * math.log(2 * math.pi)
    )


def _broadcast_sigma(sigma: torch.Tensor, ndim: int) -> torch.Tensor:
    sigma = sigma.float()
    if sigma.dim() == 0:
        sigma = sigma.unsqueeze(0)
    while sigma.dim() < ndim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def replay_transition_logp(
    noise_pred: torch.Tensor,
    x_t: torch.Tensor,
    x_next: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    eta: float = 1.0,
    sigma_max: float = 0.99,
    reduce: bool = True,
    sde_type: str = "sde",
) -> torch.Tensor:
    """Per-sample log-prob of a stored ``x_t -> x_next`` transition at replay
    time.

    ``noise_pred`` is the model velocity prediction at ``x_t`` under the
    current weights. All math runs in float32 (the stored trajectory is bf16;
    the log-prob and the PPO ratio are not). With ``reduce`` the log-prob is
    mean-reduced over all non-batch dims to shape ``[B]``; otherwise the
    elementwise log-prob is returned.
    """
    noise_pred = noise_pred.float()
    x_t = x_t.float()
    x_next = x_next.float()

    sigma = _broadcast_sigma(sigma, x_t.dim())
    sigma_next = _broadcast_sigma(sigma_next, x_t.dim())

    prev_sample_mean, std_var = flow_sde_transition_moments(
        noise_pred=noise_pred,
        sample=x_t,
        sigma=sigma,
        sigma_next=sigma_next,
        eta=eta,
        sigma_max=sigma_max,
        sde_type=sde_type,
    )
    logp = flow_sde_log_prob(x_next, prev_sample_mean, std_var)
    if reduce:
        logp = logp.mean(dim=tuple(range(1, logp.ndim)))
    return logp


# ---------------------------------------------------------------------------
# Clipped PPO objective
# ---------------------------------------------------------------------------


def grpo_clip_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
    clip_eps_high: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """PPO-style clipped objective (per-sample); reduction is the caller's job.

    ``ratio = exp(new_logp - old_logp)``,
    ``loss = max(-adv * ratio, -adv * clamp(ratio, 1-eps, 1+eps_high))``.

    ``clip_eps_high`` (DAPO clip-higher) defaults to ``clip_eps`` (symmetric).
    Returns ``(loss_per_sample, detached_metrics)``.
    """
    high = clip_eps if clip_eps_high is None else clip_eps_high
    log_diff = new_logp - old_logp
    ratio = torch.exp(log_diff)
    adv = advantages.detach()
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + high)
    loss_per_sample = torch.maximum(unclipped, clipped)

    ratio_std = ratio.std() if ratio.numel() > 1 else torch.zeros((), dtype=ratio.dtype, device=ratio.device)
    gt = (ratio - 1.0 > high).float()
    lt = (1.0 - ratio > clip_eps).float()
    metrics = {
        "ratio_mean": ratio.mean().detach(),
        "ratio_std": ratio_std.detach(),
        "ratio_min": ratio.min().detach(),
        "ratio_max": ratio.max().detach(),
        "clip_fraction": torch.maximum(gt, lt).mean().detach(),
        "approx_kl": (0.5 * log_diff.pow(2)).mean().detach(),
    }
    return loss_per_sample, metrics


# ---------------------------------------------------------------------------
# Advantage normalization
# ---------------------------------------------------------------------------


def normalize_grouped(
    rewards: torch.Tensor,
    group_indices: List[List[int]],
    epsilon: float = 1e-6,
    std_mode: str = "group",
) -> torch.Tensor:
    """Per-group centered advantage with an explicit std divisor mode.

    ``group_indices`` lists the flat sample indices belonging to each prompt
    group. The group mean is always subtracted. ``std_mode`` then controls the
    divisor:

    - ``group``: divide by each group's own std, matching Relax/Text GRPO.
    - ``batch``: divide every group-centered value by one batch-wide std,
      matching reference diffusion recipes that enable global std.
    - ``none``: do not divide, i.e. Dr.GRPO's centered-only variant.
    """
    if std_mode not in ADVANTAGE_STD_MODES:
        raise ValueError(f"normalize_grouped: std_mode must be one of {ADVANTAGE_STD_MODES}, got {std_mode!r}.")
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    rewards = rewards.float()
    batch_std: Optional[torch.Tensor] = None
    if std_mode == "batch" and rewards.numel() > 1:
        batch_std = rewards.std() + epsilon
    for indices in group_indices:
        if not indices:
            continue
        group_rewards = rewards[indices]
        centered = group_rewards - group_rewards.mean()
        if std_mode == "none":
            advantages[indices] = centered
            continue
        if std_mode == "batch":
            advantages[indices] = centered / batch_std if batch_std is not None else centered
            continue
        # A group with fewer than 2 members carries no relative signal, so its
        # advantage is 0 (already the initialized value). Skipping is also what
        # keeps this finite: torch's std is unbiased (dof = n - 1), so a
        # singleton group's std is NaN and (x - mean) / (NaN + eps) would put a
        # NaN advantage into the loss and poison the whole optimizer step.
        # Reachable: relax/utils/arguments.py forces grpo_std_normalization=False
        # at n_samples_per_prompt == 1, but the group lists themselves can still
        # be ragged.
        if len(indices) < 2:
            continue
        advantages[indices] = centered / (group_rewards.std() + epsilon)
    return advantages


def combine_component_advantages(
    component_rewards: Mapping[str, torch.Tensor],
    component_weights: Mapping[str, float],
    group_indices: List[List[int]],
    epsilon: float = 1e-6,
    std_mode: str = "group",
) -> torch.Tensor:
    """Normalize each reward component, then weight-combine.

    Every weighted component must be present in ``component_rewards``; a
    missing required component is the caller's fail-fast responsibility (see
    the generative reward manager).
    """
    total: Optional[torch.Tensor] = None
    for name, weight in component_weights.items():
        if name not in component_rewards:
            raise ValueError(
                f"combine_component_advantages: reward component {name!r} missing; have {sorted(component_rewards)}."
            )
        comp_adv = normalize_grouped(component_rewards[name], group_indices, epsilon=epsilon, std_mode=std_mode)
        contrib = float(weight) * comp_adv
        total = contrib if total is None else total + contrib
    if total is None:
        raise ValueError("combine_component_advantages: component_weights is empty.")
    return total
