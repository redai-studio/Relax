# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Top-K score centering with detached importance weights and head
residuals."""

import math

import torch


def compute_score_centering_loss(
    *,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    train_topk_log_probs: torch.Tensor,
    rollout_topk_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    mode: str = "none",
    tis_clip: float = 2.0,
    mis_low: float = 0.5,
    mis_high: float = 5.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return token losses and detached token diagnostics before reduction.

    Sample tensors are [N], candidate tensors are [N, K]. Candidate padding has
    rollout log-probability -inf. Masked rows contribute zero, including rows
    whose placeholders contain non-finite values. FP64 is retained for
    numerical oracles; model probabilities otherwise use FP32.
    """
    if (
        log_probs.ndim != 1
        or advantages.shape != log_probs.shape
        or rollout_log_probs.shape != log_probs.shape
        or loss_mask.shape != log_probs.shape
        or train_topk_log_probs.ndim != 2
        or train_topk_log_probs.shape[0] != log_probs.shape[0]
        or rollout_topk_log_probs.shape != train_topk_log_probs.shape
    ):
        raise ValueError("Score centering requires sample tensors [N] and candidate tensors [N, K]")
    dtype = torch.float64 if log_probs.dtype == torch.float64 else torch.float32
    active = loss_mask != 0
    head_mask = active[:, None] & ~torch.isneginf(rollout_topk_log_probs)
    train_head = torch.where(head_mask, train_topk_log_probs.to(dtype), 0.0)
    rollout_head = torch.where(head_mask, rollout_topk_log_probs.to(dtype), 0.0)
    train_sample = torch.where(active, log_probs.to(dtype), 0.0)
    rollout_sample = torch.where(active, rollout_log_probs.to(dtype), 0.0)
    advantage = torch.where(active, advantages.detach().to(dtype), 0.0)

    with torch.no_grad():
        p = train_head.exp().masked_fill(~head_mask, 0.0)
        q = rollout_head.exp().masked_fill(~head_mask, 0.0)
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        p_tail, q_tail = 1 - p_mass, 1 - q_mass
        rho = q_tail.clamp_min(1e-6) / p_tail.clamp_min(1e-6)
        sample_ratio_log = train_sample - rollout_sample
        clip_fraction = torch.zeros_like(train_sample)
        if mode == "none":
            head_mass, alpha = q, rho
            weight = torch.ones_like(train_sample)
        elif mode == "tis":
            head_mass = torch.minimum(p, tis_clip * q)
            alpha = (tis_clip * rho).clamp_max(1)
            weight = sample_ratio_log.clamp_max(math.log(tis_clip)).exp()
            clip_fraction = (sample_ratio_log > math.log(tis_clip)).to(dtype)
        elif mode == "mis":
            head_ratio_log = train_head - rollout_head
            head_inside = (head_ratio_log >= math.log(mis_low)) & (head_ratio_log <= math.log(mis_high))
            head_mass = torch.where(head_inside, p, 0.0)
            alpha = ((rho >= 1 / mis_high) & (rho <= 1 / mis_low)).to(dtype)
            inside = (sample_ratio_log >= math.log(mis_low)) & (sample_ratio_log <= math.log(mis_high))
            weight = torch.where(inside, sample_ratio_log.clamp_max(math.log(mis_high)).exp(), 0.0)
            clip_fraction = (~inside).to(dtype)
        else:
            raise ValueError(f"Unknown score-centering importance weighting: {mode}")
        residual = head_mass - alpha[:, None] * p

    # Mask before multiplication: zero weights and padding may accompany -inf.
    correction = (residual * torch.where(residual != 0, train_head, 0.0)).sum(-1)
    sampled_term = weight * torch.where(weight != 0, train_sample, 0.0)
    loss = -advantage * torch.where(advantage != 0, sampled_term - correction, 0.0)
    metrics = {
        "sc_rollout_head_mass": q_mass,
        "sc_train_head_mass": p_mass,
        "sc_tail_ratio": rho,
        "sc_tail_floor_fraction": ((p_tail < 1e-6) | (q_tail < 1e-6)).to(dtype),
        "sc_correction": correction,
        "sc_importance_weight": weight,
        "sc_is_clip_fraction": clip_fraction,
        "sc_zero_weight_fraction": (weight == 0).to(dtype),
    }
    return loss, {name: torch.where(active, value.detach(), 0.0) for name, value in metrics.items()}
