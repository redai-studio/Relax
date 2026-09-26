# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Policy-loss replay adapter.

Reuses compute_policy_loss and restates CP=1 sum_of_sample_mean (per-sample
masked mean, then summed). KL-loss, OPD and TIS are out of V1.
"""

from __future__ import annotations

from typing import Any

import torch

from relax.utils.replay.bundle import LoadedBundle
from relax.utils.replay.compare import report_scalar_fields, tolerance_for
from relax.utils.replay.report import StageResult, StageStatus
from relax.utils.replay.schema import StageId
from relax.utils.replay.stages import register_adapter
from relax.utils.training.ppo_utils import compute_policy_loss


def _sum_of_sample_mean(x: torch.Tensor, response_lengths: list[int], loss_masks: list[list[int]]) -> torch.Tensor:
    """CP=1 sum_of_sample_mean: per-sample masked mean, summed over samples."""
    total = 0.0
    for chunk, mask in zip(torch.split(x, response_lengths), loss_masks, strict=False):
        mask_t = torch.tensor(mask, dtype=x.dtype, device=x.device)
        denominator = torch.clamp_min(mask_t.sum(), 1)
        total = total + (chunk * mask_t).sum() / denominator
    return total


@register_adapter(StageId.LOSS_POLICY)
def replay_loss_policy(bundle: LoadedBundle, ctx: dict[str, Any]) -> StageResult:
    """Recompute the GRPO policy loss and its reduction metrics."""
    config = bundle.index.config
    result = StageResult(stage=StageId.LOSS_POLICY.value, status=StageStatus.PASS)

    advantages = ctx.get(StageId.ADVANTAGE_ESTIMATE.value)
    if advantages is None:
        # Loss-only capture: recorded advantages are the stage input. Full-chain
        # replays still prefer ctx so upstream divergence propagates.
        advantages = bundle.tensors.get("advantages")
    if advantages is None:
        result.status = StageStatus.FAIL
        result.message = "upstream advantage.estimate output missing"
        return result

    old_log_probs = bundle.tensors["old_log_probs"]
    log_probs = bundle.tensors["log_probs"]
    entropy = bundle.tensors["entropy"]
    loss_masks = [record.loss_mask for record in bundle.index.samples]

    ppo_kl = old_log_probs - log_probs
    pg_loss, clipfrac = compute_policy_loss(ppo_kl, advantages, config.eps_clip, config.eps_clip_high)

    pg_loss_scalar = _sum_of_sample_mean(pg_loss, bundle.response_lengths, loss_masks)
    clipfrac_scalar = _sum_of_sample_mean(clipfrac, bundle.response_lengths, loss_masks)
    ppo_kl_scalar = _sum_of_sample_mean(ppo_kl, bundle.response_lengths, loss_masks)
    entropy_loss = _sum_of_sample_mean(entropy, bundle.response_lengths, loss_masks)

    atol, rtol = tolerance_for(bundle.manifest, StageId.LOSS_POLICY)
    return report_scalar_fields(
        result,
        expected=bundle.expected.get(StageId.LOSS_POLICY.value),
        actual={
            "loss": float((pg_loss_scalar - config.entropy_coef * entropy_loss).item()),
            "pg_loss": float(pg_loss_scalar.item()),
            "entropy_loss": float(entropy_loss.item()),
            "pg_clipfrac": float(clipfrac_scalar.item()),
            "ppo_kl": float(ppo_kl_scalar.item()),
        },
        atol=atol,
        rtol=rtol,
        missing="no expected loss metrics recorded",
        mismatch="loss divergence in fields {fields}",
    )
