# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Advantage-stage replay adapters.

Reuses compute_approx_kl / get_grpo_returns from relax.utils.training.ppo_utils
(torch-only imports, safe offline).
"""

from __future__ import annotations

from typing import Any

import torch

from relax.utils.replay.bundle import LoadedBundle
from relax.utils.replay.compare import report_tensor, tolerance_for
from relax.utils.replay.report import StageResult, StageStatus
from relax.utils.replay.schema import StageId
from relax.utils.replay.stages import register_adapter
from relax.utils.training.ppo_utils import compute_approx_kl, get_grpo_returns


@register_adapter(StageId.ADVANTAGE_KL)
def replay_advantage_kl(bundle: LoadedBundle, ctx: dict[str, Any]) -> StageResult:
    """Per-token KL between rollout/old log-probs and reference log-probs."""
    config = bundle.index.config
    result = StageResult(stage=StageId.ADVANTAGE_KL.value, status=StageStatus.PASS)

    old_log_probs = bundle.tensors["old_log_probs"]
    if config.kl_coef == 0:
        kl = torch.zeros_like(old_log_probs, dtype=torch.float32)
    else:
        kl = compute_approx_kl(old_log_probs, bundle.tensors["ref_log_probs"], kl_loss_type=config.kl_loss_type)

    ctx[StageId.ADVANTAGE_KL.value] = kl
    atol, rtol = tolerance_for(bundle.manifest, StageId.ADVANTAGE_KL)
    return report_tensor(
        result,
        expected=bundle.tensors.get("kl"),
        actual=kl,
        field="kl",
        atol=atol,
        rtol=rtol,
        sample_ids=bundle.sample_ids,
        response_lengths=bundle.response_lengths,
        missing="no expected kl payload recorded",
        mismatch="{n} kl token(s) diverged",
    )


@register_adapter(StageId.ADVANTAGE_ESTIMATE)
def replay_advantage_estimate(bundle: LoadedBundle, ctx: dict[str, Any]) -> StageResult:
    """GRPO-family returns/advantages from normalized rewards + KL."""
    result = StageResult(stage=StageId.ADVANTAGE_ESTIMATE.value, status=StageStatus.PASS)

    normalized_rewards = ctx.get(StageId.REWARD_POST_PROCESS.value)
    if normalized_rewards is None:
        result.status = StageStatus.FAIL
        result.message = "upstream reward.post_process output missing"
        return result
    kl = ctx.get(StageId.ADVANTAGE_KL.value)
    if kl is None:
        result.status = StageStatus.FAIL
        result.message = "upstream advantage.kl output missing"
        return result

    rewards = torch.tensor(normalized_rewards, dtype=torch.float32)
    advantages = torch.cat(get_grpo_returns(rewards, list(torch.split(kl, bundle.response_lengths))))
    ctx[StageId.ADVANTAGE_ESTIMATE.value] = advantages

    atol, rtol = tolerance_for(bundle.manifest, StageId.ADVANTAGE_ESTIMATE)
    return report_tensor(
        result,
        expected=bundle.tensors.get("advantages"),
        actual=advantages,
        field="advantages",
        atol=atol,
        rtol=rtol,
        sample_ids=bundle.sample_ids,
        response_lengths=bundle.response_lengths,
        missing="no expected advantages payload recorded",
        mismatch="{n} advantage token(s) diverged",
    )
