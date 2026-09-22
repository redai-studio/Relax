# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reward-stage replay adapters.

reward.post_process restates group normalization from
relax.utils.utils.post_process_rewards (that module imports Ray at import
time). Semantics are pinned by the PR #65 fixture.
"""

from __future__ import annotations

from typing import Any

import torch

from relax.utils.replay.bundle import LoadedBundle
from relax.utils.replay.compare import report_scalar_list, tolerance_for
from relax.utils.replay.report import StageResult, StageStatus
from relax.utils.replay.schema import StageId
from relax.utils.replay.stages import register_adapter


@register_adapter(StageId.REWARD_RAW)
def replay_reward_raw(bundle: LoadedBundle, ctx: dict[str, Any]) -> StageResult:
    """Resolve the raw reward per sample (V1: scalar rewards only)."""
    result = StageResult(stage=StageId.REWARD_RAW.value, status=StageStatus.PASS)
    atol, rtol = tolerance_for(bundle.manifest, StageId.REWARD_RAW)
    return report_scalar_list(
        result,
        expected=bundle.expected.get(StageId.REWARD_RAW.value, {}).get("raw_rewards"),
        actual=[record.raw_reward for record in bundle.index.samples],
        sample_ids=bundle.sample_ids,
        field="raw_reward",
        atol=atol,
        rtol=rtol,
        missing="no expected raw_rewards recorded",
        mismatch="raw reward mismatch in {n} sample(s)",
    )


@register_adapter(StageId.REWARD_POST_PROCESS)
def replay_reward_post_process(bundle: LoadedBundle, ctx: dict[str, Any]) -> StageResult:
    """Recompute group-normalized rewards (PR #65: group by group_index)."""
    config = bundle.index.config
    result = StageResult(stage=StageId.REWARD_POST_PROCESS.value, status=StageStatus.PASS)

    rewards = torch.tensor([record.raw_reward for record in bundle.index.samples], dtype=torch.float)
    positions_by_group: dict[int, list[int]] = {}
    for position, record in enumerate(bundle.index.samples):
        if record.group_index is None:
            result.status = StageStatus.FAIL
            result.message = f"sample {record.sample_id!r} has no group_index"
            return result
        positions_by_group.setdefault(record.group_index, []).append(position)

    normalized = torch.empty_like(rewards)
    for group_index, positions in positions_by_group.items():
        if len(positions) != config.n_samples_per_prompt:
            result.status = StageStatus.FAIL
            result.message = (
                f"reward group {group_index} has {len(positions)} samples, expected {config.n_samples_per_prompt}"
            )
            return result
        group_rewards = rewards[positions]
        group_rewards = group_rewards - group_rewards.mean()
        if config.grpo_std_normalization:
            group_rewards = group_rewards / (group_rewards.std() + 1e-6)
        normalized[positions] = group_rewards

    ctx[StageId.REWARD_POST_PROCESS.value] = normalized.tolist()
    atol, rtol = tolerance_for(bundle.manifest, StageId.REWARD_POST_PROCESS)
    return report_scalar_list(
        result,
        expected=bundle.expected.get(StageId.REWARD_POST_PROCESS.value, {}).get("rewards"),
        actual=normalized.tolist(),
        sample_ids=bundle.sample_ids,
        field="reward",
        atol=atol,
        rtol=rtol,
        missing="no expected normalized rewards recorded",
        mismatch="normalized reward mismatch in {n} sample(s)",
    )
