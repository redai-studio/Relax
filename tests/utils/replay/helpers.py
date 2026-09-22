# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Test helpers that build small, hand-checkable GRPO replay bundles.

The fixture is deliberately tiny and deterministic: 4 samples in 2 semantic
groups of 2 (n_samples_per_prompt=2), each with 2 response tokens. Expected
outputs are computed from pristine inputs with reference implementations that
are independent of the adapters under test; a corrupt=... option then tampers
only the inputs written to the bundle, so replay must detect the divergence
against the pristine expected outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from relax.utils.replay.bundle import BundleWriter, metadata_checksums
from relax.utils.replay.capture import CaptureRecord
from relax.utils.replay.schema import (
    ActorStepId,
    BundleIndex,
    ComparisonPolicy,
    Identity,
    Manifest,
    ProducerInfo,
    RecomputeConfig,
    SampleRecord,
    StageCapability,
    StageContract,
    StageId,
    WeightLineage,
)
from relax.utils.training.ppo_utils import compute_approx_kl, compute_policy_loss, get_grpo_returns


GROUP_INDICES = [0, 0, 1, 1]
RAW_REWARDS = [1.0, 3.0, 10.0, 12.0]
RESPONSE_LENGTHS = [2, 2, 2, 2]
TOTAL_LENGTHS = [3, 3, 3, 3]
LOSS_MASKS = [[1, 1], [1, 1], [1, 1], [1, 1]]
# Two physical micro-batches: s-0/s-1 in mb-0007, s-2/s-3 in mb-0008 (each is
# also a complete semantic group, so batch selection has a well-defined closure).
MICRO_BATCH_IDS = ["mb-0007", "mb-0007", "mb-0008", "mb-0008"]

# Hand-derived ground truth for the default fixture (ratio == 1, no std norm).
# group0 rewards [1, 3] -> mean 2 -> [-1, 1]; group1 [10, 12] -> mean 11 -> [-1, 1].
NORMALIZED_REWARDS = [-1.0, 1.0, -1.0, 1.0]
# get_grpo_returns broadcasts each reward to its response length (2 tokens).
ADVANTAGES = [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0]
# ratio == 1 => pg_loss == -advantage, per-sample mean sums to 0; entropy_loss == 2.0.
DEFAULT_LOSS = -0.01 * 2.0


def _group_normalize(
    raw_rewards: list[float], group_indices: list[int], n_samples_per_prompt: int, std_norm: bool
) -> list[float]:
    """Reference group normalization, independent of the reward adapter."""
    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    positions: dict[int, list[int]] = {}
    for position, group_index in enumerate(group_indices):
        positions.setdefault(group_index, []).append(position)
    normalized = torch.empty_like(rewards)
    for group_index, group_positions in positions.items():
        assert len(group_positions) == n_samples_per_prompt, group_index
        group_rewards = rewards[group_positions]
        group_rewards = group_rewards - group_rewards.mean()
        if std_norm:
            group_rewards = group_rewards / (group_rewards.std() + 1e-6)
        normalized[group_positions] = group_rewards
    return normalized.tolist()


def _reduce(x: torch.Tensor, response_lengths: list[int], loss_masks: list[list[int]]) -> float:
    """Reference CP=1 reduction, independent of the loss adapter."""
    total = 0.0
    for chunk, mask in zip(torch.split(x, response_lengths), loss_masks, strict=False):
        mask_t = torch.tensor(mask, dtype=x.dtype)
        total += (chunk * mask_t).sum() / torch.clamp_min(mask_t.sum(), 1)
    return float(total.item())


def _build_manifest(bundle_id: str) -> Manifest:
    stage_contracts = {
        stage: StageContract(
            stage=stage,
            version="v1",
            capability=StageCapability.RECOMPUTE,
            implementation="reimplemented" if stage == StageId.REWARD_POST_PROCESS else "reuse",
        )
        for stage in (
            StageId.SAMPLE,
            StageId.REWARD_RAW,
            StageId.REWARD_POST_PROCESS,
            StageId.ADVANTAGE_KL,
            StageId.ADVANTAGE_ESTIMATE,
            StageId.LOSS_POLICY,
        )
    }
    stage_contracts[StageId.LOSS_VALUE] = StageContract(
        stage=StageId.LOSS_VALUE,
        version="v1",
        capability=StageCapability.UNSUPPORTED,
    )
    return Manifest(
        format_version="1.0.0",
        bundle_id=bundle_id,
        producer=ProducerInfo(commit="test", dirty_patch_digest="", torch_version=torch.__version__),
        stage_contracts=stage_contracts,
        payloads={},
        comparison_policy=ComparisonPolicy(
            exact_fields=["sample_id", "loss_mask", "group_index"],
            tolerances={},
        ),
        redaction={"prompt": "hash", "response": "truncate:64"},
    )


def _make_index(
    bundle_id: str, config: RecomputeConfig, raw_rewards: list[float], loss_masks: list[list[int]]
) -> BundleIndex:
    samples = [
        SampleRecord(
            sample_id=f"s-{index}",
            group_index=GROUP_INDICES[index],
            response_length=RESPONSE_LENGTHS[index],
            total_length=TOTAL_LENGTHS[index],
            loss_mask=list(loss_masks[index]),
            raw_reward=raw_rewards[index],
            reward=raw_rewards[index],
            label_hash="hash:label",
            micro_batch_id=MICRO_BATCH_IDS[index],
        )
        for index in range(len(raw_rewards))
    ]
    return BundleIndex(
        bundle_id=bundle_id,
        identity=Identity(
            actor_step_id=ActorStepId(rollout_id=120, step_id=0),
            rollout_partition_ids=["rollout-119"],
            consumer_batch_ids=["tq-7"],
            micro_batch_ids=["mb-0007"],
            semantic_group_ids=["g-0", "g-1"],
            normalization_cohort_ids=["g-0", "g-1"],
            weight_lineage=WeightLineage(rollout_weight="w-119", actor_weight="w-120"),
            rank={"dp": 0, "tp": 0, "pp": 0, "cp": 1},
        ),
        samples=samples,
        config=config,
    )


def build_grpo_bundle(
    path: str | Path,
    *,
    bundle_id: str = "b-00001",
    ratio_one: bool = True,
    pr65_bug: bool = False,
    kl_coef: float = 0.1,
    entropy_coef: float = 0.01,
    corrupt: str | None = None,
    raw_rewards: list[float] | None = None,
    old_log_probs: torch.Tensor | None = None,
    log_probs: torch.Tensor | None = None,
    ref_log_probs: torch.Tensor | None = None,
    loss_masks: list[list[int]] | None = None,
) -> tuple[Path, BundleIndex, dict[str, Any]]:
    """Build a GRPO CP=1 bundle and return (path, index, expected)."""
    raw_rewards = list(raw_rewards) if raw_rewards is not None else list(RAW_REWARDS)
    loss_masks = [list(mask) for mask in (loss_masks or LOSS_MASKS)]

    num_tokens = sum(RESPONSE_LENGTHS)
    if old_log_probs is None:
        old_log_probs = torch.zeros(num_tokens)
    if log_probs is None:
        log_probs = torch.zeros(num_tokens) if ratio_one else torch.full((num_tokens,), 0.5)
    if ref_log_probs is None:
        # Distinct from the (zero) rollout log-probs so the rollout-vs-ref KL is
        # actually exercised, matching production compute_advantages_and_returns.
        ref_log_probs = torch.full((num_tokens,), 0.25)
    entropy = torch.full((num_tokens,), 0.5)

    config = RecomputeConfig(
        advantage_estimator="grpo",
        n_samples_per_prompt=2,
        grpo_std_normalization=False,
        kl_loss_type="k1",
        kl_coef=kl_coef,
        eps_clip=0.2,
        eps_clip_high=0.2,
        entropy_coef=entropy_coef,
    )

    # Compute expected outputs from PRISTINE inputs.
    if pr65_bug:
        # Historical bug: normalize all samples as one physical batch.
        rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float)
        rewards_tensor = rewards_tensor - rewards_tensor.mean()
        normalized_rewards = rewards_tensor.tolist()
    else:
        normalized_rewards = _group_normalize(
            raw_rewards, GROUP_INDICES, config.n_samples_per_prompt, config.grpo_std_normalization
        )

    kl = (
        torch.zeros_like(old_log_probs, dtype=torch.float32)
        if config.kl_coef == 0
        else compute_approx_kl(old_log_probs, ref_log_probs, kl_loss_type=config.kl_loss_type)
    )
    rewards_tensor = torch.tensor(normalized_rewards, dtype=torch.float32)
    returns = get_grpo_returns(rewards_tensor, list(torch.split(kl, RESPONSE_LENGTHS)))
    advantages = torch.cat(returns)

    ppo_kl = old_log_probs - log_probs
    pg_loss, clipfrac = compute_policy_loss(ppo_kl, advantages, config.eps_clip, config.eps_clip_high)
    expected_loss = {
        "loss": _reduce(pg_loss, RESPONSE_LENGTHS, loss_masks)
        - config.entropy_coef * _reduce(entropy, RESPONSE_LENGTHS, loss_masks),
        "pg_loss": _reduce(pg_loss, RESPONSE_LENGTHS, loss_masks),
        "entropy_loss": _reduce(entropy, RESPONSE_LENGTHS, loss_masks),
        "pg_clipfrac": _reduce(clipfrac, RESPONSE_LENGTHS, loss_masks),
        "ppo_kl": _reduce(ppo_kl, RESPONSE_LENGTHS, loss_masks),
    }
    expected = {
        StageId.REWARD_RAW.value: {"raw_rewards": list(raw_rewards)},
        StageId.REWARD_POST_PROCESS.value: {"rewards": normalized_rewards},
        StageId.LOSS_POLICY.value: expected_loss,
    }

    # Tamper only the INPUTS written to the bundle, keeping expected pristine.
    if corrupt == "reward":
        raw_rewards[0] = 2.0
    elif corrupt == "mask_token":
        loss_masks[1][0] = 0
    elif corrupt == "old_log_probability":
        old_log_probs = old_log_probs.clone()
        old_log_probs[0] = 0.5
    elif corrupt == "nan":
        advantages[0] = float("nan")

    index = _make_index(bundle_id, config, raw_rewards, loss_masks)
    manifest = _build_manifest(bundle_id)

    writer = BundleWriter(path, manifest, index, expected)
    writer.write_payload("old_log_probs", old_log_probs.float())
    writer.write_payload("log_probs", log_probs.float())
    writer.write_payload("ref_log_probs", ref_log_probs.float())
    writer.write_payload("entropy", entropy.float())
    writer.write_payload("kl", kl.float())
    writer.write_payload("advantages", advantages.float())
    writer.finalize(ranks=[0])

    return Path(path), index, expected


def resign_metadata_checksums(bundle: Path) -> None:
    """Recompute COMPLETE / COMPLETE.<rank> metadata hashes after a test
    mutates files."""
    checksums = metadata_checksums(bundle)
    complete_path = bundle / "COMPLETE"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["metadata"] = checksums
    complete_path.write_text(
        json.dumps(complete, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for shard in sorted(bundle.glob("COMPLETE.*")):
        data = json.loads(shard.read_text(encoding="utf-8"))
        data["metadata"] = checksums
        shard.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def make_capture_record(bundle_id: str = "b-capture", ratio_one: bool = True) -> CaptureRecord:
    """Build a pristine GRPO CP=1 CaptureRecord (same reference data as
    build_grpo_bundle).

    This is the production-capture equivalent of the PR A fixture: a record the
    CaptureManager/build_bundle_from_record turns into a replayable bundle.
    """
    config = RecomputeConfig(
        advantage_estimator="grpo",
        n_samples_per_prompt=2,
        grpo_std_normalization=False,
        kl_loss_type="k1",
        kl_coef=0.1,
        eps_clip=0.2,
        eps_clip_high=0.2,
        entropy_coef=0.01,
    )
    num_tokens = sum(RESPONSE_LENGTHS)
    old_log_probs = torch.zeros(num_tokens)
    log_probs = torch.zeros(num_tokens) if ratio_one else torch.full((num_tokens,), 0.5)
    ref_log_probs = torch.full((num_tokens,), 0.25)
    entropy = torch.full((num_tokens,), 0.5)

    normalized_rewards = _group_normalize(
        RAW_REWARDS, GROUP_INDICES, config.n_samples_per_prompt, config.grpo_std_normalization
    )
    kl = compute_approx_kl(old_log_probs, ref_log_probs, kl_loss_type=config.kl_loss_type)
    rewards_tensor = torch.tensor(normalized_rewards, dtype=torch.float32)
    returns = get_grpo_returns(rewards_tensor, list(torch.split(kl, RESPONSE_LENGTHS)))
    advantages = torch.cat(returns)

    ppo_kl = old_log_probs - log_probs
    pg_loss, clipfrac = compute_policy_loss(ppo_kl, advantages, config.eps_clip, config.eps_clip_high)
    expected = {
        StageId.REWARD_RAW.value: {"raw_rewards": list(RAW_REWARDS)},
        StageId.REWARD_POST_PROCESS.value: {"rewards": normalized_rewards},
        StageId.LOSS_POLICY.value: {
            "loss": _reduce(pg_loss, RESPONSE_LENGTHS, LOSS_MASKS)
            - config.entropy_coef * _reduce(entropy, RESPONSE_LENGTHS, LOSS_MASKS),
            "pg_loss": _reduce(pg_loss, RESPONSE_LENGTHS, LOSS_MASKS),
            "entropy_loss": _reduce(entropy, RESPONSE_LENGTHS, LOSS_MASKS),
            "pg_clipfrac": _reduce(clipfrac, RESPONSE_LENGTHS, LOSS_MASKS),
            "ppo_kl": _reduce(ppo_kl, RESPONSE_LENGTHS, LOSS_MASKS),
        },
    }

    index = _make_index(bundle_id, config, RAW_REWARDS, LOSS_MASKS)
    actor_step_id = index.identity.actor_step_id
    return CaptureRecord(
        actor_step_id=(actor_step_id.rollout_id, actor_step_id.step_id),
        identity=index.identity,
        samples=index.samples,
        config=config,
        tensors={
            "old_log_probs": old_log_probs.float(),
            "log_probs": log_probs.float(),
            "ref_log_probs": ref_log_probs.float(),
            "entropy": entropy.float(),
            "kl": kl.float(),
            "advantages": advantages.float(),
        },
        expected=expected,
        bundle_id=bundle_id,
        producer=ProducerInfo(commit="test", torch_version=torch.__version__),
        redaction={"prompt": "hash", "response": "truncate:64"},
    )


def make_rollout_capture_record(bundle_id: str = "b-rollout-capture") -> CaptureRecord:
    """Build a pristine rollout-level (reward → advantage) CaptureRecord.

    Mirrors the production capture_hooks.capture_rollout_advantage payload: an
    identity anchored by rollout_id, raw per-sample metadata as deferred
    tensors, and only the reward/advantage stages declared (no loss stage).
    """
    config = RecomputeConfig(
        advantage_estimator="grpo",
        n_samples_per_prompt=2,
        grpo_std_normalization=False,
        kl_loss_type="k1",
        kl_coef=0.1,
        eps_clip=0.2,
        eps_clip_high=0.2,
        entropy_coef=0.01,
    )
    num_tokens = sum(RESPONSE_LENGTHS)
    old_log_probs = torch.zeros(num_tokens)
    ref_log_probs = torch.full((num_tokens,), 0.25)

    normalized_rewards = _group_normalize(
        RAW_REWARDS, GROUP_INDICES, config.n_samples_per_prompt, config.grpo_std_normalization
    )
    kl = compute_approx_kl(old_log_probs, ref_log_probs, kl_loss_type=config.kl_loss_type)
    rewards_tensor = torch.tensor(normalized_rewards, dtype=torch.float32)
    returns = get_grpo_returns(rewards_tensor, list(torch.split(kl, RESPONSE_LENGTHS)))
    advantages = torch.cat(returns)

    identity = Identity(rollout_id=120, rank={"dp": 0, "tp": 0, "pp": 0, "cp": 1})
    flat_masks = [value for sample_mask in LOSS_MASKS for value in sample_mask]
    return CaptureRecord(
        identity=identity,
        samples=[],
        config=config,
        tensors={
            "old_log_probs": old_log_probs.float(),
            "ref_log_probs": ref_log_probs.float(),
            "kl": kl.float(),
            "advantages": advantages.float(),
        },
        expected={},
        bundle_id=bundle_id,
        rollout_id=120,
        producer=ProducerInfo(commit="test", torch_version=torch.__version__),
        stages={
            StageId.REWARD_RAW,
            StageId.REWARD_POST_PROCESS,
            StageId.ADVANTAGE_KL,
            StageId.ADVANTAGE_ESTIMATE,
        },
        response_lengths=list(RESPONSE_LENGTHS),
        total_lengths=list(TOTAL_LENGTHS),
        loss_masks_tensor=torch.tensor(flat_masks, dtype=torch.float32),
        group_indices_tensor=torch.tensor(GROUP_INDICES, dtype=torch.long),
        raw_rewards_tensor=torch.tensor(RAW_REWARDS, dtype=torch.float32),
        rewards_tensor=rewards_tensor,
    )
