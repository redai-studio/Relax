# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay bundle contract.

The on-disk schema for an offline trajectory replay bundle. This module owns
data description only — no production math, no tensors, no I/O beyond
dict round-trips. The bundle is a directory laid out as:

    <bundle>/
    ├── manifest.json    # versions, producer, stage contracts, payload specs
    ├── index.json       # identity + per-sample records + recompute config
    ├── expected.json    # JSON-serializable expected outputs per stage
    ├── payloads/        # tensor payloads (inputs and expected tensors)
    └── COMPLETE         # completion sentinel (or COMPLETE.<rank> per rank)

Only JSON-compatible metadata and weights_only=True tensor payloads are
accepted, so a bundle can be exchanged without executing arbitrary Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# Bundle format version. A reader refuses to open a bundle whose major version
# differs. format_major is derived, never stored.
FORMAT_VERSION = "1.0.0"
FORMAT_MAJOR = FORMAT_VERSION.split(".", maxsplit=1)[0]

# Top-level keys the V1 reader understands for each metadata file. Validation
# uses these to flag unknown fields (which would otherwise be silently dropped)
# instead of letting a newer-but-same-major schema degrade without a warning.
MANIFEST_KEYS = frozenset(
    {"format_version", "bundle_id", "producer", "stage_contracts", "payloads", "comparison_policy", "redaction"}
)
INDEX_KEYS = frozenset({"bundle_id", "identity", "samples", "config"})


class StageCapability(str, Enum):
    """How faithfully a stage can be reproduced offline."""

    RECOMPUTE = "recompute"
    RECORDED_ONLY = "recorded-only"
    INSPECT_ONLY = "inspect-only"
    UNSUPPORTED = "unsupported"


class StageId(str, Enum):
    """Pipeline stages along the reward -> advantage -> loss chain."""

    SAMPLE = "sample"
    REWARD_RAW = "reward.raw"
    REWARD_POST_PROCESS = "reward.post_process"
    ADVANTAGE_KL = "advantage.kl"
    ADVANTAGE_ESTIMATE = "advantage.estimate"
    LOSS_POLICY = "loss.policy"
    LOSS_VALUE = "loss.value"


# Stage DAG order used by the offline runner. A stage may only consume outputs
# produced by stages listed before it.
STAGE_ORDER: tuple[StageId, ...] = (
    StageId.SAMPLE,
    StageId.REWARD_RAW,
    StageId.REWARD_POST_PROCESS,
    StageId.ADVANTAGE_KL,
    StageId.ADVANTAGE_ESTIMATE,
    StageId.LOSS_POLICY,
    StageId.LOSS_VALUE,
)


@dataclass(frozen=True)
class ActorStepId:
    """Training-step coordinate (Task 34 identity anchor).

    This is the (rollout_id, step_id) coordinate of one
    train_one_step call — the stable identity within a run. The derived
    accumulated_step_id scalar is intentionally not the identity: under
    dynamic batching and streaming schedules num_steps_per_rollout can vary
    per rollout, so rollout_id * num_steps_per_rollout + step_id is not a
    monotonic global counter (see relax/backends/megatron/model.py).
    """

    rollout_id: int
    step_id: int


@dataclass(frozen=True)
class WeightLineage:
    """Weight versions of the producers that shaped a sample."""

    rollout_weight: str | None = None
    actor_weight: str | None = None


@dataclass
class Identity:
    """Logical identity + physical provenance of a captured cohort.

    Anchored by exactly one of actor_step_id (per-step, loss) or rollout_id
    (per-rollout, reward/advantage).
    """

    actor_step_id: ActorStepId | None = None
    rollout_id: int | None = None
    rollout_partition_ids: list[str] = field(default_factory=list)
    consumer_batch_ids: list[str] = field(default_factory=list)
    micro_batch_ids: list[str] = field(default_factory=list)
    semantic_group_ids: list[str] = field(default_factory=list)
    normalization_cohort_ids: list[str] = field(default_factory=list)
    weight_lineage: WeightLineage = field(default_factory=WeightLineage)
    # Parallel world sizes {dp, tp, pp, cp}, not the capturing process rank.
    rank: dict[str, int] = field(default_factory=dict)


@dataclass
class SampleRecord:
    """Replay-relevant, JSON-safe slice of one Sample.

    Text (prompt/response/label) is never stored raw; the caller substitutes a
    redaction digest. Per-token tensors live in payloads/, not here.
    """

    sample_id: str
    group_index: int | None
    response_length: int
    total_length: int
    loss_mask: list[int]
    raw_reward: float
    reward: float
    label_hash: str | None = None
    # Physical micro-batch membership (single-batch replay selection). None when
    # the producer did not record it; batch selection then fails closed.
    micro_batch_id: str | None = None


@dataclass
class RecomputeConfig:
    """Numerical configuration required to recompute the frozen V1 path."""

    advantage_estimator: str = "grpo"
    n_samples_per_prompt: int = 1
    grpo_std_normalization: bool = False
    kl_loss_type: str = "k1"
    kl_coef: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float = 0.2
    entropy_coef: float = 0.0


@dataclass
class StageContract:
    """Versioned capability declaration for one pipeline stage."""

    stage: StageId
    version: str
    capability: StageCapability
    # reuse — adapter calls the same production kernel; reimplemented
    # — adapter restates production math and must carry its own parity evidence.
    implementation: str = "reuse"
    inputs: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)


@dataclass
class PayloadSpec:
    """Checksummed description of one tensor payload file."""

    name: str
    dtype: str
    shape: list[int]
    bytes: int
    sha256: str


@dataclass
class ComparisonPolicy:
    """Exact-field and floating-tolerance comparison rules."""

    exact_fields: list[str] = field(default_factory=list)
    tolerances: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class ProducerInfo:
    """Producer identity used to gate replay on a matching build."""

    commit: str = ""
    dirty_patch_digest: str = ""
    torch_version: str = ""


@dataclass
class Manifest:
    """Top-level self-description of a replay bundle."""

    format_version: str
    bundle_id: str
    producer: ProducerInfo
    stage_contracts: dict[StageId, StageContract]
    payloads: dict[str, PayloadSpec]
    comparison_policy: ComparisonPolicy
    redaction: dict[str, str] = field(default_factory=dict)


@dataclass
class BundleIndex:
    """Index payload: identity, samples and recompute config."""

    bundle_id: str
    identity: Identity
    samples: list[SampleRecord]
    config: RecomputeConfig


def _payload_spec_to_dict(spec: PayloadSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "dtype": spec.dtype,
        "shape": spec.shape,
        "bytes": spec.bytes,
        "sha256": spec.sha256,
    }


def _stage_contract_to_dict(contract: StageContract) -> dict[str, Any]:
    return {
        "stage": contract.stage.value,
        "version": contract.version,
        "capability": contract.capability.value,
        "implementation": contract.implementation,
        "inputs": contract.inputs,
        "expected_outputs": contract.expected_outputs,
    }


def _producer_to_dict(producer: ProducerInfo) -> dict[str, Any]:
    return {
        "commit": producer.commit,
        "dirty_patch_digest": producer.dirty_patch_digest,
        "torch_version": producer.torch_version,
    }


def manifest_to_dict(manifest: Manifest) -> dict[str, Any]:
    """Serialize a manifest to a JSON-compatible dict."""
    return {
        "format_version": manifest.format_version,
        "bundle_id": manifest.bundle_id,
        "producer": _producer_to_dict(manifest.producer),
        "stage_contracts": {
            stage.value: _stage_contract_to_dict(contract) for stage, contract in manifest.stage_contracts.items()
        },
        "payloads": {name: _payload_spec_to_dict(spec) for name, spec in manifest.payloads.items()},
        "comparison_policy": {
            "exact_fields": manifest.comparison_policy.exact_fields,
            "tolerances": manifest.comparison_policy.tolerances,
        },
        "redaction": manifest.redaction,
    }


def manifest_from_dict(data: dict[str, Any]) -> Manifest:
    """Deserialize and validate a manifest dict."""
    format_version = data["format_version"]
    if format_version.split(".", maxsplit=1)[0] != FORMAT_MAJOR:
        raise ValueError(f"Unsupported bundle format version {format_version!r} (major != {FORMAT_MAJOR})")

    producer_raw = data.get("producer", {})
    producer = ProducerInfo(
        commit=producer_raw.get("commit", ""),
        dirty_patch_digest=producer_raw.get("dirty_patch_digest", ""),
        torch_version=producer_raw.get("torch_version", ""),
    )

    stage_contracts: dict[StageId, StageContract] = {}
    for stage_name, raw in data.get("stage_contracts", {}).items():
        stage = StageId(stage_name)
        stage_contracts[stage] = StageContract(
            stage=stage,
            version=raw["version"],
            capability=StageCapability(raw["capability"]),
            implementation=raw.get("implementation", "reuse"),
            inputs=raw.get("inputs", []),
            expected_outputs=raw.get("expected_outputs", []),
        )

    payloads = {
        name: PayloadSpec(
            name=raw["name"],
            dtype=raw["dtype"],
            shape=raw["shape"],
            bytes=raw["bytes"],
            sha256=raw["sha256"],
        )
        for name, raw in data.get("payloads", {}).items()
    }

    comparison_raw = data.get("comparison_policy", {})
    comparison = ComparisonPolicy(
        exact_fields=comparison_raw.get("exact_fields", []),
        tolerances=comparison_raw.get("tolerances", {}),
    )

    return Manifest(
        format_version=format_version,
        bundle_id=data["bundle_id"],
        producer=producer,
        stage_contracts=stage_contracts,
        payloads=payloads,
        comparison_policy=comparison,
        redaction=data.get("redaction", {}),
    )


def _sample_record_to_dict(record: SampleRecord) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "group_index": record.group_index,
        "response_length": record.response_length,
        "total_length": record.total_length,
        "loss_mask": record.loss_mask,
        "raw_reward": record.raw_reward,
        "reward": record.reward,
        "label_hash": record.label_hash,
        "micro_batch_id": record.micro_batch_id,
    }


def _sample_record_from_dict(data: dict[str, Any]) -> SampleRecord:
    return SampleRecord(
        sample_id=data["sample_id"],
        group_index=data["group_index"],
        response_length=data["response_length"],
        total_length=data["total_length"],
        loss_mask=data["loss_mask"],
        raw_reward=data["raw_reward"],
        reward=data["reward"],
        label_hash=data.get("label_hash"),
        micro_batch_id=data.get("micro_batch_id"),
    )


def _identity_to_dict(identity: Identity) -> dict[str, Any]:
    actor_step_id = identity.actor_step_id
    return {
        "actor_step_id": (
            {"rollout_id": actor_step_id.rollout_id, "step_id": actor_step_id.step_id}
            if actor_step_id is not None
            else None
        ),
        "rollout_id": identity.rollout_id,
        "rollout_partition_ids": identity.rollout_partition_ids,
        "consumer_batch_ids": identity.consumer_batch_ids,
        "micro_batch_ids": identity.micro_batch_ids,
        "semantic_group_ids": identity.semantic_group_ids,
        "normalization_cohort_ids": identity.normalization_cohort_ids,
        "weight_lineage": {
            "rollout_weight": identity.weight_lineage.rollout_weight,
            "actor_weight": identity.weight_lineage.actor_weight,
        },
        "rank": identity.rank,
    }


def _identity_from_dict(data: dict[str, Any]) -> Identity:
    step_raw = data.get("actor_step_id")
    lineage_raw = data.get("weight_lineage", {})
    return Identity(
        actor_step_id=(
            ActorStepId(rollout_id=step_raw["rollout_id"], step_id=step_raw["step_id"])
            if step_raw is not None
            else None
        ),
        rollout_id=data.get("rollout_id"),
        rollout_partition_ids=data.get("rollout_partition_ids", []),
        consumer_batch_ids=data.get("consumer_batch_ids", []),
        micro_batch_ids=data.get("micro_batch_ids", []),
        semantic_group_ids=data.get("semantic_group_ids", []),
        normalization_cohort_ids=data.get("normalization_cohort_ids", []),
        weight_lineage=WeightLineage(
            rollout_weight=lineage_raw.get("rollout_weight"),
            actor_weight=lineage_raw.get("actor_weight"),
        ),
        rank=data.get("rank", {}),
    )


def index_to_dict(index: BundleIndex) -> dict[str, Any]:
    """Serialize a bundle index to a JSON-compatible dict."""
    return {
        "bundle_id": index.bundle_id,
        "identity": _identity_to_dict(index.identity),
        "samples": [_sample_record_to_dict(sample) for sample in index.samples],
        "config": {
            "advantage_estimator": index.config.advantage_estimator,
            "n_samples_per_prompt": index.config.n_samples_per_prompt,
            "grpo_std_normalization": index.config.grpo_std_normalization,
            "kl_loss_type": index.config.kl_loss_type,
            "kl_coef": index.config.kl_coef,
            "eps_clip": index.config.eps_clip,
            "eps_clip_high": index.config.eps_clip_high,
            "entropy_coef": index.config.entropy_coef,
        },
    }


def index_from_dict(data: dict[str, Any]) -> BundleIndex:
    """Deserialize and validate a bundle index dict."""
    config_raw = data.get("config", {})
    return BundleIndex(
        bundle_id=data["bundle_id"],
        identity=_identity_from_dict(data["identity"]),
        samples=[_sample_record_from_dict(sample) for sample in data["samples"]],
        config=RecomputeConfig(
            advantage_estimator=config_raw.get("advantage_estimator", "grpo"),
            n_samples_per_prompt=config_raw.get("n_samples_per_prompt", 1),
            grpo_std_normalization=config_raw.get("grpo_std_normalization", False),
            kl_loss_type=config_raw.get("kl_loss_type", "k1"),
            kl_coef=config_raw.get("kl_coef", 0.0),
            eps_clip=config_raw.get("eps_clip", 0.2),
            eps_clip_high=config_raw.get("eps_clip_high", 0.2),
            entropy_coef=config_raw.get("entropy_coef", 0.0),
        ),
    )
