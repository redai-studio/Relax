# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Train-side capture hooks. Import only from loss.py / model.py / actor.py.

Lazily imports megatron.core.mpu. Offline replay modules must not import this
file. Disabled path: two global reads, then return.
"""

from __future__ import annotations

from typing import Any

import torch

from relax.utils.replay import capture
from relax.utils.replay.schema import ActorStepId, Identity, RecomputeConfig, StageId


def build_config_from_args(args: Any) -> RecomputeConfig:
    """Build a RecomputeConfig from the runtime args namespace."""
    return RecomputeConfig(
        advantage_estimator=str(getattr(args, "advantage_estimator", "grpo")),
        n_samples_per_prompt=int(getattr(args, "n_samples_per_prompt", 1)),
        grpo_std_normalization=bool(getattr(args, "grpo_std_normalization", False)),
        kl_loss_type=str(getattr(args, "kl_loss_type", "k1")),
        kl_coef=float(getattr(args, "kl_coef", 0.0)),
        eps_clip=float(getattr(args, "eps_clip", 0.2)),
        eps_clip_high=float(getattr(args, "eps_clip_high", 0.2)),
        entropy_coef=float(getattr(args, "entropy_coef", 0.0)),
    )


def _parallel_sizes() -> dict[str, int]:
    """Megatron parallel world sizes (not the capturing process rank)."""
    from megatron.core import mpu  # lazy: hot-path / megatron only

    return {
        "dp": int(mpu.get_data_parallel_world_size()),
        "tp": int(mpu.get_tensor_model_parallel_world_size()),
        "pp": int(mpu.get_pipeline_model_parallel_world_size()),
        "cp": int(mpu.get_context_parallel_world_size()),
    }


def build_identity(rollout_id: int, step_id: int | None = None) -> Identity:
    """Identity for a step (step_id set) or a rollout (step_id omitted)."""
    rank = _parallel_sizes()
    if step_id is None:
        return Identity(rollout_id=int(rollout_id), rank=rank)
    return Identity(actor_step_id=ActorStepId(rollout_id=int(rollout_id), step_id=int(step_id)), rank=rank)


def capture_producer_ranks() -> list[int]:
    """Global ranks on the last pipeline stage (they own post-forward
    payloads).

    Init-time all-gather; not on the training hot path. Independent of rank
    order.
    """
    import torch.distributed as dist
    from megatron.core import mpu

    from relax.utils.distributed_utils import get_gloo_group

    flag = bool(mpu.is_pipeline_last_stage(ignore_virtual=True))
    world = dist.get_world_size()
    # Last-PP flags are CPU metadata. NCCL all_gather_object copies onto CUDA
    # and fails after offload_train (cudaErrorInvalidValue).
    if world == 1:
        return [0] if flag else []
    gathered: list[object] = [None] * world
    dist.all_gather_object(gathered, flag, group=get_gloo_group())
    return [index for index, item in enumerate(gathered) if item]


def maybe_enable_for_actor() -> capture.CaptureManager | None:
    """Enable capture on last-PP producer ranks only.

    Every actor rank must call this when capture env vars are set (all-gather
    is collective). Non-producers return without starting a writer thread.
    """
    if capture.config_from_env() is None:
        return None
    import torch.distributed as dist

    expected = capture_producer_ranks()
    rank = dist.get_rank()
    if rank not in expected:
        return None
    return capture.maybe_enable_from_env(rank=rank, expected_ranks=expected)


def begin_step_for(args: Any, rollout_id: int, step_id: int, bundle_id: str | None = None) -> None:
    """Open a per-step accumulator from train_one_step."""
    actor_step_id = (int(rollout_id), int(step_id))
    if not capture.should_capture(actor_step_id):
        return
    capture.begin_step(
        actor_step_id,
        identity=build_identity(rollout_id, step_id),
        config=build_config_from_args(args),
        bundle_id=bundle_id if bundle_id is not None else f"replay-{rollout_id}-{step_id}",
    )


def end_step_for() -> None:
    capture.end_step()


def begin_rollout_for(args: Any, rollout_id: int, bundle_id: str | None = None) -> None:
    """Open a per-rollout accumulator from train_actor."""
    if not capture.should_capture_rollout(int(rollout_id)):
        return
    capture.begin_rollout(
        int(rollout_id),
        identity=build_identity(int(rollout_id)),
        config=build_config_from_args(args),
        bundle_id=bundle_id if bundle_id is not None else f"replay-rollout-{rollout_id}",
    )


def end_rollout_for() -> None:
    capture.end_rollout()


def _cat(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.detach().reshape(-1) for tensor in tensors])


def _detach_1d(value: Any, dtype: torch.dtype) -> torch.Tensor:
    """Detached 1-D tensor; no GPU→CPU sync.

    Lists are already on CPU.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().reshape(-1).to(dtype)
    return torch.tensor(list(value), dtype=dtype)


def _group_indices_from_rollout(rollout_data: Any, args: Any) -> Any:
    """Prefer TransferQueue ``group_index``; else consecutive GRPO groups."""
    if "group_index" in rollout_data:
        return rollout_data["group_index"]
    n_samples = len(rollout_data["response_lengths"])
    n = int(getattr(args, "n_samples_per_prompt", 1) or 1)
    return [index // n for index in range(n_samples)]


def capture_rollout_advantage(*, rollout_data: Any, args: Any) -> None:
    """Record the reward → advantage chain from train_actor."""
    step = capture.current_step()
    if step is None:
        return

    # Non-last PP returns from compute_advantages_and_returns before advantages.
    if rollout_data.get("advantages") is None:
        return

    from relax.utils.training.ppo_utils import compute_approx_kl  # train-side only

    use_rollout = bool(getattr(args, "use_rollout_logprobs", False))
    kl_loss_type = str(getattr(args, "kl_loss_type", "k1"))
    kl_coef = float(getattr(args, "kl_coef", 0.0))

    step.stages = {
        StageId.REWARD_RAW,
        StageId.REWARD_POST_PROCESS,
        StageId.ADVANTAGE_KL,
        StageId.ADVANTAGE_ESTIMATE,
    }

    step.response_lengths = [int(length) for length in rollout_data["response_lengths"]]
    step.total_lengths = [int(length) for length in rollout_data["total_lengths"]]
    if rollout_data.get("loss_masks"):
        step.loss_masks_tensor = _cat(list(rollout_data["loss_masks"]))
    step.group_indices_tensor = _detach_1d(_group_indices_from_rollout(rollout_data, args), torch.long)
    step.raw_rewards_tensor = _detach_1d(rollout_data["raw_reward"], torch.float32)
    step.rewards_tensor = _detach_1d(rollout_data["rewards"], torch.float32)

    log_probs_key = "rollout_log_probs" if use_rollout else "log_probs"
    old_log_probs = list(rollout_data[log_probs_key])
    ref_log_probs = list(rollout_data["ref_log_probs"]) if rollout_data.get("ref_log_probs") else None

    step.tensors["old_log_probs"] = _cat(old_log_probs)
    step.tensors["advantages"] = _cat(list(rollout_data["advantages"]))

    if kl_coef == 0 or ref_log_probs is None:
        ref_log_probs = [torch.zeros_like(probs, dtype=torch.float32) for probs in old_log_probs]
        kl = [torch.zeros_like(probs, dtype=torch.float32) for probs in old_log_probs]
    else:
        kl = [
            compute_approx_kl(old_log_probs[index], ref_log_probs[index], kl_loss_type=kl_loss_type)
            for index in range(len(old_log_probs))
        ]
    step.tensors["ref_log_probs"] = _cat(ref_log_probs)
    step.tensors["kl"] = _cat(kl)


def capture_policy_loss(
    *,
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    entropy: torch.Tensor,
    advantages: torch.Tensor,
    loss_masks: list[torch.Tensor],
    response_lengths: list[int],
    total_lengths: list[int],
    reported_loss: dict[str, torch.Tensor],
    micro_batch_index: int | None = None,
) -> None:
    """Record loss.policy from policy_loss_function (detach only).

    Each DataIterator micro-batch is appended. A repeated micro_batch_index
    (activation-checkpoint recompute) is ignored so the first writer wins.
    """
    step = capture.current_step()
    if step is None:
        return

    mb_id = capture.format_micro_batch_id(0 if micro_batch_index is None else int(micro_batch_index))
    if mb_id in step.captured_micro_batch_ids:
        return
    step.captured_micro_batch_ids.add(mb_id)

    step.stages = {StageId.LOSS_POLICY}
    _append_flat(step, "old_log_probs", old_log_probs)
    _append_flat(step, "log_probs", log_probs)
    _append_flat(step, "entropy", entropy)
    _append_flat(step, "advantages", advantages)

    lengths = [int(length) for length in response_lengths]
    step.response_lengths = (step.response_lengths or []) + lengths
    step.total_lengths = (step.total_lengths or []) + [int(length) for length in total_lengths]
    step.sample_micro_batch_ids = (step.sample_micro_batch_ids or []) + [mb_id] * len(lengths)

    if loss_masks:
        masks = torch.cat([mask.detach().reshape(-1) for mask in loss_masks])
        step.loss_masks_tensor = (
            masks if step.loss_masks_tensor is None else torch.cat([step.loss_masks_tensor, masks])
        )

    policy_fields = ("loss", "pg_loss", "entropy_loss", "pg_clipfrac", "ppo_kl")
    expected = step.expected.setdefault(StageId.LOSS_POLICY.value, {})
    for field, value in reported_loss.items():
        if field not in policy_fields:
            continue
        detached = value.detach()
        expected[field] = detached if field not in expected else expected[field] + detached


def _append_flat(step: capture.CaptureRecord, name: str, tensor: torch.Tensor) -> None:
    detached = tensor.detach().reshape(-1)
    existing = step.tensors.get(name)
    step.tensors[name] = detached if existing is None else torch.cat([existing, detached])
