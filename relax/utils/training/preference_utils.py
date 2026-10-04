# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure helpers shared by DPO and pairwise reward-model training."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def require_tensor_condition(condition: torch.Tensor, message: str) -> None:
    """Raise on CPU immediately and enqueue a device-side assertion on CUDA."""
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def _validate_same_shape(name: str, *values: torch.Tensor) -> None:
    expected = values[0].shape
    if any(value.shape != expected for value in values[1:]):
        shapes = [tuple(value.shape) for value in values]
        raise ValueError(f"{name} tensors must have identical shapes, got {shapes}")


def masked_sequence_sums(
    values: Sequence[torch.Tensor], masks: Sequence[torch.Tensor], device: torch.device
) -> torch.Tensor:
    """Sum each branch against its already aligned mask, preserving gradients
    and dtype."""
    if len(values) != len(masks):
        raise ValueError("preference token values/masks are not branch aligned")
    sums = []
    for value, mask in zip(values, masks, strict=True):
        value = torch.as_tensor(value, device=device)
        mask = torch.as_tensor(mask, device=device, dtype=value.dtype)
        if value.shape != mask.shape:
            raise ValueError(
                f"preference branch value/mask shape mismatch: {tuple(value.shape)} vs {tuple(mask.shape)}"
            )
        require_tensor_condition(
            mask.to(dtype=torch.bool).any(),
            "preference branch completion mask must contain at least one supervised token",
        )
        sums.append((value * mask).sum())
    return torch.stack(sums)


def dpo_rewards(
    policy_log_probs: torch.Tensor, reference_log_probs: torch.Tensor | None, *, beta: float
) -> torch.Tensor:
    """Return implicit rewards; a missing reference selects reference-free
    DPO."""
    log_ratios = policy_log_probs if reference_log_probs is None else policy_log_probs - reference_log_probs
    return beta * log_ratios


def preference_accuracy(
    margins: torch.Tensor, *, epsilon: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return strict wins, ties, and tie-aware accuracy for each pair."""
    strict = margins > 0
    ties = margins.abs() <= epsilon
    tie_aware = (margins > epsilon).to(torch.float32) + 0.5 * ties.to(torch.float32)
    return strict, ties, tie_aware


def dpo_pair_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    *,
    reference_chosen: torch.Tensor | None = None,
    reference_rejected: torch.Tensor | None = None,
    beta: float = 0.1,
    reference_free: bool = False,
) -> torch.Tensor:
    """Return unreduced sigmoid-DPO loss, one value per preference pair."""
    if beta <= 0:
        raise ValueError(f"DPO beta must be positive, got {beta}")
    _validate_same_shape("policy log-probabilities", policy_chosen, policy_rejected)
    policy_logratio = policy_chosen - policy_rejected
    if reference_free:
        if reference_chosen is not None or reference_rejected is not None:
            raise ValueError("reference-free DPO must not receive reference log-probabilities")
        reference_logratio = torch.zeros_like(policy_logratio)
    else:
        if reference_chosen is None or reference_rejected is None:
            raise ValueError("standard DPO requires chosen and rejected reference log-probabilities")
        _validate_same_shape(
            "reference log-probabilities",
            policy_chosen,
            reference_chosen,
            reference_rejected,
        )
        reference_logratio = reference_chosen - reference_rejected
    logits = beta * (policy_logratio - reference_logratio)
    require_tensor_condition(torch.isfinite(logits).all(), "DPO logits must contain only finite values")
    return -F.logsigmoid(logits)


def build_preference_pair_indices(
    branch_pair_ids: Sequence[int], branch_is_chosen: Sequence[bool]
) -> tuple[list[int], list[int]]:
    """Validate adjacent chosen/rejected pairs, preserving repeated samples."""
    if len(branch_pair_ids) != len(branch_is_chosen):
        raise ValueError(
            "preference pair identity fields must be branch aligned: "
            f"{len(branch_pair_ids)} vs {len(branch_is_chosen)}"
        )
    if not branch_pair_ids:
        raise ValueError("DPO micro-batch must contain at least one preference pair")

    if len(branch_pair_ids) % 2:
        raise ValueError("preference micro-batch must contain an even number of branches")
    for index in range(0, len(branch_pair_ids), 2):
        if not branch_is_chosen[index] or branch_is_chosen[index + 1]:
            raise ValueError(f"preference branches at positions {index}/{index + 1} must be ordered chosen/rejected")
        if int(branch_pair_ids[index]) != int(branch_pair_ids[index + 1]):
            raise ValueError(
                f"adjacent preference branches at positions {index}/{index + 1} must have the same pair ID"
            )
    return list(range(0, len(branch_pair_ids), 2)), list(range(1, len(branch_pair_ids), 2))


def reward_model_pair_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    """Return unreduced Bradley-Terry loss, one value per preference pair."""
    _validate_same_shape("reward-model scores", chosen_scores, rejected_scores)
    margins = chosen_scores - rejected_scores
    require_tensor_condition(
        torch.isfinite(margins).all(),
        "reward-model margins must contain only finite values",
    )
    return -F.logsigmoid(margins)


def select_packed_sequence_scores(
    logits: torch.Tensor,
    total_lengths: Sequence[int],
) -> torch.Tensor:
    """Select each CP=1 THD branch's last-token score, excluding tail
    padding."""
    if logits.ndim == 3 and logits.shape[0] == 1 and logits.shape[-1] == 1:
        flat_logits = logits[0, :, 0]
    elif logits.ndim == 2 and logits.shape[-1] == 1:
        flat_logits = logits[:, 0]
    elif logits.ndim == 1:
        flat_logits = logits
    else:
        raise ValueError(f"reward-model logits must have shape [1,T,1], [T,1], or [T], got {tuple(logits.shape)}")

    offsets: list[int] = []
    cursor = 0
    for index, length in enumerate(total_lengths):
        length = int(length)
        if length <= 0:
            raise ValueError(f"sequence {index} has non-positive total length {length}")
        cursor += length
        offsets.append(cursor - 1)
    if cursor > flat_logits.numel():
        raise ValueError(f"packed reward logits contain {flat_logits.numel()} tokens, expected at least {cursor}")
    if not offsets:
        return flat_logits.new_empty((0,))
    return flat_logits[torch.tensor(offsets, device=flat_logits.device, dtype=torch.long)]


def pack_preference_pair_indices(
    costs: Sequence[int],
    pair_ids: Sequence[int],
    *,
    capacity: int,
) -> list[list[int]]:
    """Pack pairs by decreasing cost, keeping oversize pairs in separate
    bins."""
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    if len(costs) != len(pair_ids):
        raise ValueError(f"costs/pair_ids length mismatch: {len(costs)} vs {len(pair_ids)}")
    normalized_costs = [int(cost) for cost in costs]
    for pair_id, cost in zip(pair_ids, normalized_costs, strict=True):
        if cost <= 0:
            raise ValueError(f"pair {pair_id!r} has non-positive cost {cost}")

    order = sorted(range(len(normalized_costs)), key=lambda index: (-normalized_costs[index], pair_ids[index]))
    bins: list[list[int]] = []
    bin_costs: list[int] = []
    for index in order:
        cost = normalized_costs[index]
        for bin_index, bin_cost in enumerate(bin_costs):
            if bin_cost + cost <= capacity:
                bins[bin_index].append(index)
                bin_costs[bin_index] += cost
                break
        else:
            bins.append([index])
            bin_costs.append(cost)

    return bins


__all__ = [
    "build_preference_pair_indices",
    "dpo_pair_loss",
    "dpo_rewards",
    "masked_sequence_sums",
    "pack_preference_pair_indices",
    "preference_accuracy",
    "require_tensor_condition",
    "reward_model_pair_loss",
    "select_packed_sequence_scores",
]
