# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Forward-only callbacks and reducers for held-out preference evaluation."""

import torch

from relax.utils.training.preference_utils import select_packed_sequence_scores


def preference_eval_chunk_sizes(pair_count: int, global_batch_size: int) -> list[int]:
    """Split every evaluation pair into capacity-bounded eval chunks."""
    if pair_count <= 0 or global_batch_size <= 0:
        raise ValueError("preference eval pair count and global batch size must be positive")
    full_chunks, remainder = divmod(pair_count, global_batch_size)
    return [global_batch_size] * full_chunks + ([remainder] if remainder else [])


def preference_eval_local_batch_sizes(pair_count: int, global_batch_size: int, dp_size: int) -> list[int]:
    if dp_size <= 0:
        raise ValueError("preference eval data-parallel size must be positive")
    chunk_sizes = preference_eval_chunk_sizes(pair_count, global_batch_size)
    invalid = [size for size in chunk_sizes if size % dp_size != 0]
    if invalid:
        raise ValueError(
            f"preference eval chunk sizes must be divisible by data-parallel size: chunks={invalid}, dp={dp_size}"
        )
    return [size // dp_size for size in chunk_sizes]


def compute_reward_model_eval_step(
    logits: torch.Tensor,
    *,
    total_lengths,
    score_positions=None,
    raw_loss_masks=None,
    packed_tokens=None,
    branch_tokens=None,
    cu_seqlens=None,
    **_,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    if score_positions is None:
        raise ValueError("reward-model eval batch is missing score_positions")
    scores = select_packed_sequence_scores(
        logits,
        total_lengths,
        score_positions,
        raw_loss_masks=raw_loss_masks,
        packed_tokens=packed_tokens,
        branch_tokens=branch_tokens,
        cu_seqlens=cu_seqlens,
    ).detach()
    # Emit one scalar per branch so ``forward_only`` can restore original
    # sample order after dynamic micro-batch length balancing.
    return torch.empty((0,), device=logits.device), {"scores": list(scores.unbind())}


def pair_metric_sums(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    losses: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return additive loss/count/score/margin/correct/tie statistics."""
    if chosen.shape != rejected.shape or chosen.shape != losses.shape:
        raise ValueError("preference eval tensors must have identical shapes")
    margin = chosen - rejected
    correct = (margin > epsilon).to(torch.float64).sum()
    ties = (margin.abs() <= epsilon).to(torch.float64).sum()
    return torch.stack(
        [
            losses.to(torch.float64).sum(),
            torch.tensor(float(losses.numel()), device=losses.device, dtype=torch.float64),
            chosen.to(torch.float64).sum(),
            rejected.to(torch.float64).sum(),
            margin.to(torch.float64).sum(),
            correct,
            ties,
        ]
    )


def finalize_pair_metrics(values: torch.Tensor, *, prefix: str) -> dict[str, float]:
    loss_sum, count, chosen_sum, rejected_sum, margin_sum, correct, ties = values.tolist()
    if count <= 0:
        raise ValueError("preference evaluator received zero pairs")
    return {
        f"eval/{prefix}_loss": loss_sum / count,
        f"eval/{prefix}_chosen": chosen_sum / count,
        f"eval/{prefix}_rejected": rejected_sum / count,
        f"eval/{prefix}_margin": margin_sum / count,
        f"eval/{prefix}_strict_accuracy": correct / count,
        f"eval/{prefix}_tie_rate": ties / count,
        f"eval/{prefix}_tie_aware_accuracy": (correct + 0.5 * ties) / count,
        f"eval/{prefix}_pairs": count,
    }


__all__ = [
    "compute_reward_model_eval_step",
    "finalize_pair_metrics",
    "pair_metric_sums",
    "preference_eval_chunk_sizes",
    "preference_eval_local_batch_sizes",
]
