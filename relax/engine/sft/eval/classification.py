# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Sequence-classification SFT evaluation callbacks and metric aggregation."""

import torch
import torch.nn.functional as F


def compute_classification_eval_step(
    logits: torch.Tensor,
    *,
    args,
    unconcat_tokens,
    total_lengths,
    response_lengths,
    classification_labels=None,
    sample_weights=None,
    max_seq_lens=None,
    padded_total_lengths=None,
    dynamic_cp_size=None,
    dynamic_cp_rank=None,
    **_,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Return CP-local sufficient statistics for one classification
    microbatch."""
    from relax.backends.megatron.loss import get_sequence_classification_outputs

    if classification_labels is None:
        raise ValueError("classification eval batch is missing classification_labels")
    if sample_weights is None:
        raise ValueError("classification eval batch is missing sample_weights")

    batch = {
        "unconcat_tokens": unconcat_tokens,
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
        "max_seq_lens": max_seq_lens,
        "padded_total_lengths": padded_total_lengths,
        "dynamic_cp_size": dynamic_cp_size,
        "dynamic_cp_rank": dynamic_cp_rank,
    }
    local_logits, local_indices = get_sequence_classification_outputs(args, batch, logits)
    device = logits.device

    if not local_indices:
        zero = torch.zeros(1, device=device, dtype=torch.float64)
        keys = (
            ("loss_sum", "num_examples", "correct")
            if args.problem_type == "single_label_classification"
            else ("loss_sum", "num_examples", "tp", "fp", "fn", "exact_match")
        )
        return torch.empty((0,), device=device), {key: [zero.clone()] for key in keys}

    weights = torch.stack([torch.as_tensor(sample_weights[i]).reshape(()) for i in local_indices]).to(
        device=device, dtype=torch.float32
    )
    local_labels = [classification_labels[i] for i in local_indices]

    if args.problem_type == "single_label_classification":
        labels = torch.stack([torch.as_tensor(label).reshape(()) for label in local_labels]).to(
            device=device, dtype=torch.long
        )
        per_sample_loss = F.cross_entropy(local_logits, labels, reduction="none")
        correct = (local_logits.argmax(dim=-1) == labels).float()
        stats = {
            "loss_sum": (per_sample_loss * weights).sum(),
            "num_examples": weights.sum(),
            "correct": (correct * weights).sum(),
        }
    else:
        labels = torch.stack([torch.as_tensor(label).reshape(-1) for label in local_labels]).to(
            device=device, dtype=torch.float32
        )
        if labels.shape != local_logits.shape:
            raise ValueError(
                f"multi-label eval targets must have shape {tuple(local_logits.shape)}, got {tuple(labels.shape)}"
            )
        per_sample_loss = F.binary_cross_entropy_with_logits(local_logits, labels, reduction="none").mean(dim=-1)
        predictions = torch.sigmoid(local_logits) >= args.classification_threshold
        targets = labels.bool()
        expanded_weights = weights.unsqueeze(-1)
        stats = {
            "loss_sum": (per_sample_loss * weights).sum(),
            "num_examples": weights.sum(),
            "tp": ((predictions & targets).float() * expanded_weights).sum(),
            "fp": ((predictions & ~targets).float() * expanded_weights).sum(),
            "fn": ((~predictions & targets).float() * expanded_weights).sum(),
            "exact_match": (predictions.eq(targets).all(dim=-1).float() * weights).sum(),
        }

    return torch.empty((0,), device=device), {
        key: [value.detach().reshape(1).to(torch.float64)] for key, value in stats.items()
    }


def compute_classification_metrics(problem_type: str, totals: dict[str, float]) -> dict[str, float]:
    """Convert globally reduced sufficient statistics into user metrics."""
    num_examples = totals["num_examples"]
    loss = totals["loss_sum"] / num_examples if num_examples > 0 else 0.0
    if problem_type == "single_label_classification":
        accuracy = totals["correct"] / num_examples if num_examples > 0 else 0.0
        return {
            "eval/loss": loss,
            "eval/accuracy": accuracy,
            "eval/num_examples": num_examples,
        }

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    subset_accuracy = totals["exact_match"] / num_examples if num_examples > 0 else 0.0
    return {
        "eval/loss": loss,
        "eval/micro_precision": precision,
        "eval/micro_recall": recall,
        "eval/micro_f1": f1,
        "eval/subset_accuracy": subset_accuracy,
        "eval/num_examples": num_examples,
    }
