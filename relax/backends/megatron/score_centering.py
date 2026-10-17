# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Megatron scoring and batch alignment for Score Centering."""

from argparse import Namespace
from typing import Any

import torch
from megatron.core import mpu

from relax.utils.opd.opd_utils import compute_log_probs_on_topk_token_ids
from relax.utils.training.ppo_utils import calculate_log_probs_and_entropy

from .cp_utils import all_gather_with_cp, slice_log_prob_with_cp


def _mask_vocab_logits(logits: torch.Tensor, args: Namespace) -> torch.Tensor:
    """Exclude TP padding from the distribution used by SC and its
    regularizers."""
    vocab_size = getattr(args, "score_centering_vocab_size", None)
    if vocab_size is None or vocab_size <= 0:
        raise ValueError("Score centering requires the model's unpadded vocabulary size.")
    tp_rank = mpu.get_tensor_model_parallel_rank()
    vocab_ids = torch.arange(logits.size(-1), device=logits.device) + tp_rank * logits.size(-1)
    # A finite floor has zero softmax mass in fp32 and keeps the existing entropy
    # kernel's p * logits finite on padded entries (0 * -inf would be NaN).
    return logits.masked_fill(vocab_ids >= vocab_size, torch.finfo(logits.dtype).min)


def align_score_centering_candidate_ids(
    token_ids: list[torch.Tensor] | None,
    *,
    logits: torch.Tensor,
    args: Namespace,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None,
    padded_total_lengths: list[int] | None,
    dynamic_cp_size: int | None,
    dynamic_cp_rank: int | None,
) -> list[torch.Tensor] | None:
    """Match ring-layout candidate rows to the contiguous allgather-CP
    logits."""
    cp_size = dynamic_cp_size if dynamic_cp_size is not None else mpu.get_context_parallel_world_size()
    if token_ids is None or not args.allgather_cp or cp_size == 1:
        return token_ids
    cp_rank = dynamic_cp_rank if dynamic_cp_rank is not None else mpu.get_context_parallel_rank()
    cp_group = (
        mpu.get_dynamic_data_context_parallel_groups(group_size=dynamic_cp_size)
        if dynamic_cp_size is not None
        else None
    )
    chunk_start = cp_rank * logits.size(1)
    chunk_end = chunk_start + logits.size(1)
    aligned_ids: list[torch.Tensor] = []
    seq_start = 0
    for i, (ids, total_length, response_length) in enumerate(
        zip(token_ids, total_lengths, response_lengths, strict=True)
    ):
        full_ids = all_gather_with_cp(
            ids,
            total_length,
            response_length,
            padded_total_length=padded_total_lengths[i] if padded_total_lengths is not None else None,
            qkv_format=args.qkv_format,
            max_seq_len=max_seq_lens[i] if max_seq_lens is not None else None,
            dynamic_cp_size=dynamic_cp_size,
            dynamic_cp_rank=dynamic_cp_rank,
            dynamic_cp_group=cp_group,
        )
        response_start = seq_start + total_length - response_length - 1
        start = max(response_start, chunk_start)
        end = min(seq_start + total_length - 1, chunk_end)
        aligned_ids.append(full_ids[start - response_start : end - response_start] if end > start else full_ids[:0])
        seq_start += total_length
    return aligned_ids


def get_score_centering_chunk(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    process_group: Any,
    *,
    args: Namespace,
    candidate_ids: torch.Tensor | None,
    with_entropy: bool,
    with_topk: bool,
    topk_k: int,
) -> dict[str, torch.Tensor | None]:
    """Score one response chunk after the shared slicer applies temperature.

    Sampled tokens, recorded candidates, and entropy use the same real
    vocabulary. Empty CP ranks retain the graph for collective backward.
    """
    logits = _mask_vocab_logits(logits, args)
    with_entropy_grad = with_entropy and getattr(args, "entropy_coef", 0.0) != 0
    log_probs, entropy = calculate_log_probs_and_entropy(
        logits,
        tokens,
        process_group,
        with_entropy=with_entropy,
        with_entropy_grad=with_entropy_grad,
        chunk_size=args.log_probs_chunk_size,
    )
    log_probs = log_probs.squeeze(-1)
    if logits.size(0) == 0:
        log_probs = log_probs + 0 * logits.sum()
        if with_entropy_grad:
            entropy = entropy + 0 * logits.sum()

    result = {"log_probs": log_probs, "entropy": entropy}
    if with_topk:
        k = min(max(int(topk_k), 1), int(logits.size(-1)))
        result["topk_token_ids"] = torch.topk(logits, k=k, dim=-1).indices
    if candidate_ids is not None:
        selected = compute_log_probs_on_topk_token_ids(logits, candidate_ids, process_group)
        if logits.size(0) == 0:
            selected = selected + 0 * logits.sum()
        result["topk_log_probs"] = selected
    return result


def build_score_centering_context(
    args: Namespace,
    batch: dict[str, Any],
    log_probs_and_entropy: dict[str, list[torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Align full-response masks with the CP-local sampled/head
    probabilities."""
    max_seq_lens = batch.get("max_seq_lens")
    padded_total_lengths = batch.get("padded_total_lengths")
    local_masks = [
        slice_log_prob_with_cp(
            mask,
            total_length,
            response_length,
            args.qkv_format,
            max_seq_lens[i] if max_seq_lens is not None else None,
            padded_total_lengths[i] if padded_total_lengths is not None else None,
            dynamic_cp_size=batch.get("dynamic_cp_size"),
            dynamic_cp_rank=batch.get("dynamic_cp_rank"),
        )
        for i, (mask, total_length, response_length) in enumerate(
            zip(batch["loss_masks"], batch["total_lengths"], batch["response_lengths"], strict=True)
        )
    ]
    train_topk_log_probs = torch.cat(log_probs_and_entropy["topk_log_probs"])
    return {
        "rollout_log_probs": torch.cat(batch["rollout_log_probs"]),
        "rollout_topk_log_probs": torch.cat(batch["rollout_topk_log_probs"]),
        "train_topk_log_probs": train_topk_log_probs,
        "loss_mask": torch.cat(local_masks).to(device=train_topk_log_probs.device),
    }
