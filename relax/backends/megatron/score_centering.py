# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Megatron scoring and batch alignment for Score Centering."""

from argparse import Namespace
from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu

from .cp_utils import all_gather_with_cp, slice_log_prob_with_cp


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


def _iter_response_chunks(spans: list[tuple[int, int]], chunk_size: int) -> Iterator[tuple[slice, slice]]:
    """Map blocks of logits rows to consecutive output rows."""
    offset = 0
    for start, stop in spans:
        for first in range(start, stop, chunk_size):
            last = min(first + chunk_size, stop)
            next_offset = offset + last - first
            yield slice(first, last), slice(offset, next_offset)
            offset = next_offset


def _scaled_logits(logits: torch.Tensor, valid_width: int, temperature: float) -> torch.Tensor:
    work = logits.to(dtype=torch.float32, copy=True)
    if temperature != 1.0:
        work.div_(temperature)
    if valid_width < work.size(-1):
        work[:, valid_width:] = torch.finfo(work.dtype).min
    return work


class _ScoreCenteringLogProbs(torch.autograd.Function):
    """Score response rows in bounded blocks and return one logits gradient.

    Sampled tokens, recorded candidates and entropy share the same vocabulary
    normalization. Backward recomputes each block from saved row statistics.
    """

    @staticmethod
    def forward(
        ctx,
        logits,
        ids,
        spans,
        process_group,
        chunk_size,
        vocab_size,
        temperature,
        with_entropy,
        with_entropy_grad,
        topk_k,
    ):
        rows = ids.size(0)
        width = logits.size(-1)
        vocab_start = mpu.get_tensor_model_parallel_rank() * width
        valid_width = min(max(vocab_size - vocab_start, 0), width)

        values = logits.new_empty(ids.shape, dtype=torch.float32)
        maxima = values.new_empty((rows, 1))
        denominators = torch.empty_like(maxima)
        entropy = values.new_empty(rows) if with_entropy else None
        means = torch.empty_like(maxima) if with_entropy_grad else None
        topk = ids.new_empty((rows, topk_k)) if topk_k else None
        for logit_rows, result_rows in _iter_response_chunks(spans, chunk_size):
            work = _scaled_logits(logits[logit_rows], valid_width, temperature)
            if topk is not None:
                topk[result_rows] = work.topk(topk_k, dim=-1).indices
            maximum = work.max(dim=-1, keepdim=True).values
            if process_group is not None:
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=process_group)

            selected = ids[result_rows]
            local = (selected >= vocab_start) & (selected < vocab_start + valid_width)
            local_ids = (selected - vocab_start).clamp(0, width - 1)
            targets = work.gather(-1, local_ids).masked_fill_(~local, 0)
            if process_group is not None:
                dist.all_reduce(targets, group=process_group)

            work.sub_(maximum).exp_()
            denominator = work.sum(dim=-1, keepdim=True)
            if process_group is not None:
                dist.all_reduce(denominator, group=process_group)
            log_denominator = denominator.log()
            invalid = (selected < 0) | (selected >= vocab_size)
            values[result_rows] = (targets - maximum - log_denominator).masked_fill_(invalid, -torch.inf)

            if entropy is not None:
                # For u = exp(logits - max), entropy is log(sum(u)) - sum(u log(u)) / sum(u).
                mean = work.xlogy_(work).sum(dim=-1, keepdim=True)
                if process_group is not None:
                    dist.all_reduce(mean, group=process_group)
                mean.div_(denominator)
                entropy[result_rows] = (log_denominator - mean).squeeze(-1)
                if means is not None:
                    means[result_rows] = mean
            maxima[result_rows] = maximum
            denominators[result_rows] = denominator
            del work

        ctx.save_for_backward(logits, ids, maxima, denominators, means)
        ctx.spans = spans
        ctx.chunk_size = chunk_size
        ctx.vocab_start = vocab_start
        ctx.valid_width = valid_width
        ctx.vocab_size = vocab_size
        ctx.temperature = temperature
        if entropy is not None and not with_entropy_grad:
            ctx.mark_non_differentiable(entropy)
        return values, entropy, topk

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_values, grad_entropy, _grad_topk):
        logits, ids, maxima, denominators, means = ctx.saved_tensors
        width = logits.size(-1)
        grad_logits = torch.zeros_like(logits)
        for logit_rows, result_rows in _iter_response_chunks(ctx.spans, ctx.chunk_size):
            work = _scaled_logits(logits[logit_rows], ctx.valid_width, ctx.temperature)
            work.sub_(maxima[result_rows])
            selected = ids[result_rows]
            invalid = (selected < 0) | (selected >= ctx.vocab_size)
            grad = grad_values[result_rows].masked_fill(invalid, 0)
            scale = -grad.sum(dim=-1, keepdim=True)
            if means is not None:
                scale = (means[result_rows] - work).mul_(grad_entropy[result_rows, None]).add_(scale)

            work.exp_().div_(denominators[result_rows]).mul_(scale)
            local = (selected >= ctx.vocab_start) & (selected < ctx.vocab_start + ctx.valid_width)
            local_ids = (selected - ctx.vocab_start).clamp(0, width - 1)
            work.scatter_add_(-1, local_ids, grad.masked_fill(~local, 0))
            if ctx.valid_width < width:
                work[:, ctx.valid_width :] = 0
            if ctx.temperature != 1.0:
                work.div_(ctx.temperature)
            grad_logits[logit_rows] = work
            del work, scale
        # The loss is replicated over TP. Each rank already has its local derivative.
        return grad_logits, None, None, None, None, None, None, None, None, None


def get_score_centering_batch(
    logits: torch.Tensor,
    responses: list[tuple[torch.Tensor, torch.Tensor]],
    process_group: Any,
    *,
    args: Namespace,
    candidate_ids: list[torch.Tensor] | None,
    with_entropy: bool,
    with_topk: bool,
    topk_k: int,
) -> dict[str, list[torch.Tensor]]:
    """Score response views from one microbatch without copying their
    logits."""
    if not responses:
        fields = {"log_probs": []}
        if with_entropy:
            fields["entropy"] = []
        if with_topk:
            fields["topk_token_ids"] = []
        if candidate_ids is not None:
            fields["topk_log_probs"] = []
        return fields

    vocab_size = getattr(args, "score_centering_vocab_size", None)
    if vocab_size is None or vocab_size <= 0:
        raise ValueError("Score centering requires the model's unpadded vocabulary size.")
    flat = logits.view(-1, logits.size(-1))
    spans = []
    origin, row_stride = flat.storage_offset(), flat.stride(0)
    for response, _ in responses:
        start = (response.storage_offset() - origin) // row_stride
        spans.append((start, start + response.size(0)))

    lengths = [stop - start for start, stop in spans]
    tokens = [targets for _, targets in responses]
    if candidate_ids is None:
        ids = torch.cat(tokens)[:, None]
    else:
        ids = tokens[0].new_empty((sum(lengths), 1 + candidate_ids[0].size(-1)))
        for row_ids, targets, candidates in zip(ids.split(lengths), tokens, candidate_ids, strict=True):
            row_ids[:, 0] = targets
            row_ids[:, 1:] = candidates
    chunk_size = args.log_probs_chunk_size if args.log_probs_chunk_size > 0 else 1024
    k = min(max(int(topk_k), 1), flat.size(-1)) if with_topk else 0
    values, entropy, topk = _ScoreCenteringLogProbs.apply(
        flat,
        ids,
        spans,
        process_group,
        chunk_size,
        vocab_size,
        args.rollout_temperature,
        with_entropy,
        with_entropy and getattr(args, "entropy_coef", 0.0) != 0,
        k,
    )

    result = {"log_probs": list(values[:, 0].split(lengths))}
    if entropy is not None:
        result["entropy"] = list(entropy.split(lengths))
    if topk is not None:
        result["topk_token_ids"] = list(topk.split(lengths))
    if candidate_ids is not None:
        result["topk_log_probs"] = list(values[:, 1:].split(lengths))
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
    context = {
        "rollout_log_probs": torch.cat(batch["rollout_log_probs"]),
        "rollout_topk_log_probs": torch.cat(batch["rollout_topk_log_probs"]),
        "train_topk_log_probs": train_topk_log_probs,
        "loss_mask": torch.cat(local_masks).to(device=train_topk_log_probs.device),
    }
    if (support_complete := batch.get("rollout_topk_support_complete")) is not None:
        context["support_complete"] = torch.cat(support_complete)
    return context
