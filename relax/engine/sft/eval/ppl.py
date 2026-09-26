# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SFT periodic PPL eval.

Forward-only callback used inside ``backends/megatron/model.py:forward_only``
plus a final aggregator that turns per-microbatch sums into loss / PPL.

Loss-mask handling: ``forward_only``'s ``forward_step`` passes the per-sample
``loss_masks`` (pre-CP-chunked) into the callback; we reuse Relax's existing
``get_sum_of_sample_mean(calculate_per_token_loss=True)`` to compute the
masked sum exactly as the SFT training loss does (so PPL aligns with the
quantity the optimizer minimizes).

CP > 1: both numerator and denominator are computed on the per-CP-rank
chunked loss mask (the same one ``get_sum_of_sample_mean`` builds for the
numerator), so summing across the CP group recovers the full-sequence totals.
Caller must include the CP group in the final all-reduce.
"""

import math

import torch


def compute_sft_eval_step(
    logits: torch.Tensor,
    *,
    args,
    unconcat_tokens,
    total_lengths,
    response_lengths,
    with_entropy: bool = False,  # noqa: ARG001 — signature parity with get_log_probs_and_entropy
    max_seq_lens=None,
    padded_total_lengths=None,
    loss_masks=None,
    dynamic_cp_size=None,
    dynamic_cp_rank=None,
    **_,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Per-microbatch callback: returns ``(loss_placeholder, dict)`` where the
    dict carries ``sum_neg_log_prob`` and ``num_tokens``.

    Both dict values are 1-element tensors so ``forward_only`` can stack them
    across microbatches. The leading empty tensor matches the 2-tuple shape
    Megatron's pipeline scheduler expects from a forward_only callback (see
    ``get_log_probs_and_entropy``); without it Megatron unpacks the dict's keys
    as ``(output_tensor, loss_reduced)`` and then runs ``output_tensor /=
    num_microbatches`` against a string.
    """
    from relax.backends.megatron.cp_utils import get_sum_of_sample_mean
    from relax.backends.megatron.loss import get_log_probs_and_entropy

    assert loss_masks is not None, "compute_sft_eval_step requires loss_masks (passed by forward_step)."

    _, lp = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=max_seq_lens,
        padded_total_lengths=padded_total_lengths,
        dynamic_cp_size=dynamic_cp_size,
        dynamic_cp_rank=dynamic_cp_rank,
    )
    log_probs_flat = torch.cat(lp["log_probs"], dim=0)

    sum_of_token = get_sum_of_sample_mean(
        total_lengths,
        response_lengths,
        loss_masks,
        calculate_per_token_loss=True,
        qkv_format=args.qkv_format,
        max_seq_lens=max_seq_lens,
        padded_total_lengths=padded_total_lengths,
        dynamic_cp_size=dynamic_cp_size,
        dynamic_cp_rank=dynamic_cp_rank,
    )
    sum_neg_lp = -sum_of_token(log_probs_flat).detach()
    # Count tokens on the *chunked* mask (via sum_of_token with x=1) so that
    # under CP > 1 each rank reports only its slice's count; summing across the
    # CP group then recovers the full token count. Under CP == 1 this collapses
    # to sum(loss_masks).
    num_tokens_t = sum_of_token(torch.ones_like(log_probs_flat)).detach()

    device = log_probs_flat.device
    return torch.empty((0,), device=device), {
        "sum_neg_log_prob": [sum_neg_lp.reshape(1).to(torch.float32)],
        "num_tokens": [num_tokens_t.reshape(1).to(torch.long)],
    }


def compute_ppl_metrics(total_neg_log_prob: float, total_tokens: int) -> dict[str, float]:
    """Aggregate already-reduced sums into the final metric dict.

    ``eval/loss`` matches the training-side ``loss`` key emitted by
    ``sft_loss_function`` (masked mean cross-entropy), so train and eval curves
    can be overlaid directly.
    """
    if total_tokens == 0:
        return {"eval/loss": 0.0, "eval/ppl": 0.0, "eval/num_tokens": 0}
    loss = total_neg_log_prob / total_tokens
    return {
        "eval/loss": loss,
        "eval/ppl": math.exp(loss),
        "eval/num_tokens": total_tokens,
    }
