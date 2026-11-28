# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Combine ETP1 expert slots with Megatron's token restoration program."""

import torch
import torch.distributed as dist


def combine_routed_experts(
    slots: torch.Tensor, local_ids: torch.Tensor, local_experts: int, group: dist.ProcessGroup
) -> torch.Tensor:
    from transformer_engine.pytorch import moe_permute, moe_unpermute

    tokens, topk, hidden = slots.shape
    if not tokens:
        return slots[:, 0]

    # Each valid slot has exactly one owner. Exchange it without adding expert
    # contributions together until all of the slots for the token are present.
    valid = local_ids >= 0
    slots.masked_fill_(~valid.unsqueeze(-1), 0)
    ids = torch.where(valid, local_ids + group.rank() * local_experts, -1)
    if group.size() > 1:
        # A routed slot has one owner. Zero from every other rank makes the
        # integer SUM a bit-preserving exchange of the slot and its expert ID.
        words = torch.cat((slots.contiguous().view(torch.int32), (ids.to(torch.int32) + 1).unsqueeze(-1)), dim=-1)
        dist.all_reduce(words, group=group)
        slots = words[..., :-1].contiguous().view(slots.dtype)
        ids = words[..., -1] - 1

    # TE uses expert-major rows and its mask map restores the original tokens.
    experts = group.size() * local_experts
    ids = ids.long()
    ids.masked_fill_(ids < 0, experts)
    mask = torch.zeros(tokens, experts + 1, device=slots.device, dtype=torch.bool)
    mask.scatter_(1, ids, True)
    order = ids.flatten().argsort(stable=True)
    _, row_map = moe_permute(slots.new_zeros((tokens, 16)), mask[:, :experts].contiguous(), tokens * topk)
    return moe_unpermute(slots.flatten(0, 1).index_select(0, order), row_map, restore_shape=(tokens, hidden))


class _SharedGateLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight):
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import mm_batch_invariant

        ctx.save_for_backward(inputs, weight)
        return mm_batch_invariant(inputs, weight.t())

    @staticmethod
    def backward(ctx, grad):
        inputs, weight = ctx.saved_tensors
        return grad @ weight, grad.t() @ inputs


def shared_expert_gate(hidden_states, weight):
    logits = _SharedGateLinear.apply(hidden_states.reshape(-1, weight.shape[-1]), weight)
    return logits.view(*hidden_states.shape[:-1], 1).sigmoid()
