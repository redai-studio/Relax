# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Replay the unfinished FLA chunk from its last committed FP32 state."""

from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F


_CHUNK_SIZE = 64


@torch.compiler.disable
def gates(
    a_log: torch.Tensor, dt_bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the Megatron Qwen3.8 decay and BF16 beta program."""
    # Keep the optimizer parameter in FP32 and use the rollout projection dtype for exp.
    rate = a_log.to(a.dtype).exp()
    return -rate * F.softplus(a.float() + dt_bias), b.sigmoid()


def output_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, *, zero_centered: bool = False) -> torch.Tensor:
    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine_torch import rmsnorm_fwd

    # Match Megatron Bridge's conversion of the HF gain to zero-centered gamma.
    centered_weight = weight if zero_centered else weight - 1
    normalized, _, _ = rmsnorm_fwd(x.contiguous(), centered_weight, eps, None, None, TE_DType[x.dtype], 0, True)
    return normalized


@torch.compiler.disable
def output_gate(normalized: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    # Use the same eager FP32 sigmoid and separate multiply in both engines.
    return (normalized * gate.reshape_as(normalized).float().sigmoid()).to(normalized.dtype)


def ordered_tp_sum(local: torch.Tensor, group: dist.ProcessGroup, *, sequence_parallel: bool = False) -> torch.Tensor:
    """Sum BF16 projection partials in process-group rank order using FP32."""
    size = group.size()
    if size == 1:
        return local

    local = local.contiguous()
    if sequence_parallel:
        # Exchange only the rows owned by each rank. Receive one block per source rank.
        gathered = torch.empty_like(local)
        dist.all_to_all_single(gathered, local, group=group)
    else:
        gathered = local.new_empty((size * local.shape[0], *local.shape[1:]))
        dist.all_gather_into_tensor(gathered, local, group=group)

    first, *rest = gathered.view(size, -1, *local.shape[1:]).unbind(0)
    total = first.float()
    for partial in rest:
        total.add_(partial)
    return total.to(local.dtype)


class GDNBitwiseState:
    """One pending chunk per SGLang state slot and GDN layer.

    The SGLang temporal pool holds the anchor at the last 64-token boundary.
    Sequence positions come from the scheduler, so a fresh request reusing a
    slot never reads the previous owner's anchor or unfinished rows.
    """

    def __init__(self) -> None:
        self._pending: dict[int, tuple[torch.Tensor, ...]] = {}

    def _pending_for(self, layer_id: int, inputs: tuple[torch.Tensor, ...], slots: int) -> tuple[torch.Tensor, ...]:
        if layer_id not in self._pending:
            self._pending[layer_id] = tuple(
                value.new_empty((slots, _CHUNK_SIZE, *value.shape[2:])) for value in inputs
            )
        return self._pending[layer_id]

    def forward_decode(
        self,
        layer_id: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        states: torch.Tensor,
        slots: torch.Tensor,
        prefix_lens: Sequence[int],
    ) -> torch.Tensor:
        if len(prefix_lens) < 2:
            return self.forward(layer_id, q, k, v, g, beta, states, slots, prefix_lens, [1] * len(prefix_lens))

        from fla.modules.l2norm import l2norm
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        slots = slots.long()
        tails = torch.tensor([n % _CHUNK_SIZE for n in prefix_lens], device=slots.device)
        padding = torch.arange(_CHUNK_SIZE, device=slots.device)[None, :] > tails[:, None]
        inputs = (l2norm(q.contiguous()), l2norm(k.contiguous()), v, g.unsqueeze(0), beta.unsqueeze(0))
        pending = self._pending_for(layer_id, inputs, states.shape[0])
        padded = []
        for history, token in zip(pending, inputs, strict=True):
            history[slots, tails] = token[0]
            chunk = history.index_select(0, slots)
            chunk.masked_fill_(padding.view(*padding.shape, *([1] * (chunk.ndim - 2))), 0)
            padded.append(chunk)

        # FLA uses [K,V] for the anchor, while the SGLang cache stores [V,K].
        anchor = states.index_select(0, slots).transpose(-1, -2).contiguous()
        fresh = [row for row, n in enumerate(prefix_lens) if n == 0]
        if fresh:
            rows = torch.tensor(fresh, device=slots.device)
            anchor.index_fill_(0, rows, 0)
            states.index_fill_(0, slots.index_select(0, rows), 0)

        # Only full chunks advance the committed anchor. Future zero rows
        # preserve the 64-token FLA program without extending a request.
        completed = [row for row, n in enumerate(prefix_lens) if (n + 1) % _CHUNK_SIZE == 0]
        query, key, value, decay, strength = padded
        ratio = value.shape[2] // query.shape[2]
        output, final_state = chunk_gated_delta_rule(
            query.repeat_interleave(ratio, dim=2),
            key.repeat_interleave(ratio, dim=2),
            value,
            decay,
            strength,
            initial_state=anchor,
            state_v_first=False,
            output_final_state=bool(completed),
        )
        if completed:
            rows = torch.tensor(completed, device=slots.device)
            states.index_copy_(0, slots.index_select(0, rows), final_state.index_select(0, rows).transpose(-1, -2))
        return output[torch.arange(tails.numel(), device=slots.device), tails].unsqueeze(0)

    def forward(
        self,
        layer_id: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        states: torch.Tensor,
        slots: torch.Tensor,
        prefix_lens: Sequence[int],
        lengths: Sequence[int],
    ) -> torch.Tensor:
        from fla.modules.l2norm import l2norm
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        if not lengths:
            return torch.empty_like(v)

        q = l2norm(q.contiguous())
        k = l2norm(k.contiguous())
        inputs = (q, k, v, g.unsqueeze(0), beta.unsqueeze(0))
        pending = self._pending_for(layer_id, inputs, states.shape[0])
        ratio = v.shape[2] // q.shape[2]
        slots = slots.long()
        output = torch.empty_like(v)
        offset = 0
        for row, (prefix, length) in enumerate(zip(prefix_lens, lengths, strict=True)):
            if length == 0:
                continue
            slot = slots[row : row + 1]
            tail = prefix % _CHUNK_SIZE
            end = offset + length
            values = tuple(value[:, offset:end] for value in inputs)
            # The SGLang cache is [V,K]; Megatron's FLA program uses [K,V].
            anchor = states.index_select(0, slot).transpose(-1, -2).contiguous() if prefix else None
            if tail:
                values = tuple(
                    torch.cat((history[:, :tail].index_select(0, slot), value), dim=1)
                    for history, value in zip(pending, values, strict=True)
                )

            completed = (tail + length) // _CHUNK_SIZE * _CHUNK_SIZE
            if completed:
                query, key, value, decay, strength = (x[:, :completed] for x in values)
                chunks, anchor = chunk_gated_delta_rule(
                    q=query.repeat_interleave(ratio, dim=2),
                    k=key.repeat_interleave(ratio, dim=2),
                    v=value.contiguous(),
                    g=decay.contiguous(),
                    beta=strength.contiguous(),
                    state_v_first=False,
                    initial_state=anchor,
                    output_final_state=True,
                )
                states.index_copy_(0, slot, anchor.transpose(-1, -2))
                output[:, offset : offset + completed - tail] = chunks[:, tail:]
            elif prefix == 0:
                states.index_fill_(0, slot, 0)

            remaining = tail + length - completed
            if remaining:
                tail_rows = tuple(x[:, completed:].contiguous() for x in values)
                query, key, value, decay, strength = tail_rows
                trailing, _ = chunk_gated_delta_rule(
                    q=query.repeat_interleave(ratio, dim=2),
                    k=key.repeat_interleave(ratio, dim=2),
                    v=value,
                    g=decay,
                    beta=strength,
                    state_v_first=False,
                    initial_state=anchor,
                )
                already_emitted = tail if completed == 0 else 0
                output[:, end - remaining + already_emitted : end] = trailing[:, already_emitted:]
                for history, saved in zip(pending, tail_rows, strict=True):
                    history[:, :remaining].index_copy_(0, slot, saved)
            offset = end
        return output
