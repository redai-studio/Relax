# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Chunked reference GQA/DSA with tensor parallelism and zigzag CP."""

from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from torch import nn

from .indexer import KeyeIndexer, indexer_scores
from .sparse_attention import compact_indices, sparse_attention


def query_chunks(length: int, cp_size: int, cp_rank: int, chunk_size: int) -> Iterator[tuple[int, int, int]]:
    """Yield local query ranges and their last causal key, without GPU
    reads."""
    if cp_size == 1:
        segments = [(0, length, 0)]
    else:
        half = length // 2
        segments = [(0, half, cp_rank * half), (half, length, (2 * cp_size - cp_rank - 1) * half)]
    for start, end, global_start in segments:
        for offset in range(start, end, chunk_size):
            stop = min(offset + chunk_size, end)
            yield offset, stop, global_start + stop - start


class KeyeSelfAttention(SelfAttention):
    def forward(
        self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Full-layer recomputation re-enters this method with the correct input.
        # Selective core-attention recomputation is deliberately unsupported.
        if self.config.recompute_granularity == "selective":
            raise ValueError("Keye currently requires full recomputation or no recomputation")
        self._keye_hidden = hidden_states
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            self._keye_hidden = None

    def _run_core_attention(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return super()._run_core_attention(*args, hidden_states=self._keye_hidden, **kwargs)


class KeyeCoreAttention(nn.Module):
    def __init__(self, config: Any, layer_number: int, **kwargs: Any) -> None:
        super().__init__()
        self.config = config
        self.tp_size = config.tensor_model_parallel_size
        self.tp_group = getattr(kwargs.get("pg_collection"), "tp", None)
        if self.tp_size > 1 and (not config.sequence_parallel or self.tp_group is None):
            raise ValueError("Keye TP attention requires sequence parallelism and an explicit TP group")
        self.cp_group = getattr(kwargs.get("pg_collection"), "cp", None)
        self.sa_indexer = KeyeIndexer(config.hidden_size, config.layernorm_epsilon, config.rotary_base)
        for parameter in self.sa_indexer.parameters():
            parameter.allreduce = True
            parameter.sequence_parallel = False
            parameter.average_gradients_across_tp_domain = self.tp_size > 1

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        packed_seq_params: Any = None,
        hidden_states: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if query.ndim != 3 or hidden_states.shape[1] != 1 or packed_seq_params is None:
            raise ValueError("Keye training requires packed THD inputs with micro batch size 1")
        if self.tp_size > 1:
            from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

            hidden_states = gather_from_sequence_parallel_region(
                hidden_states.detach(), tensor_parallel_output_grad=False, group=self.tp_group
            )
        boundaries = getattr(packed_seq_params, "cu_seqlens_q_cpu", None)
        if boundaries is None:
            raise ValueError("Keye requires CPU packed boundaries supplied by Relax's data loader")
        from relax.backends.megatron.cp_utils import gdn_cp_gather_full, gdn_cp_slice

        size = self.config.context_parallel_size
        rank = self.cp_group.rank() if size > 1 else 0
        indices = torch.arange(boundaries[-1], device=query.device)
        positions = getattr(packed_seq_params, "keye_positions", None)
        if positions is None:
            positions = torch.cat(
                [torch.arange(end - start, device=query.device) for start, end in zip(boundaries, boundaries[1:])]
            )
        valid = getattr(packed_seq_params, "keye_valid_mask", None)
        if valid is None:
            valid_lengths = getattr(packed_seq_params, "valid_seqlens_q_cpu", None)
            if valid_lengths is None:
                valid = indices < getattr(packed_seq_params, "num_valid_tokens", boundaries[-1])
            else:
                valid = torch.cat(
                    [
                        indices[start:end] - start < length
                        for start, end, length in zip(boundaries[:-1], boundaries[1:], valid_lengths, strict=True)
                    ]
                )
        local_indices = gdn_cp_slice(indices, boundaries, size, rank) if size > 1 else indices
        local_positions = positions.index_select(0, local_indices)
        iq, ik, iw = self.sa_indexer(hidden_states[:, 0], local_positions)
        if size > 1:
            key, value, ik = [
                gdn_cp_gather_full(tensor, boundaries, size, self.cp_group) for tensor in (key, value, ik)
            ]
        outputs = []
        auxiliary = query.new_zeros((), dtype=torch.float32)
        train_indexer = self.training and torch.is_grad_enabled()
        local_start = 0
        for start, end in zip(boundaries, boundaries[1:]):
            length = (end - start) // size
            q = query[local_start : local_start + length]
            k, v = key[start:end], value[start:end]
            kv_heads = k.shape[1]
            # FP32 teacher keys retain GQA layout and are converted once per
            # document, rather than converting 32 expanded heads per chunk.
            teacher_k = k.detach().float().permute(1, 2, 0) if train_indexer else None
            token_offsets = local_indices[local_start : local_start + length]
            chunk_size = getattr(self.config, "keye_attention_chunk_size", 1024)
            for offset, stop, prefix in query_chunks(length, size, rank, chunk_size):
                lo, hi = local_start + offset, local_start + stop
                scores = indexer_scores(iq[lo:hi], ik[start : start + prefix], iw[lo:hi])
                causal = (indices[start : start + prefix][None] <= token_offsets[offset:stop, None]) & valid[
                    start : start + prefix
                ][None]
                scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
                selected = scores.detach().topk(min(2048, prefix), -1).indices
                support = torch.zeros_like(causal).scatter_(1, selected, True) & causal
                slots, counts = compact_indices(selected, causal.gather(1, selected))
                chunk_q = q[offset:stop].transpose(0, 1)
                output = sparse_attention(q[offset:stop], k[:prefix], v[:prefix], slots, counts, support)
                outputs.append(output.reshape(stop - offset, -1))
                if train_indexer:
                    with torch.no_grad():
                        teacher_q = chunk_q.detach().float().reshape(kv_heads, -1, q.shape[-1])
                        target_logits = (
                            torch.bmm(teacher_q, teacher_k[:, :, :prefix]).reshape(q.shape[1], stop - offset, prefix)
                            * q.shape[-1] ** -0.5
                        )
                        target = target_logits.masked_fill(~support.unsqueeze(0), float("-inf")).softmax(-1).mean(0)
                        target = target.nan_to_num()
                        if self.tp_size > 1:
                            dist.all_reduce(target, op=dist.ReduceOp.AVG, group=self.tp_group)
                    log_probs = scores.float().masked_fill(~support, torch.finfo(torch.float32).min).log_softmax(-1)
                    kl = F.kl_div(log_probs, target, reduction="none").sum(-1)
                    auxiliary = auxiliary + (kl * valid.index_select(0, token_offsets[offset:stop])).sum()
            local_start += length
        output = torch.cat(outputs, dim=0)
        if train_indexer:
            coefficient = 0.0 if getattr(packed_seq_params, "is_dummy", False) else self.config.keye_indexer_loss_coeff
            output = DSAIndexerLossAutoScaler.apply(output, auxiliary * coefficient)
        return output
