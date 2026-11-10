# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay native causal modules on a prefix without changing their math."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from examples.chunkflow.metadata import int32_metadata
from examples.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR, ChunkflowRuntime
from examples.chunkflow.state_store import ChunkState


TOKENS_ATTR = "chunkflow_prefix_tokens"
MAX_LENGTH_ATTR = "chunkflow_parent_max_seqlen"
PARENT_CU_ATTR = "chunkflow_parent_physical_cu"


def validate(module: Any) -> None:
    config = module.config
    # PP keeps each layer's complete prefix on its owning stage; CP splits it.
    if getattr(config, "context_parallel_size", 1) != 1:
        raise NotImplementedError("ChunkFlow native prefix replay requires context_parallel_size=1")
    granularity = getattr(config, "recompute_granularity", None)
    # Native MoE checkpointing stays inside the MLP and never replays QSA/PLE
    # after their per-unit context has been released.
    moe_only = granularity == "selective" and set(getattr(config, "recompute_modules", None) or []) == {"moe"}
    if (granularity is not None and not moe_only) or getattr(module, "checkpoint_core_attention", False):
        raise NotImplementedError("Native prefix replay only supports selective MoE recomputation")
    if getattr(config, "hidden_dropout", 0) or getattr(config, "attention_dropout", 0):
        raise NotImplementedError("ChunkFlow native prefix replay requires zero dropout")


def parallel(module: Any) -> tuple[int, int, Any]:
    if not getattr(module.config, "sequence_parallel", False):
        return 1, 0, None
    group = getattr(getattr(module, "pg_collection", None), "tp", None)
    if group is None:
        group = getattr(module, "tp_group", None)
    if group is None:
        if getattr(module.config, "tensor_model_parallel_size", 1) > 1:
            raise ValueError("Native prefix replay requires an explicit sequence-parallel process group")
        return 1, 0, None
    return torch.distributed.get_world_size(group), torch.distributed.get_rank(group), group


def gather(value: torch.Tensor, size: int, group: Any) -> torch.Tensor:
    if size == 1:
        return value
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

    return gather_from_sequence_parallel_region(value, tensor_parallel_output_grad=True, group=group)


def pad(value: torch.Tensor, total: int) -> torch.Tensor:
    return F.pad(value, (0, 0) * (value.ndim - 1) + (0, total - value.shape[0]))


def layout(packed: Any, size: int, *, minimum: int = 0) -> tuple[Any, int, int]:
    """Fresh physical boundaries: history padding and RoPE offsets are
    excluded."""
    from megatron.core.packed_seq_params import PackedSeqParams

    from relax.backends.megatron.qsa_layout_metadata import bind_physical_layout

    unit = getattr(packed, UNIT_ATTR)
    if getattr(packed, "qkv_format", None) != "thd":
        raise NotImplementedError("Native prefix replay requires packed THD inputs")
    segments = [segment for chunk in unit.chunks for segment in chunk.segments]
    if unit.is_dependent and len(segments) != 1:
        raise NotImplementedError("Native prefix replay requires one dependent segment per unit")
    lengths = [segment.prefix + segment.length for segment in segments]
    logical = sum(lengths)
    alignment = math.lcm(128, size)
    total = ((max(logical, minimum) + alignment - 1) // alignment) * alignment
    cu = [0, *itertools.accumulate(lengths)]
    max_document = getattr(packed, MAX_LENGTH_ATTR, max(lengths))
    if not isinstance(max_document, int) or max_document < max(lengths):
        raise ValueError("Native prefix replay requires a valid parent maximum document length")
    while cu[-1] < total:
        # Keep dummy documents within the original selector's geometry too.
        cu.append(min(total, cu[-1] + max_document))
    boundaries = int32_metadata(cu, packed.cu_seqlens_q.device)
    result = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=boundaries,
        cu_seqlens_kv=boundaries,
        max_seqlen_q=max(b - a for a, b in zip(cu[:-1], cu[1:])),
        max_seqlen_kv=max(b - a for a, b in zip(cu[:-1], cu[1:])),
    )
    result.cu_seqlens_q_cpu = cu
    setattr(result, MAX_LENGTH_ATTR, max_document)
    result.qsa_select_offsets_cpu = tuple(cu)
    bind_physical_layout(result, tuple(cu))
    return result, logical, total


@dataclass
class PrefixCall:
    hidden: torch.Tensor
    packed: Any
    history: list[ChunkState]
    unit_tokens: int
    original_tokens: int
    logical_tokens: int
    size: int
    rank: int
    group: Any

    def restore(self, output: torch.Tensor) -> torch.Tensor:
        output = gather(output, self.size, self.group)
        output = output[self.logical_tokens - self.unit_tokens : self.logical_tokens]
        output = pad(output, self.original_tokens)
        # Plain slices are intentional: a differentiable scatter would gather
        # gradients again, duplicating the gather's reduce-scatter backward.
        return output.chunk(self.size, dim=0)[self.rank].contiguous()

    def rotary(self, frequencies: Any, *, absolute: bool) -> Any:
        if isinstance(frequencies, tuple):
            return tuple(self._rotary(value, absolute, f"rope_{i}") for i, value in enumerate(frequencies))
        return self._rotary(frequencies, absolute, "rope")

    def _rotary(self, frequencies: torch.Tensor, absolute: bool, key: str) -> torch.Tensor:
        if frequencies is None or frequencies.requires_grad:
            raise NotImplementedError("Native QSA prefix replay requires non-trainable rotary frequencies")
        total = self.hidden.shape[0] * self.size
        if absolute:
            if frequencies.shape[0] != self.original_tokens:
                raise ValueError("MRoPE frequencies must cover the current physical unit")
            current = frequencies[: self.unit_tokens]
            if self.history:
                self.history[-1].tensors.setdefault(key, current.detach())
                current = torch.cat([*(state.tensors[key] for state in self.history[:-1]), current])
            return pad(current, total)
        needed = max(self.packed.max_seqlen_q, getattr(self.packed, MAX_LENGTH_ATTR))
        # Dummy padding positions do not affect real causal outputs. The real
        # prefix must be covered by the original RoPE table generated upstream.
        cu = self.packed.cu_seqlens_q_cpu
        real_max = max(b - a for a, b in zip(cu[:-1], cu[1:]) if b <= self.logical_tokens)
        if frequencies.shape[0] < real_max:
            raise ValueError("RoPE table does not cover the complete prefix")
        return pad(frequencies[:needed], needed) if frequencies.shape[0] < needed else frequencies


def build(
    module: Any,
    hidden: torch.Tensor,
    packed: Any,
    runtime: ChunkflowRuntime,
    namespace: str,
    *,
    minimum: int = 0,
) -> PrefixCall:
    size, rank, group = parallel(module)
    original_tokens = hidden.shape[0] * size
    unit = getattr(packed, UNIT_ATTR)
    if hidden.ndim != 3 or hidden.shape[1] != 1 or original_tokens < unit.num_tokens:
        raise ValueError("Native prefix replay expects [physical tokens / SP, 1, hidden]")
    native_packed, logical, total = layout(packed, size, minimum=minimum)
    current = gather(hidden, size, group)[: unit.num_tokens]
    history = []
    if unit.is_dependent:
        chunk = unit.chunks[0]
        history = runtime.history(
            (namespace, module.layer_number),
            chunk.group,
            chunk.group_idx,
            recompute=getattr(packed, RECOMPUTE_ATTR, False),
        )
        history[-1].link_tensor("hidden", current, accumulate_fp32=True)
        current = torch.cat([*(state.tensors["hidden"] for state in history[:-1]), current])
    if current.shape[0] != logical:
        raise ValueError("Native prefix history does not match the unit's sequence position")
    hidden = pad(current, total).chunk(size, dim=0)[rank].contiguous()
    return PrefixCall(hidden, native_packed, history, unit.num_tokens, original_tokens, logical, size, rank, group)
