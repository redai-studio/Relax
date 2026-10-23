# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Modified for Relax integration; see NOTICE.

"""Instance-owned cross-layer state and tensor carry for the V4.1 text model.

The carry layout is derived from the layer plan instead of reading a GPU
metadata header on every layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any, Literal

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class V41Runtime:
    """One model instance's current microbatch state, never stored on a shared
    config."""

    pre: torch.Tensor | None = None
    input_ids: torch.Tensor | None = None
    latent: dict[int, torch.Tensor] = field(default_factory=dict)
    index_k: dict[int, torch.Tensor] = field(default_factory=dict)
    topk: torch.Tensor | None = None
    candidates: torch.Tensor | None = None

    def reset(self) -> None:
        self.pre = None
        self.input_ids = None
        self.latent.clear()
        self.index_k.clear()
        self.topk = None
        self.candidates = None


def v41_aggregate(hidden: torch.Tensor, pre: torch.Tensor | None, n: int) -> torch.Tensor:
    """Consume the previous sublayer's pre coefficients, as required by V4.1
    single-pass mHC."""
    s, b, width = hidden.shape
    streams = hidden.view(s, b, n, width // n)
    if pre is None:
        return streams[:, :, 0].to(hidden.dtype)
    return (pre.float().unsqueeze(-1) * streams.float()).sum(dim=2).to(hidden.dtype)


def v41_pack_pre(pre: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Carry FP32 pre coefficients through BF16 tensors using differentiable
    high/low parts."""
    high = pre.to(dtype)
    low = (pre.float() - high.float()).to(dtype)
    return torch.cat((high, low), dim=-1)


def v41_unpack_pre(packed: torch.Tensor, n: int) -> torch.Tensor:
    return packed[..., :n].float() + packed[..., n:].float()


def v41_kv_source_layer(config: Any, layer_id: int) -> int | None:
    ratio = config.csa_compress_ratios[layer_id]
    if ratio == 0:
        return None
    sources = [
        source
        for source in config.v41_kv_source_layer_ids
        if source <= layer_id and config.csa_compress_ratios[source] == ratio
    ]
    if not sources:
        raise ValueError(f"DeepSeek V4.1: layer {layer_id} has no KV source with compression ratio {ratio}.")
    return max(sources)


@dataclass(frozen=True, slots=True)
class _CarrySection:
    name: str
    key: int | None
    shape: tuple[int, ...]
    encoding: Literal["pre", "float", "int", "candidate"]
    elements: int


@dataclass(frozen=True, slots=True, init=False)
class V41CarryPlan:
    """Static carry layout immediately after a global decoder layer (zero-
    based)."""

    pre: bool
    input_ids: bool
    latent_keys: tuple[int, ...]
    index_k_keys: tuple[int, ...]
    topk: bool
    candidates: bool
    candidate_block_size: int
    candidate_ratio: int
    compress_ratios: tuple[int, ...]
    hidden_size: int
    num_residual_streams: int
    sequence_parallel: bool
    tp_size: int
    cp_size: int
    head_dim: int
    index_head_dim: int
    index_topk: int

    def __init__(self, config: Any, layer_id: int):
        if not -1 <= layer_id < config.num_layers:
            raise ValueError(f"DeepSeek V4.1: invalid carry boundary {layer_id} for {config.num_layers} layers.")
        ratios = tuple(config.csa_compress_ratios[: config.num_layers])
        later = range(layer_id + 1, config.num_layers)
        past = set(range(layer_id + 1))
        latent_keys = {v41_kv_source_layer(config, layer) for layer in later if ratios[layer]} & past
        index_k_keys = {
            v41_kv_source_layer(config, layer)
            for layer in later
            if ratios[layer] and layer in config.v41_index_source_layer_ids
        } & past
        next_layer = layer_id + 1
        candidate_source = config.v41_candidate_source_layer_id
        candidates = 0 <= candidate_source <= layer_id and any(
            candidate_source < layer and ratios[layer] for layer in later
        )
        values = {
            "pre": layer_id >= 0,
            "input_ids": bool(config.v41_engram_layer_ids) and layer_id < max(config.v41_engram_layer_ids),
            "latent_keys": tuple(sorted(latent_keys)),
            "index_k_keys": tuple(sorted(index_k_keys)),
            "topk": next_layer < config.num_layers
            and ratios[next_layer] > 0
            and next_layer not in config.v41_index_source_layer_ids,
            "candidates": candidates,
            "candidate_block_size": config.v41_candidate_block_size,
            "candidate_ratio": ratios[candidate_source] if candidates else 0,
            "compress_ratios": ratios,
            "hidden_size": config.hidden_size,
            "num_residual_streams": config.num_residual_streams,
            "sequence_parallel": config.sequence_parallel,
            "tp_size": config.tensor_model_parallel_size,
            "cp_size": config.context_parallel_size,
            "head_dim": config.kv_lora_rank,
            "index_head_dim": config.dsa_indexer_head_dim,
            "index_topk": config.dsa_indexer_topk,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def base_cols(self) -> int:
        return self.hidden_size * self.num_residual_streams

    def sections(self, s: int, b: int) -> tuple[_CarrySection, ...]:
        local_length = s * self.tp_size if self.sequence_parallel else s
        global_length = local_length * self.cp_size
        sections = []
        if self.pre:
            shape = (s, b, self.num_residual_streams)
            sections.append(_CarrySection("pre", None, shape, "pre", 2 * prod(shape)))
        if self.input_ids:
            shape = (b, local_length)
            sections.append(_CarrySection("input_ids", None, shape, "int", 2 * prod(shape)))
        for name, keys, width in (
            ("latent", self.latent_keys, self.head_dim),
            ("index_k", self.index_k_keys, self.index_head_dim),
        ):
            for key in keys:
                ratio = self.compress_ratios[key]
                if self.cp_size > 1 and local_length % ratio:
                    raise ValueError("DeepSeek V4.1: CP chunk length must be divisible by its compression ratio.")
                shape = (b, global_length // ratio, width)
                sections.append(_CarrySection(name, key, shape, "float", prod(shape)))
        if self.topk:
            shape = (b, local_length, self.index_topk)
            sections.append(_CarrySection("topk", None, shape, "int", 2 * prod(shape)))
        if self.candidates:
            if self.candidate_ratio <= 0 or self.candidate_block_size <= 0:
                raise ValueError("DeepSeek V4.1: candidate source ratio and block size must be positive.")
            width = global_length // self.candidate_ratio
            shape = (b, local_length, width)
            blocks = (width + self.candidate_block_size - 1) // self.candidate_block_size
            elements = (b * local_length * blocks + 1) // 2
            sections.append(_CarrySection("candidates", None, shape, "candidate", elements))
        return tuple(sections)


class V41IdsOnlyPlan(V41CarryPlan):
    __slots__ = ()

    def __init__(self, config: Any):
        super().__init__(config, -1)


def _bits_to_bf16(value: torch.Tensor) -> torch.Tensor:
    flat = value.detach().contiguous().flatten()
    if flat.dtype == torch.bool:
        flat = flat.view(torch.uint8)
    if flat.dtype == torch.uint8:
        if flat.numel() % 2:
            flat = F.pad(flat, (0, 1))
        return flat.view(torch.bfloat16)
    return flat.to(torch.int32).view(torch.bfloat16)


def _carry_sections(hidden: torch.Tensor, plan: V41CarryPlan) -> tuple[_CarrySection, ...]:
    if hidden.dtype != torch.bfloat16:
        raise ValueError("DeepSeek V4.1 carry requires BF16; casting its encoded integers would corrupt state.")
    if hidden.ndim != 3 or hidden.shape[0] == 0 or hidden.shape[1] == 0:
        raise ValueError("DeepSeek V4.1 carry expects a nonempty [sequence, batch, hidden] tensor.")
    return plan.sections(hidden.shape[0], hidden.shape[1])


def v41_pack_carry(hidden: torch.Tensor, runtime: V41Runtime, plan: V41CarryPlan) -> torch.Tensor:
    sections = _carry_sections(hidden, plan)
    if hidden.shape[-1] != plan.base_cols:
        raise ValueError(f"DeepSeek V4.1 carry expected hidden width {plan.base_cols}, got {hidden.shape[-1]}.")
    parts = []
    for section in sections:
        value = getattr(runtime, section.name)
        if section.key is not None:
            value = value.get(section.key)
        if value is None or tuple(value.shape) != section.shape:
            actual = None if value is None else tuple(value.shape)
            raise ValueError(
                f"DeepSeek V4.1 carry {section.name}[{section.key}] expected {section.shape}, got {actual}."
            )
        if section.encoding == "pre":
            flat = v41_pack_pre(value, hidden.dtype).flatten()
        elif section.encoding == "int":
            if value.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"DeepSeek V4.1 carry {section.name} must contain integer indices.")
            flat = _bits_to_bf16(value)
        elif section.encoding == "candidate":
            if value.dtype != torch.bool:
                raise ValueError("DeepSeek V4.1 carry candidates must be a boolean mask.")
            flat = _bits_to_bf16(value[..., :: plan.candidate_block_size])
        else:
            if value.dtype != hidden.dtype:
                raise ValueError(f"DeepSeek V4.1 carry {section.name} must have the hidden-state BF16 dtype.")
            flat = value.flatten()
        parts.append(flat)
    if not parts:
        return hidden
    flat = torch.cat(parts)
    rows = hidden.shape[0] * hidden.shape[1]
    padding = (-flat.numel()) % rows
    if padding:
        flat = F.pad(flat, (0, padding))
    return torch.cat((hidden, flat.view(*hidden.shape[:2], -1)), dim=-1)


def v41_unpack_carry(
    hidden: torch.Tensor, runtime: V41Runtime, plan: V41CarryPlan, base_cols: int | None = None
) -> torch.Tensor:
    sections = _carry_sections(hidden, plan)
    base_cols = plan.base_cols if base_cols is None else base_cols
    if base_cols != plan.base_cols:
        raise ValueError("DeepSeek V4.1 carry base width does not agree with the layer plan.")
    rows = hidden.shape[0] * hidden.shape[1]
    elements = sum(section.elements for section in sections)
    expected_width = base_cols + (elements + rows - 1) // rows
    if hidden.shape[-1] != expected_width:
        raise ValueError(f"DeepSeek V4.1 carry expected packed width {expected_width}, got {hidden.shape[-1]}.")
    carry = hidden[..., base_cols:].contiguous().flatten()
    runtime.reset()
    offset = 0
    for section in sections:
        flat = carry[offset : offset + section.elements]
        offset += section.elements
        if section.encoding == "pre":
            value = v41_unpack_pre(flat.view(*section.shape[:-1], 2 * section.shape[-1]), section.shape[-1])
        elif section.encoding == "int":
            value = flat.detach().view(torch.int32).view(section.shape).to(torch.int64)
        elif section.encoding == "candidate":
            b, s, width = section.shape
            blocks = (width + plan.candidate_block_size - 1) // plan.candidate_block_size
            raw = flat.detach().view(torch.uint8)[: b * s * blocks]
            value = raw.view(b, s, blocks).bool().repeat_interleave(plan.candidate_block_size, dim=-1)[..., :width]
        else:
            value = flat.view(section.shape)
        if section.key is None:
            setattr(runtime, section.name, value)
        else:
            getattr(runtime, section.name)[section.key] = value
    return hidden[..., :base_cols].contiguous()
