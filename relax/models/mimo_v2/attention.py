# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from Miles (RadixArk/miles, fa13e2ec), licensed under Apache-2.0.
# Copyright 2025-2026 RadixArk, Inc. Miles includes software developed from Slime,
# Copyright 2025 Zhipu AI, licensed under Apache-2.0.

from __future__ import annotations

import dataclasses

import torch
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.transformer.attention import SelfAttention


def _qkv_group_sizes(*, heads_per_group: int, qk_dim: int, v_dim: int) -> list[int]:
    """Rows of one KV group in the fused ``linear_qkv`` output: its query
    heads, one key head, one value head."""
    return [heads_per_group * qk_dim, qk_dim, v_dim]


def _split_qkv(qkv: torch.Tensor, *, dim: int, groups: int, heads_per_group: int, qk_dim: int, v_dim: int):
    """Split ``dim`` of a fused qkv into (query, key, value), each keeping a
    KV-group axis at ``dim``."""
    sizes = _qkv_group_sizes(heads_per_group=heads_per_group, qk_dim=qk_dim, v_dim=v_dim)
    dim %= qkv.dim()
    grouped = qkv.view(*qkv.shape[:dim], groups, sum(sizes), *qkv.shape[dim + 1 :])
    return torch.split(grouped, sizes, dim=dim + 1)


def _sized_linear(build, *, input_size: int | None = None, output_size: int | None = None):
    """``build`` with the given input/output size in place of the one
    SelfAttention passes."""

    def build_sized(base_input_size, base_output_size, **kwargs):
        return build(
            base_input_size if input_size is None else input_size,
            base_output_size if output_size is None else output_size,
            **kwargs,
        )

    return build_sized


class MiMoV2SelfAttention(SelfAttention):
    """SelfAttention whose V heads (``v_head_dim``) are narrower than its Q/K
    heads (``kv_channels``).

    The stock layer sizes ``linear_qkv``, ``linear_proj`` and the q/k/v split
    with ``kv_channels`` for V; this builds the two projections once at the
    MiMo sizes and splits accordingly. The layer config already carries this
    layer's KV heads (``MiMoV2ModelProvider.get_config_for_layer``).
    """

    def __init__(self, config, submodules, layer_number, *args, **kwargs):
        assert not (config.attention_output_gate or config.head_wise_attn_gate), "MiMo-V2 has no attention gate"
        # qk_clip reads linear_qkv as q + 2 * kv rows of kv_channels each, which does not hold for a narrower V.
        assert not config.qk_clip, "qk_clip does not support a V head narrower than Q/K"
        heads, groups = config.num_attention_heads, config.num_query_groups
        group_sizes = _qkv_group_sizes(
            heads_per_group=heads // groups, qk_dim=config.kv_channels, v_dim=config.v_head_dim
        )
        qkv_out_dim = groups * sum(group_sizes)
        submodules = dataclasses.replace(
            submodules,
            linear_qkv=_sized_linear(submodules.linear_qkv, output_size=qkv_out_dim),
            linear_proj=_sized_linear(submodules.linear_proj, input_size=heads * config.v_head_dim),
        )
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        assert groups >= self.world_size, "TP must not exceed the layer's KV heads"
        self.linear_qkv_out_dim = qkv_out_dim
        self.val_hidden_size = config.v_head_dim

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states=None, output_gate=False, head_wise_gate=False, split_qkv=True
    ):
        assert not (output_gate or head_wise_gate) and split_qkv, "MiMo-V2 attention needs the split q/k/v path"
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        groups = self.num_query_groups_per_partition
        heads_per_group = self.num_attention_heads_per_partition // groups
        qk_dim, v_dim = self.hidden_size_per_attention_head, self.config.v_head_dim
        # [sq, b, ng * (np/ng * hn + hn + hv)] -> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hv]
        query, key, value = _split_qkv(
            mixed_qkv, dim=-1, groups=groups, heads_per_group=heads_per_group, qk_dim=qk_dim, v_dim=v_dim
        )
        query = query.reshape(query.size(0), query.size(1), -1, qk_dim)
        return query, key, value


class MiMoV2TEDotProductAttention(TEDotProductAttention):
    """TE core attention with separate Q/K and V channels and V scaled by
    ``attention_value_scale``.

    The learnable sink comes from the layer config's ``softmax_type``.
    """

    def __init__(self, config, layer_number, attn_mask_type, attention_type, attention_dropout=None, **kwargs):
        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            k_channels=config.kv_channels,
            v_channels=config.v_head_dim,
            **kwargs,
        )
        self.attention_value_scale = config.attention_value_scale

    def forward(self, query, key, value, attention_mask, attn_mask_type, **kwargs):
        if self.attention_value_scale is not None:
            value = value * self.attention_value_scale
        return super().forward(query, key, value, attention_mask, attn_mask_type, **kwargs)


AutoMapping.register_module_type("MiMoV2TEDotProductAttention", "column")
