# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from Miles (RadixArk/miles, fa13e2ec), licensed under Apache-2.0.
# Copyright 2025-2026 RadixArk, Inc. Miles includes software developed from Slime,
# Copyright 2025 Zhipu AI, licensed under Apache-2.0.

"""Megatron-Bridge support for XiaomiMiMo MiMo-V2 (``MiMoV2ForCausalLM``, e.g.
MiMo-V2.6-Flash-RL).

Builds on NVIDIA Megatron-Bridge ``megatron/bridge/models/mimo_v2_flash``, which covers the earlier
MiMo-V2-Flash checkpoint, and reuses its q/k/v weight mapping. MiMo-V2 attention differs from the
stock GPT layer in four ways:

* per-layer KV heads: SWA layers use ``swa_num_key_value_heads``, global layers ``num_key_value_heads``;
* a learnable attention sink on SWA layers only, while ``softmax_type`` is one global setting;
* asymmetric head dims: Q/K use ``head_dim`` (192) and V uses ``v_head_dim`` (128);
* V is multiplied by ``attention_value_scale`` before attention.

The first two are per-layer config values: the provider sets ``heterogeneous_block_specs``, so
``TransformerBlock`` builds each layer from ``MiMoV2ModelProvider.get_config_for_layer``. The other
two need ``MiMoV2SelfAttention`` and ``MiMoV2TEDotProductAttention``. The per-layer RoPE base and the
SWA window reuse existing config fields (``rotary_base_per_layer``, ``window_size`` with a per-layer
``window_attn_skip_freq``) instead of Megatron-Bridge's dual-base RoPE and window rule.

The bridge reads the BF16 split-q/k/v layout written by the Miles BF16 converter; the
official checkpoint (FP8/MXFP4, fused kv-head-interleaved ``qkv_proj``) must be converted first.
LoRA adapters of ``linear_qkv`` export through the same per-group q/k/v split as the attention.
Only the text decoder is built: vision/audio towers and the MTP layers are not.
"""

from .attention import MiMoV2SelfAttention, MiMoV2TEDotProductAttention
from .bridge import MiMoV2Bridge
from .provider import MiMoV2ModelProvider, mimo_v2_layer_spec


__all__ = [
    "MiMoV2Bridge",
    "MiMoV2ModelProvider",
    "MiMoV2SelfAttention",
    "MiMoV2TEDotProductAttention",
    "mimo_v2_layer_spec",
]
