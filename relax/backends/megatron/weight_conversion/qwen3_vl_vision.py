# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Convert native Megatron and legacy HF-backed Qwen3-VL vision towers."""

import re
from argparse import Namespace

import torch


_LAYER_NAMES = {
    "self_attention.linear_proj.weight": "attn.proj.weight",
    "self_attention.linear_proj.bias": "attn.proj.bias",
    "self_attention.linear_qkv.layer_norm_weight": "norm1.weight",
    "self_attention.linear_qkv.layer_norm_bias": "norm1.bias",
    "mlp.linear_fc1.layer_norm_weight": "norm2.weight",
    "mlp.linear_fc1.layer_norm_bias": "norm2.bias",
    "mlp.linear_fc1.weight": "mlp.linear_fc1.weight",
    "mlp.linear_fc1.bias": "mlp.linear_fc1.bias",
    "mlp.linear_fc2.weight": "mlp.linear_fc2.weight",
    "mlp.linear_fc2.bias": "mlp.linear_fc2.bias",
}


def convert_qwen3vl_vision_to_hf(args: Namespace, name: str, param: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """The caller has already gathered TP shards; undo native QKV
    interleaving."""
    suffix = name.removeprefix("module.module.vision_model.")
    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", suffix)
    if match:
        layer, local = match.groups()
        prefix = f"model.visual.blocks.{layer}."
        if local in _LAYER_NAMES:
            return [(prefix + _LAYER_NAMES[local], param)]
        if local in ("self_attention.linear_qkv.weight", "self_attention.linear_qkv.bias"):
            from relax.utils.misc import get_hf_config

            config = get_hf_config(args.hf_checkpoint).vision_config
            heads = config.num_heads
            width = config.hidden_size // heads
            if config.hidden_size % heads or param.shape[0] != 3 * config.hidden_size:
                raise ValueError("Qwen3-VL vision QKV shape does not match its attention heads")
            # Native rows are [q_head0, k_head0, v_head0, q_head1, ...].
            # HF stores all Q heads, then all K heads, then all V heads.
            value = param.reshape(heads, 3, width, *param.shape[1:]).transpose(0, 1).reshape_as(param)
            return [(prefix + "attn.qkv." + local.rsplit(".", 1)[1], value)]
        raise ValueError(f"Unknown native Qwen3-VL vision parameter: {name}")
    if suffix.startswith("decoder.deepstack_merger_list."):
        suffix = suffix.removeprefix("decoder.").replace(".patch_norm.", ".norm.")
    elif suffix.startswith("merger.patch_norm."):
        suffix = suffix.replace("merger.patch_norm.", "merger.norm.", 1)
    elif suffix.startswith("decoder."):
        raise ValueError(f"Unknown native Qwen3-VL vision parameter: {name}")
    # HF-backed towers already use blocks/deepstack_merger_list/merger.norm;
    # patch embedding, position embedding and merger linears share both names.
    return [("model.visual." + suffix, param)]
