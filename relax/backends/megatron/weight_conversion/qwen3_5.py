# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re

import torch

from relax.utils.megatron_argument_compat import resolve_kv_channels
from relax.utils.misc import get_hf_config


def _convert_mtp_layer(args, name, param, layer_idx):
    """Convert MTP layer parameters from Megatron to HuggingFace format."""
    if "enorm.weight" in name:
        return [("mtp.pre_fc_norm_embedding.weight", param)]
    if "hnorm.weight" in name:
        return [("mtp.pre_fc_norm_hidden.weight", param)]
    if "final_layernorm.weight" in name:
        return [("mtp.norm.weight", param)]
    if "eh_proj.weight" in name:
        if param.dim() < 2:
            raise ValueError(f"eh_proj weight expects 2D tensor, got {param.shape}")
        first_half, second_half = param.chunk(2, dim=1)
        new_param = torch.cat([second_half, first_half], dim=1)
        return [("mtp.fc.weight", new_param)]

    if "transformer_layer" in name:
        proxy_name = name.replace(f"mtp.layers.{layer_idx}.transformer_layer", f"decoder.layers.{layer_idx}")
        mapped_params = convert_qwen3_5_to_hf(args, proxy_name, param)

        final_params = []
        for hf_name, tensor in mapped_params:
            target_prefix = f"mtp.layers.{layer_idx}"
            if f"model.language_model.layers.{layer_idx}" in hf_name:
                new_hf_name = hf_name.replace(f"model.language_model.layers.{layer_idx}", target_prefix)
                final_params.append((new_hf_name, tensor))
            else:
                final_params.append((hf_name, tensor))
        return final_params

    return None


def gdn_hf_to_mca(config, weights):
    qkv_weight, z_weight, b_weight, a_weight = weights
    qk_head_dim = config.linear_key_head_dim
    v_head_dim = config.linear_value_head_dim
    num_qk_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads

    q, k, v = torch.split(
        qkv_weight,
        [
            qk_dim,
            qk_dim,
            v_dim,
        ],
        dim=0,
    )
    z = z_weight.reshape(v_dim, -1)
    b = b_weight.reshape(num_v_heads, -1)
    a = a_weight.reshape(num_v_heads, -1)
    return torch.cat([q, k, v, z, b, a], dim=0)


def gdn_mca_to_hf(config, weight):
    """
    Args:
        args: 包含 hf_checkpoint 路径的参数
        weight: hf_to_mca 输出的拼接张量，shape=(qk_dim*2 + v_dim + v_dim + num_v_heads*2, hidden_size)
    Returns:
        (qkv_weight, z_weight, b_weight, a_weight)：与 hf_to_mca 输入等价的四元组
    """
    qk_head_dim = config.linear_key_head_dim
    v_head_dim = config.linear_value_head_dim
    num_qk_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    hidden_size = config.hidden_size
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads
    q, k, v, z, b, a = torch.split(
        weight,
        [qk_dim, qk_dim, v_dim, v_dim, num_v_heads, num_v_heads],
        dim=0,
    )
    qkv_weight = torch.cat([q, k, v], dim=0)  # (qk_dim*2 + v_dim, hidden_size)
    z_weight = z.reshape(v_dim, hidden_size)  # 还原 z_weight 原始 shape
    b_weight = b.reshape(num_v_heads, hidden_size)  # 还原 b_weight 原始 shape
    a_weight = a.reshape(num_v_heads, hidden_size)  # 还原 a_weight 原始 shape
    return qkv_weight, z_weight, b_weight, a_weight


def split_qkv_weights(config, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Megatron's interleaved QKV tensor into separate Q, K, V matrices.

    Args:
        config: transformer config
        qkv (torch.Tensor): Interleaved QKV weights in Megatron format.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of (Q, K, V)
            weight matrices.
    """
    head_num = config.num_heads
    heads_per_group = 1
    num_query_groups = head_num // heads_per_group
    head_size = config.hidden_size // head_num
    qkv_total_dim = head_num + 2 * num_query_groups
    total_heads_per_group = heads_per_group + 2
    is_bias = qkv.ndim == 1

    if is_bias:
        hidden_size = 1
        qkv_reshaped = qkv.view(qkv_total_dim, head_size)
    else:
        hidden_size = qkv.shape[-1]
        qkv_reshaped = qkv.view(qkv_total_dim, head_size, hidden_size)

    # Extract Q, K, V from interleaved pattern
    q_slice = torch.cat(
        [
            torch.arange(total_heads_per_group * i, total_heads_per_group * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(total_heads_per_group - 2, qkv_total_dim, total_heads_per_group)
    v_slice = torch.arange(total_heads_per_group - 1, qkv_total_dim, total_heads_per_group)
    q = qkv_reshaped[q_slice]
    k = qkv_reshaped[k_slice]
    v = qkv_reshaped[v_slice]

    assert q.numel() + k.numel() + v.numel() == qkv.numel(), (
        f"QKV weights are not correctly merged, {q.shape=}, {k.shape=}, {v.shape=}, {qkv.shape=}"
    )

    if is_bias:
        q = q.reshape(-1)
        k = k.reshape(-1)
        v = v.reshape(-1)
    else:
        q = q.reshape(-1, hidden_size)
        k = k.reshape(-1, hidden_size)
        v = v.reshape(-1, hidden_size)

    return q, k, v


def convert_qwen3_5_vision_to_hf(args, name: str, param: torch.Tensor) -> list:
    config = get_hf_config(args.hf_checkpoint).vision_config
    # 'module.module.vision_model.decoder.layers.0.self_attention.linear_qkv.weight'
    if name.startswith("module.module.vision_model"):
        name = "vision_model." + name[len("module.module.vision_model.") :]

    while name.startswith("module."):
        name = name.replace("module.", "", 1)

    if name.startswith("vision_model.merger.patch_norm."):
        return [("model.visual.merger.norm." + name[len("vision_model.merger.patch_norm.") :], param)]

    for k in [
        "vision_model.patch_embed.proj.",
        "vision_model.pos_embed.",
        "vision_model.merger.linear_fc1.",
        "vision_model.merger.linear_fc2.",
    ]:
        if name.startswith(k):
            return [(name.replace("vision_model.", "model.visual."), param)]

    decoder_layers_pattern = r"vision_model\.decoder\.layers\.(\d+)\.(.+)"
    m = re.match(decoder_layers_pattern, name)
    if m:
        layer_idx, rest = m.groups()
        prefix = f"model.visual.blocks.{layer_idx}"

        if rest == "self_attention.linear_proj.weight":
            return [(f"{prefix}.attn.proj.weight", param)]
        if rest == "self_attention.linear_proj.bias":
            return [(f"{prefix}.attn.proj.bias", param)]

        if rest == "mlp.linear_fc1.weight":
            return [(f"{prefix}.mlp.linear_fc1.weight", param)]
        if rest == "mlp.linear_fc1.bias":
            return [(f"{prefix}.mlp.linear_fc1.bias", param)]
        if rest == "mlp.linear_fc2.weight":
            return [(f"{prefix}.mlp.linear_fc2.weight", param)]
        if rest == "mlp.linear_fc2.bias":
            return [(f"{prefix}.mlp.linear_fc2.bias", param)]

        if rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"{prefix}.norm1.weight", param)]
        if rest == "self_attention.linear_qkv.layer_norm_bias":
            return [(f"{prefix}.norm1.bias", param)]

        if rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"{prefix}.norm2.weight", param)]
        if rest == "mlp.linear_fc1.layer_norm_bias":
            return [(f"{prefix}.norm2.bias", param)]

        if rest == "self_attention.linear_qkv.weight":
            q, k, v = split_qkv_weights(config, param)
            return [(f"{prefix}.attn.qkv.weight", torch.cat((q, k, v), dim=0))]
        if rest == "self_attention.linear_qkv.bias":
            q, k, v = split_qkv_weights(config, param)
            return [(f"{prefix}.attn.qkv.bias", torch.cat((q, k, v), dim=0))]

    raise ValueError(f"Unknown parameter name: {name}")


def convert_qwen3_5_to_hf(args, name, param):
    """Convert Qwen3.5 model parameters from Megatron to HuggingFace format.

    Qwen3.5 uses model.language_model.layers prefix and has separate
    in_proj_qkv, in_proj_z, in_proj_b, in_proj_a for linear attention.
    """
    # Handle MTP layers
    if "mtp.layers" in name:
        parts = name.split(".")
        try:
            layer_idx_loc = parts.index("layers") + 1
            layer_idx = parts[layer_idx_loc]
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid MTP layer name format: {name}") from e

        result = _convert_mtp_layer(args, name, param, layer_idx)
        if result is not None:
            return result

    if name.startswith("module.module.language_model."):
        name = "module.module." + name[len("module.module.language_model.") :]

    # (Optional safety) if you ever see extra "module." prefixes
    while name.startswith("module.module.module."):
        name = name.replace("module.module.module.", "module.module.", 1)

    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    if name.startswith("module.module.vision_model."):
        return convert_qwen3_5_vision_to_hf(args, name, param)

    head_dim = resolve_kv_channels(args)
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        prefix = f"model.language_model.layers.{layer_idx}"

        # experts (grouped gemm - fused format)
        if rest == "mlp.experts.linear_fc1":
            return [(f"{prefix}.mlp.experts.gate_up_proj", param)]
        elif rest == "mlp.experts.linear_fc2":
            return [(f"{prefix}.mlp.experts.down_proj", param)]

        # experts (ungrouped - individual expert format)
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2":
                return [(f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight", param)]
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # shared expert
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"{prefix}.mlp.shared_expert.gate_proj.weight", gate_weight),
                    (f"{prefix}.mlp.shared_expert.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                return [(f"{prefix}.mlp.shared_expert.down_proj.weight", param)]
            elif rest == "gate_weight":
                return [(f"{prefix}.mlp.shared_expert_gate.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_proj.weight":
            return [(f"{prefix}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(
                param, split_size_or_sections=[2 * value_num_per_group, 1, 1], dim=1
            )
            q_param = (
                q_param.reshape(args.num_query_groups, 2, value_num_per_group, head_dim, args.hidden_size)
                .transpose(1, 2)
                .reshape(-1, args.hidden_size)
            )
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"{prefix}.self_attn.q_proj.weight", q_param),
                (f"{prefix}.self_attn.k_proj.weight", k_param),
                (f"{prefix}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"{prefix}.self_attn.q_proj.bias", q_bias),
                (f"{prefix}.self_attn.k_proj.bias", k_bias),
                (f"{prefix}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"{prefix}.mlp.gate_proj.weight", gate_weight),
                (f"{prefix}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"{prefix}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.in_proj.layer_norm_weight":
            return [(f"{prefix}.input_layernorm.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"{prefix}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"{prefix}.post_attention_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"{prefix}.post_attention_layernorm.weight", param)]
        elif rest == "mlp.router.weight":
            return [(f"{prefix}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"{prefix}.mlp.gate.e_score_correction_bias", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"{prefix}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"{prefix}.self_attn.k_norm.weight", param)]

        # linear attention
        elif rest == "self_attention.out_norm.weight":
            # RMSNorm2ZeroCenteredRMSNormMapping
            # https://github.com/coding-famer/Megatron-Bridge-slime/blob/a069bdffa41fdc4d7cb377e72a89d484167b8a85/src/megatron/bridge/models/conversion/param_mapping.py#L2194
            return [(f"{prefix}.linear_attn.norm.weight", param + 1)]

        if rest.startswith("self_attention.") and rest[len("self_attention.") :] in [
            # linear attn (Qwen3.5 uses separate in_proj_b/in_proj_a)
            "A_log",
            "out_proj.weight",
            "dt_bias",
        ]:
            rest = "linear_attn." + rest[len("self_attention.") :]
            return [(f"{prefix}.{rest}", param)]

        if rest == "self_attention.conv1d.weight":
            # GDNConv1dMapping, params are already gathered across TP in `all_gather_param`
            # https://github.com/coding-famer/Megatron-Bridge-slime/blob/a069bdffa41fdc4d7cb377e72a89d484167b8a85/src/megatron/bridge/models/conversion/param_mapping.py#L1689
            return [(f"{prefix}.linear_attn.conv1d.weight", param)]

        if rest == "self_attention.in_proj.weight":
            # GDNLinearMappingSeparate
            # https://github.com/coding-famer/Megatron-Bridge-slime/blob/a069bdffa41fdc4d7cb377e72a89d484167b8a85/src/megatron/bridge/models/conversion/param_mapping.py#L1816
            config = get_hf_config(args.hf_checkpoint).text_config
            qkv, z, b, a = gdn_mca_to_hf(config, param)

            return [
                (f"{prefix}.linear_attn.in_proj_qkv.weight", qkv),
                (f"{prefix}.linear_attn.in_proj_z.weight", z),
                (f"{prefix}.linear_attn.in_proj_b.weight", b),
                (f"{prefix}.linear_attn.in_proj_a.weight", a),
            ]

    raise ValueError(f"Unknown parameter name: {name}")
