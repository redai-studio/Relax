# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re

import torch

from relax.utils.megatron_argument_compat import resolve_kv_channels


def convert_qwen3omni_to_hf(args, name, param):
    if ".audio_model." in name:
        name = name.replace("module.module.thinker.audio_model.", "thinker.audio_tower.")
        return [(name, param)]
    if ".vision_model." in name:
        name = name.replace("module.module.thinker.vision_model.", "thinker.visual.")
        return [(name, param)]
    if ".video_model." in name:
        name = name.replace("module.module.thinker.video_model.", "thinker.video_tower.")
        return [(name, param)]
    if name == "module.module.thinker.language_model.embedding.word_embeddings.weight":
        return [("thinker.model.embed_tokens.weight", param)]
    if name == "module.module.thinker.language_model.output_layer.weight":
        return [("thinker.lm_head.weight", param)]
    if name == "module.module.thinker.language_model.decoder.final_layernorm.weight":
        return [("thinker.model.norm.weight", param)]

    head_dim = resolve_kv_channels(args)
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.thinker\.language_model\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # experts
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                outputs = [
                    (f"thinker.model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"thinker.model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
                return outputs
            elif rest == "linear_fc2":
                outputs = [
                    (f"thinker.model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param),
                ]
                return outputs
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
                    (f"thinker.model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight", gate_weight),
                    (f"thinker.model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                return [(f"thinker.model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight", param)]
            elif rest == "gate_weight":
                return [(f"thinker.model.layers.{layer_idx}.mlp.shared_expert_gate.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_proj.weight":
            return [(f"thinker.model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"thinker.model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"thinker.model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"thinker.model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
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
                (f"thinker.model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"thinker.model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"thinker.model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"thinker.model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"thinker.model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"thinker.model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"thinker.model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"thinker.model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"thinker.model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "mlp.router.weight":
            return [(f"thinker.model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"thinker.model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"thinker.model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"thinker.model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
