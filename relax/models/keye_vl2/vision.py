# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared Keye vision weights and per-image forward for training and
rollout."""

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def build_vision(config: Any, model_path: str) -> tuple[nn.Module, nn.Module]:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    def load(name: str) -> type:
        return get_class_from_dynamic_module(f"modeling_keye_vl_2.{name}", model_path, trust_remote_code=True)

    vision_config = deepcopy(config.vision_config)
    vision_config.vision_use_head = False
    visual = load("KeyeVL2VisionModel")(vision_config)
    # The interpolated position path never uses this checkpoint parameter.
    visual.vision_model.embeddings.packing_position_embedding.requires_grad_(False)
    return visual, load("Projector")(config, vision_config)


def _vision_block(block: nn.Module, hidden: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Original noncausal ViT block using SDPA, with no device scalar reads."""
    attention = block.self_attn
    normalized = block.layer_norm1(hidden)
    shape = (*normalized.shape[:-1], attention.num_heads, attention.head_dim)
    query, key, value = [
        projection(normalized).view(shape) for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
    ]
    if attention.q_norm is not None:
        query, key = attention.q_norm(query), attention.k_norm(key)

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        first, second = tensor.float().chunk(2, -1)
        rotated = torch.cat((-second, first), -1)
        return (tensor.float() * angles.cos()[None, :, None] + rotated * angles.sin()[None, :, None]).to(tensor.dtype)

    query, key = rotate(query), rotate(key)
    output = (
        F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=attention.dropout if block.training else 0.0,
            scale=attention.scale,
        )
        .transpose(1, 2)
        .reshape_as(hidden)
    )
    hidden = hidden + attention.out_proj(output)
    return hidden + block.mlp(block.layer_norm2(hidden))


def image_features(visual: nn.Module, projector: nn.Module, pixels: torch.Tensor, grids: torch.Tensor) -> torch.Tensor:
    if grids.device.type != "cpu":
        raise ValueError("Keye image geometry must remain on CPU (map image_grid_thw to grid_thws for training)")
    tower = visual.vision_model
    parameter = next(tower.parameters())
    pixels = pixels.to(device=parameter.device, dtype=parameter.dtype)
    features = []
    start = 0
    for t, h, w in grids.tolist():
        if t != 1:
            raise ValueError("Keye integration currently supports images only")
        count = h * w
        ids = torch.arange(count, device=pixels.device)
        hidden = tower.embeddings(
            pixels[start : start + count].unsqueeze(0),
            position_ids=ids,
            image_grid_thw=torch.tensor([[t, h, w]]),
            interpolate_pos_encoding=True,
        )
        height, width = ids // w, ids % w
        rotary_module = tower.encoder.rotary_pos_emb
        # Float16Module casts registered buffers too; preserve the HF FP32
        # frequency construction independently of that wrapper.
        frequencies = 1.0 / (
            rotary_module.theta
            ** (torch.arange(0, rotary_module.dim, 2, device=pixels.device, dtype=torch.float32) / rotary_module.dim)
        )
        rotary = torch.outer(torch.arange(max(h, w), device=pixels.device, dtype=torch.float32), frequencies)
        angles = rotary[torch.stack((height, width), -1)].flatten(1).repeat(1, 2)
        for block in tower.encoder.layers:
            if torch.is_grad_enabled() and any(p.requires_grad for p in block.parameters()):
                hidden = checkpoint(_vision_block, block, hidden, angles, use_reentrant=False)
            else:
                hidden = _vision_block(block, hidden, angles)
        features.append(tower.post_layernorm(hidden).squeeze(0))
        start += count
    if start != pixels.shape[0]:
        raise ValueError("Keye pixel count does not match image grids")
    return projector(features, grids)
