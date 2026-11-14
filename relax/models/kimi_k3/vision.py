# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Trainable MoonViT for K3.1 checkpoints without HF modeling code.

Implements the K3/K3.1 vision architecture with the checkpoint's original
parameter layout. Spatial positions and attention are independent per media
item; video frames attend jointly and are pooled before the projector.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _PositionEmbedding(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.init_pos_emb_height, config.init_pos_emb_width, config.vt_hidden_size)
        )
        nn.init.normal_(self.weight)
        self.interpolation_mode = getattr(config, "pos_emb_interpolation_mode", "bilinear")
        self.num_frames = config.init_pos_emb_time
        frequencies = 10000 ** (
            -torch.arange(config.vt_hidden_size // 2, dtype=torch.float32) / (config.vt_hidden_size / 2)
        )
        angles = torch.arange(self.num_frames, dtype=torch.float32)[:, None] * frequencies
        self.register_buffer("time_weight", torch.cat((angles.sin(), angles.cos()), dim=-1)[:, None], persistent=False)

    def forward(self, patches: Tensor, grids: list[list[int]]) -> Tensor:
        positions = []
        for frames, height, width in grids:
            if frames > self.num_frames:
                raise ValueError(f"MoonViT supports at most {self.num_frames} frames per media item, got {frames}.")
            if (height, width) == self.weight.shape[:2]:
                spatial = self.weight.flatten(0, 1)
            else:
                spatial = (
                    F.interpolate(
                        self.weight.permute(2, 0, 1)[None], size=(height, width), mode=self.interpolation_mode
                    )[0]
                    .permute(1, 2, 0)
                    .flatten(0, 1)
                )
            if frames > 1:
                spatial = spatial[None].repeat(frames, 1, 1) + self.time_weight[:frames]
            positions.append(spatial.reshape(-1, patches.shape[-1]))
        return patches + torch.cat(positions)


class _PatchEmbedding(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            3,
            config.vt_hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=getattr(config, "patch_embed_proj_bias", True),
        )
        self.pos_emb = _PositionEmbedding(config)

    def forward(self, pixels: Tensor, grids: list[list[int]]) -> Tensor:
        return self.pos_emb(self.proj(pixels).flatten(1), grids)


def _norm(config: Any) -> nn.Module:
    norm_type = getattr(config, "norm_type", "layernorm")
    if norm_type == "rmsnorm":
        return nn.RMSNorm(config.vt_hidden_size)
    if norm_type == "layernorm":
        return nn.LayerNorm(config.vt_hidden_size)
    raise ValueError(f"Unsupported MoonViT norm: {norm_type}.")


class _MLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        bias = getattr(config, "linear_bias", True)
        self.fc0 = nn.Linear(config.vt_hidden_size, config.vt_intermediate_size, bias=bias)
        self.fc1 = nn.Linear(config.vt_intermediate_size, config.vt_hidden_size, bias=bias)
        for layer in (self.fc0, self.fc1):
            nn.init.trunc_normal_(layer.weight, std=math.sqrt(2 / layer.in_features))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.fc1(F.gelu(self.fc0(hidden_states), approximate="tanh"))


def _rotary_frequencies(grids: list[list[int]], head_dim: int, device: torch.device) -> Tensor:
    frequency = 1 / (10000 ** (torch.arange(0, head_dim, 4, device=device).float() / head_dim))
    rotations = []
    for frames, height, width in grids:
        positions = torch.arange(height * width, device=device).float()
        x_angles = (positions % width)[:, None] * frequency
        y_angles = (positions // width)[:, None] * frequency
        angles = torch.stack((x_angles, y_angles), dim=-1).flatten(1)
        rotations.append(torch.polar(torch.ones_like(angles), angles).repeat(frames, 1))
    return torch.cat(rotations)[:, None]


class _EncoderLayer(nn.Module):
    def __init__(self, config: Any, attn_implementation: str) -> None:
        super().__init__()
        self.num_heads = config.vt_num_attention_heads
        qkv_width = getattr(config, "qkv_hidden_size", None) or config.vt_hidden_size
        self.head_dim = qkv_width // self.num_heads
        if qkv_width % self.num_heads or self.head_dim % 4:
            raise ValueError("MoonViT QKV width must divide its head count with a head dimension divisible by four.")
        self.attn_implementation = attn_implementation
        self.norm0 = _norm(config)
        self.norm1 = _norm(config)
        self.mlp = _MLP(config)
        self.wqkv = nn.Linear(config.vt_hidden_size, 3 * qkv_width, bias=getattr(config, "attn_bias", True))
        self.wo = nn.Linear(qkv_width, config.vt_hidden_size, bias=getattr(config, "attn_bias", True))

    def forward(self, hidden_states: Tensor, rotations: Tensor, lengths: list[int], cu_seqlens: Tensor) -> Tensor:
        qkv = self.wqkv(self.norm0(hidden_states)).view(-1, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(1)
        query = (
            torch.view_as_real(torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2)) * rotations)
            .flatten(-2)
            .to(query.dtype)
        )
        key = (
            torch.view_as_real(torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2)) * rotations)
            .flatten(-2)
            .to(key.dtype)
        )
        if self.attn_implementation == "flash_attention_2":
            from flash_attn import flash_attn_varlen_func

            attention = flash_attn_varlen_func(
                query, key, value, cu_seqlens, cu_seqlens, max(lengths), max(lengths), causal=False
            ).flatten(1)
        else:
            # Isolate media items without allocating a mask over the whole batch.
            outputs = []
            for q, k, v in zip(query.split(lengths), key.split(lengths), value.split(lengths), strict=True):
                scores = q.transpose(0, 1) @ k.transpose(0, 1).transpose(-2, -1) / math.sqrt(self.head_dim)
                probabilities = scores.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
                outputs.append((probabilities @ v.transpose(0, 1)).transpose(0, 1).flatten(1))
            attention = torch.cat(outputs)
        hidden_states = hidden_states + self.wo(attention)
        return hidden_states + self.mlp(self.norm1(hidden_states))


class _Encoder(nn.Module):
    def __init__(self, config: Any, attn_implementation: str) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_EncoderLayer(config, attn_implementation) for _ in range(config.vt_num_hidden_layers)]
        )
        self.final_layernorm = _norm(config)
        qkv_width = getattr(config, "qkv_hidden_size", None) or config.vt_hidden_size
        self.head_dim = qkv_width // config.vt_num_attention_heads

    def forward(self, hidden_states: Tensor, grids: list[list[int]]) -> Tensor:
        lengths = [frames * height * width for frames, height, width in grids]
        rotations = _rotary_frequencies(grids, self.head_dim, hidden_states.device)
        cu_seqlens = torch.tensor([0, *lengths], device=hidden_states.device, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        for block in self.blocks:
            hidden_states = block(hidden_states, rotations, lengths, cu_seqlens)
        return self.final_layernorm(hidden_states)


class MoonViT(nn.Module):
    def __init__(self, config: Any, attn_implementation: str = "eager") -> None:
        super().__init__()
        if config.pos_emb_type != "divided_fixed" or config.merge_type != "sd2_tpool":
            raise ValueError("Kimi K3.1 MoonViT requires divided_fixed positions and sd2_tpool merging.")
        if getattr(config, "mlp_type", "mlp2") != "mlp2":
            raise ValueError("Kimi K3.1 MoonViT requires mlp2 blocks.")
        if attn_implementation not in ("eager", "flash_attention_2"):
            raise ValueError(f"Unsupported MoonViT attention implementation: {attn_implementation}.")
        self.merge_kernel_size = tuple(config.merge_kernel_size)
        self.patch_embed = _PatchEmbedding(config)
        self.encoder = _Encoder(config, attn_implementation)

    def forward(self, pixel_values: Tensor, grid_thws: Tensor) -> list[Tensor]:
        if grid_thws.device.type != "cpu":
            raise ValueError("Kimi K3.1 vision requires CPU grid_thws metadata to avoid GPU synchronization.")
        if grid_thws.ndim != 2 or grid_thws.shape[1] != 3:
            raise ValueError("MoonViT grid_thws must have shape [media, 3].")
        grids = grid_thws.tolist()
        merge_h, merge_w = self.merge_kernel_size
        if not grids or any(t <= 0 or h <= 0 or w <= 0 or h % merge_h or w % merge_w for t, h, w in grids):
            raise ValueError("MoonViT grids must be nonempty, positive, and divisible by the spatial merge kernel.")
        if sum(t * h * w for t, h, w in grids) != pixel_values.shape[0]:
            raise ValueError("MoonViT pixel patch count does not match grid_thws.")
        hidden_states = self.encoder(self.patch_embed(pixel_values, grids), grids)
        features = []
        offset = 0
        for frames, height, width in grids:
            count = frames * height * width
            patches = hidden_states[offset : offset + count].view(
                frames, height // merge_h, merge_h, width // merge_w, merge_w, -1
            )
            pooled = patches.permute(0, 1, 3, 2, 4, 5).contiguous().mean(0)
            features.append(pooled.reshape(height // merge_h * (width // merge_w), merge_h * merge_w, -1))
            offset += count
        return features


class PatchMergerMLPV2(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden_size = config.mm_hidden_size * math.prod(config.merge_kernel_size)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.text_hidden_size, bias=False),
        )
        self.post_norm = nn.RMSNorm(config.text_hidden_size, eps=config.projector_ln_eps)
        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=math.sqrt(2 / layer.in_features))

    def forward(self, features: list[Tensor] | tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        lengths = [feature.shape[0] for feature in features]
        flattened = torch.cat([feature.reshape(feature.shape[0], self.hidden_size) for feature in features])
        return self.post_norm(self.proj(flattened)).split(lengths)
