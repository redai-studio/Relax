# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Vision operations adapted from DeepSeek-V4.1-Flash-1017 (MIT). See NOTICE.

"""Frozen checkpoint-backed vision input for the first pipeline stage."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@lru_cache(maxsize=4)
def _weight_map(checkpoint: str) -> dict[str, str]:
    with (Path(checkpoint) / "model.safetensors.index.json").open() as source:
        return json.load(source)["weight_map"]


def load_frozen_weights(checkpoint: str, names: Collection[str], device: torch.device) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    index = _weight_map(checkpoint)
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in names:
        if name not in index:
            raise ValueError(f"Missing frozen DeepSeek V4.1 tensor {name!r} in {checkpoint}.")
        by_shard[index[name]].append(name)
    weights = {}
    for shard, shard_names in by_shard.items():
        with safe_open(Path(checkpoint) / shard, framework="pt", device="cpu") as source:
            for name in shard_names:
                weights[name] = source.get_tensor(name).to(device=device)
    return weights


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    num_hidden_layers: int
    patch_size: int
    rope_theta: float
    downsample_ratio: int

    @classmethod
    def from_hf(cls, config: dict[str, Any]) -> VisionConfig:
        return cls(**{key: config[key] for key in cls.__dataclass_fields__})


class VisionRMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return (self.weight.float() * (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + 1e-6))).to(x.dtype)


class PatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(3 * config.patch_size**2, config.hidden_size)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(patches.flatten(1))


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size)

    @staticmethod
    def _rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        first, second = x.float().chunk(2, dim=-1)
        return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(x.dtype)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        q, k, v = (t.reshape(len(x), self.heads, -1) for t in self.wqkv(x).chunk(3, dim=-1))
        q = self._rotary(q, cos, sin).transpose(0, 1)
        k = self._rotary(k, cos, sin).transpose(0, 1)
        v = v.transpose(0, 1)
        # Keep the reference's 3D math SDPA precision. Each query still attends to
        # the entire image. Tiling queries bounds the FP32 softmax workspace when
        # the processor supplies up to ~9k ViT patches for a single image.
        if len(x) <= 512:
            output = F.scaled_dot_product_attention(q, k, v)
        else:
            output = torch.cat([F.scaled_dot_product_attention(rows, k, v) for rows in q.split(512, dim=1)], dim=1)
        return self.wo(output.transpose(0, 1).reshape_as(x))


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = VisionRMSNorm(config.hidden_size)
        self.attn = VisionAttention(config)
        self.norm2 = VisionRMSNorm(config.hidden_size)
        self.mlp = VisionMLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        self.blocks = nn.ModuleList(VisionBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = VisionRMSNorm(config.hidden_size)

    def forward(self, patches: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.patch_embed(patches.to(self.patch_embed.proj.weight))
        dim = self.config.hidden_size // self.config.num_attention_heads // 2
        inv_freq = 1.0 / (self.config.rope_theta ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
        y, z = torch.meshgrid(
            torch.arange(height, device=x.device), torch.arange(width, device=x.device), indexing="ij"
        )
        coords = torch.stack((y, z), dim=-1).reshape(-1, 2, 1).float()
        phase = (coords * inv_freq).flatten(1).unsqueeze(1)
        cos, sin = phase.cos(), phase.sin()
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class VisionAligner(nn.Module):
    def __init__(self, config: VisionConfig, language_width: int) -> None:
        super().__init__()
        self.ratio = config.downsample_ratio
        self.w1 = nn.Linear(config.hidden_size * self.ratio**2, language_width)
        self.w2 = nn.Linear(language_width, language_width)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x.reshape(height, width, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -width % self.ratio, 0, -height % self.ratio))
        rows = F.unfold(x.unsqueeze(0), self.ratio, stride=self.ratio).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(rows)))


class VisionInput(nn.Module):
    def __init__(self, config: VisionConfig, language_width: int) -> None:
        super().__init__()
        self.vision = VisionTransformer(config)
        self.aligner = VisionAligner(config, language_width)
        self.image_start = nn.Parameter(torch.empty(language_width))
        self.image_end = nn.Parameter(torch.empty(language_width))
        self.image_newline = nn.Parameter(torch.empty(language_width))

    @torch.no_grad()
    def forward(self, patches: torch.Tensor, height: int, width: int) -> torch.Tensor:
        ratio = self.aligner.ratio
        rows = self.aligner(self.vision(patches, height, width), height, width)
        rows = rows.reshape(-(-height // ratio), -(-width // ratio), -1)
        rows = torch.cat((rows, self.image_newline.expand(len(rows), 1, -1)), dim=1)
        return torch.cat((self.image_start[None], rows.flatten(0, 1), self.image_end[None]), dim=0)


class FrozenVisionInput:
    """Read-only base assets, independent of the trainable module and its optimizer/DCP state."""

    def __init__(self, checkpoint: str, config: dict[str, Any], language_width: int) -> None:
        self.checkpoint = checkpoint
        self.config = VisionConfig.from_hf(config)
        self.language_width = language_width
        self.module: VisionInput | None = None

    def release(self) -> None:
        self.module = None

    def _on_device(self, device: torch.device) -> VisionInput:
        if self.module is None or self.module.image_start.device != device:
            with torch.device("meta"):
                module = VisionInput(self.config, self.language_width)
            # assign=True preserves checkpoint dtypes, including FP32 norm weights.
            weights = load_frozen_weights(self.checkpoint, module.state_dict().keys(), device)
            module.load_state_dict(weights, strict=True, assign=True)
            self.module = module.requires_grad_(False).eval()
        return self.module

    def merge(
        self,
        embeddings: torch.Tensor,
        pixel_values: torch.Tensor,
        grids: Sequence[tuple[int, int]],
        spans: Sequence[tuple[int, int]],
        global_start: int,
    ) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[1] != 1 or len(spans) != len(grids):
            raise ValueError("DeepSeek V4.1 image spans require packed singleton-batch embeddings and one grid each.")
        expected_shape = (sum(h * w for h, w in grids), 3, self.config.patch_size, self.config.patch_size)
        if tuple(pixel_values.shape) != expected_shape:
            raise ValueError(f"DeepSeek V4.1 image grids require patches of shape {expected_shape}.")
        parts = []
        patch_start = previous_end = local_pos = 0
        for (start, length), (height, width) in zip(spans, grids, strict=True):
            ratio = self.config.downsample_ratio
            expected_length = -(-height // ratio) * (-(-width // ratio) + 1) + 2
            if min(height, width) <= 0 or start < previous_end or length != expected_length:
                raise ValueError("DeepSeek V4.1 image spans must be ordered and match the downsampled grids.")
            previous_end = start + length
            low, high = max(global_start, start), min(global_start + len(embeddings), previous_end)
            if low < high:
                module = self._on_device(embeddings.device)
                features = module(pixel_values[patch_start : patch_start + height * width], height, width)
                replacement = features[low - start : high - start, None].to(embeddings.dtype)
                parts.extend((embeddings[local_pos : low - global_start], replacement))
                local_pos = high - global_start
            patch_start += height * width
        if not parts:
            return embeddings
        return torch.cat((*parts, embeddings[local_pos:]), dim=0)
