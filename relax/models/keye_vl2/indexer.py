# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keye sparse indexer and contiguous three-axis rotary embeddings."""

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(x.dtype)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).to(
            x.dtype
        )


def text_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Equal text axes reduce the original contiguous mRoPE sections to
    ordinary half rotation."""
    dim = x.shape[-1]
    frequencies = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    if positions.ndim == x.ndim - 2:
        angles = positions.float().unsqueeze(-1) * frequencies
    else:
        # Coordinates are [tokens, 3]; each axis owns a contiguous section.
        sections = [dim // 8, dim * 3 // 16, dim * 3 // 16]
        axis_angles = positions.float().unsqueeze(-1) * frequencies
        angles = torch.cat([part[..., axis, :] for axis, part in enumerate(axis_angles.split(sections, -1))], -1)
    angles = torch.cat((angles, angles), -1).unsqueeze(-2)
    first, second = x.chunk(2, -1)
    return x * angles.cos().to(x.dtype) + torch.cat((-second, first), -1) * angles.sin().to(x.dtype)


class KeyeIndexer(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, theta: float = 1e7) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, 16 * 64 + 16, bias=False)
        self.k_proj = nn.Linear(hidden_size, 64, bias=False)
        self.q_norm = RMSNorm(64, eps)
        self.k_norm = LayerNorm(64, eps=eps)
        self.theta = theta

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        from fast_hadamard_transform import hadamard_transform

        projected = self.q_proj(hidden.detach())
        query, weights = projected.split((1024, 16), -1)
        query = self.q_norm(query.reshape(*hidden.shape[:-1], 16, 64))
        key = self.k_norm(self.k_proj(hidden.detach()).unsqueeze(-2))
        query = text_rope(query, positions, self.theta)
        key = text_rope(key, positions, self.theta)
        return (
            hadamard_transform(query.contiguous(), scale=0.125),
            hadamard_transform(key.contiguous(), scale=0.125),
            weights,
        )


def indexer_scores(query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Match the public BF16 reference's matmul, scale, ReLU, and head
    reduction."""
    scores = torch.einsum("qhd,kd->hqk", query, key.squeeze(-2)) * 0.125
    return (scores.relu() * weights.transpose(0, 1).unsqueeze(-1)).sum(0)
