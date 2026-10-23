# Modified for Relax integration; see NOTICE.

from typing import Any

import torch
from megatron.core.transformer.module import mark_keep_in_fp32
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        mark_keep_in_fp32(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * value).to(x.dtype)


class DeepSeekV41Compressor(nn.Module):
    def __init__(self, config: Any, head_dim: int, compress_ratio: int) -> None:
        super().__init__()
        if compress_ratio not in (1, 2):
            raise ValueError("DeepSeek V4.1 compressor requires ratio 1 or 2.")
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.linear_wkv = nn.Linear(config.hidden_size, head_dim, bias=False, dtype=config.params_dtype)
        if compress_ratio == 2:
            self.linear_wgate = nn.Linear(config.hidden_size, head_dim, bias=False, dtype=config.params_dtype)
        self.norm = RMSNorm(head_dim, config.layernorm_epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seqlen, batch, _ = x.shape
        if self.compress_ratio == 1:
            return self.norm(F.linear(x, self.linear_wkv.weight))
        groups = seqlen // self.compress_ratio
        x = x[: groups * self.compress_ratio].float()
        values = F.linear(x, self.linear_wkv.weight.float()).view(groups, self.compress_ratio, batch, self.head_dim)
        scores = F.linear(x, self.linear_wgate.weight.float()).view_as(values)
        return self.norm((values * scores.softmax(dim=1)).sum(dim=1).to(self.linear_wkv.weight.dtype))
