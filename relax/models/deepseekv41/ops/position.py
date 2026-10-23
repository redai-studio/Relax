# Modified for Relax integration; see NOTICE.

import math
from functools import lru_cache
from typing import Any

import torch
from torch.distributed.nn.functional import all_gather


@lru_cache(maxsize=32)
def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    device: torch.device,
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def correction(rotations: float) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(correction(beta_fast)), 0)
        high = min(math.ceil(correction(beta_slow)), dim - 1)
        upper = high + 0.001 if low == high else high
        smooth = 1 - ((torch.arange(dim // 2, dtype=torch.float32) - low) / (upper - low)).clamp(0, 1)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    angles = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(angles), angles).to(device)


def attention_freqs(config: Any, seqlen: int, compressed: bool, device: torch.device) -> torch.Tensor:
    return precompute_freqs_cis(
        config.qk_pos_emb_head_dim,
        seqlen,
        config.original_max_position_embeddings if compressed else 0,
        config.csa_compress_rotary_base if compressed else config.rotary_base,
        config.rotary_scaling_factor,
        config.beta_fast,
        config.beta_slow,
        device,
    )


def apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    from .kernels.rope import apply_rotary_emb as rotate

    return rotate(x, freqs, inverse)


def all_gather_cp(tensor: torch.Tensor, dim: int, cp_group: torch.distributed.ProcessGroup) -> torch.Tensor:
    # Never cache activation tensors: their graphs belong to one microbatch only.
    return torch.cat(all_gather(tensor, group=cp_group), dim=dim)


def cp_positions(seqlen: int, cp_group: torch.distributed.ProcessGroup | None, device: torch.device) -> torch.Tensor:
    rank = torch.distributed.get_rank(cp_group) if cp_group is not None else 0
    return torch.arange(rank * seqlen, (rank + 1) * seqlen, device=device)


def cp_freqs(
    freqs: torch.Tensor, seqlen: int, cp_group: torch.distributed.ProcessGroup | None, stride: int = 1
) -> torch.Tensor:
    rank = torch.distributed.get_rank(cp_group) if cp_group is not None else 0
    return freqs[rank * seqlen : (rank + 1) * seqlen : stride]


def window_indices(positions: torch.Tensor, window: int, global_seqlen: int, batch: int) -> torch.Tensor:
    current = positions.unsqueeze(1)
    keys = (current - window + 1).clamp_min(0) + torch.arange(min(global_seqlen, window), device=positions.device)
    return torch.where(keys > current, -1, keys).int().unsqueeze(0).expand(batch, -1, -1)
