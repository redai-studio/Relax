# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in packed K3.1 expanded attention without per-query selected K/V copies.

The dense support mask retains sparse top-k semantics; projection, indexer and
autograd CP gather remain owned by the production attention layer. This trades
dense attention arithmetic for much smaller intermediates on bounded contexts.
"""

from functools import wraps
from typing import Any

import torch
from torch import Tensor


def packed_expanded_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    topk_indices: Tensor,
    softmax_scale: float,
    *,
    varlen_starts: Tensor,
    varlen_ends: Tensor,
    key_positions: Tensor | None = None,
    backend: str = "math",
) -> Tensor:
    """Apply packed sparse support to SBHD tensors using one explicit SDPA
    backend.

    Duplicate top-k slots retain their multiplicity via log-count bias.
    Negative indices are invalid; nonnegative indices clamp like Core's unfused
    reference. Fully masked rows return finite zero outputs and zero Q/K/V
    gradients.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention

    backends = {"math": SDPBackend.MATH, "efficient": SDPBackend.EFFICIENT_ATTENTION}
    selected = backends[backend]
    sq, batch = query.shape[:2]
    sk = key.shape[0]
    indices = topk_indices.to(device=query.device, dtype=torch.int64)
    counts = torch.zeros((batch, sq, sk), dtype=torch.int32, device=query.device)
    counts.scatter_add_(-1, indices.clamp(0, sk - 1), (indices >= 0).to(torch.int32))
    positions = torch.arange(sk, device=query.device) if key_positions is None else key_positions
    valid = counts > 0
    valid &= positions[None, None, :] >= varlen_starts[None, :, None]
    valid &= positions[None, None, :] < varlen_ends[None, :, None]
    bias_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    bias = counts.clamp_min(1).to(bias_dtype).log().masked_fill(~valid, float("-inf"))
    row_valid = valid.any(-1)
    # Keep every backend's softmax defined, then remove the dummy key's output
    # and gradient. No device-to-host synchronization is needed for this case.
    bias[..., 0] = torch.where(row_valid, bias[..., 0], torch.zeros_like(bias[..., 0]))
    if backend == "efficient":
        bias = bias.to(query.dtype)
    with sdpa_kernel([selected]):
        output = scaled_dot_product_attention(
            query.permute(1, 2, 0, 3),
            key.permute(1, 2, 0, 3),
            value.permute(1, 2, 0, 3),
            attn_mask=bias.unsqueeze(1),
            dropout_p=0.0,
            is_causal=False,
            scale=softmax_scale,
        )
    output = torch.where(row_valid[:, None, :, None], output, torch.zeros_like(output))
    return output.permute(2, 0, 1, 3).contiguous().flatten(-2)


def configure_expanded_attention(config: Any, backend: str) -> None:
    """Enable only marked K3.1 packed expanded calls; retain other Core
    paths."""
    from megatron.core.transformer.experimental_attention_variant import dsa

    from relax.utils.logging_utils import get_logger

    if backend not in ("math", "efficient"):
        raise ValueError(f"Unsupported K3.1 expanded DSA backend: {backend!r}")
    config.relax_k31_expanded_dsa_backend = backend
    if getattr(dsa._run_sparse_attention, "_relax_k31_expanded", False):
        return
    original = dsa._run_sparse_attention

    @wraps(original)
    def run_sparse_attention(**kwargs: Any) -> Tensor:
        selected = getattr(kwargs["config"], "relax_k31_expanded_dsa_backend", None)
        query, key, value = kwargs["query"], kwargs["key"], kwargs["value"]
        if (
            selected
            and not kwargs["absorbed_mla"]
            and value is not None
            and query.is_cuda
            and query.ndim == key.ndim == value.ndim == 4
            and query.shape[2] == key.shape[2] == value.shape[2]
            and kwargs["mask"] is None
            and kwargs["varlen_starts"] is not None
            and kwargs["varlen_ends"] is not None
        ):
            return packed_expanded_attention(
                query,
                key,
                value,
                kwargs["topk_indices"],
                kwargs["softmax_scale"],
                varlen_starts=kwargs["varlen_starts"],
                varlen_ends=kwargs["varlen_ends"],
                key_positions=kwargs["key_positions"],
                backend=selected,
            )
        return original(**kwargs)

    run_sparse_attention._relax_k31_expanded = True
    dsa._run_sparse_attention = run_sparse_attention
    get_logger(__name__).info("Enabled opt-in K3.1 packed expanded DSA using SDPA %s", backend)
