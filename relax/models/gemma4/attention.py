# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Gemma-4 core attention that survives the packed (THD) backward.

TE's cuDNN FusedAttention emits NaN gradients in the THD backward on gemma-4's
sliding layers. The forward is bit-identical to the unpacked run, so the defect
is that kernel's backward alone; the originating layer varies between otherwise
identical runs. Upstream report (context-parallel variant of the same failure):
https://github.com/NVIDIA/TransformerEngine/issues/2186

Ported from THUDM/slime ``slime_plugins/models/gemma4.py::SDPACoreAttention``.

Dispatch, CP == 1 only:

    thd + cu_seqlens + head_dim <= 256  -> flash_attn_varlen_func, window (sw-1, 0)
    thd + cu_seqlens + head_dim  > 256  -> per-sub-sequence causal SDPA
    thd without cu_seqlens              -> raises
    sbhd                                -> raises
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _is_sliding_layer(config, layer_number: int) -> bool:
    """Is this layer a sliding-window layer?

    Dense providers encode the pattern in ``window_attn_skip_freq``, MoE ones
    in ``interleaved_attn_pattern``.
    """
    window_size = getattr(config, "window_size", None)
    if not window_size:
        return False
    skip_freq = getattr(config, "window_attn_skip_freq", None)
    if isinstance(skip_freq, list):
        layer_type = skip_freq[layer_number - 1]
        if isinstance(layer_type, str):
            return layer_type == "sliding_attention"
        return bool(layer_type)
    if skip_freq is None:
        # MoE: pattern lives in interleaved_attn_pattern. Returning True here
        # unconditionally would put a window on the global layers.
        pattern = getattr(config, "interleaved_attn_pattern", None)
        if pattern:
            return layer_number % sum(pattern) != 0
        return True
    if isinstance(skip_freq, int):
        return layer_number % skip_freq != 0
    return False


def _window_pair(window_size) -> tuple:
    """Normalise gemma-4's two window conventions to flash-attn's ``(left,
    right)``.

    Dense stores the inclusive pair, MoE the raw span. Do not subtract from the
    dense tuple -- the -1 is already baked in and doing it twice shifts the
    window by a token.
    """
    if not window_size:
        return (-1, -1)
    if isinstance(window_size, int):
        return (window_size - 1, 0)
    return tuple(window_size)


class Gemma4CoreAttention(nn.Module):
    """Drop-in replacement for TEDotProductAttention on gemma-4 dense
    layers."""

    def __init__(
        self,
        config,
        layer_number: int,
        attn_mask_type=None,
        attention_type: str = "self",
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        # Megatron/TE hand core_attention a moving set of kwargs; accept and drop.
        del kwargs
        self.config = config
        self.layer_number = layer_number
        self.attention_type = attention_type
        # gemma-4 sets softmax_scale to 1.0, not head_dim**-0.5, because the query
        # is pre-scaled upstream. Do not "fix" it -- doing so made per-layer
        # divergence go 0.25 -> 14.6.
        self.softmax_scale = softmax_scale if softmax_scale is not None else getattr(config, "softmax_scale", None)
        self.dropout_p = config.attention_dropout if attention_dropout is None else attention_dropout
        self.is_sliding = _is_sliding_layer(config, layer_number)
        self.window_size = _window_pair(getattr(config, "window_size", None)) if self.is_sliding else (-1, -1)

    def _scale(self, head_dim: int) -> float:
        return self.softmax_scale if self.softmax_scale is not None else head_dim**-0.5

    # ---------------- THD ----------------

    def _thd_flash(self, query, key, value, cu_seqlens, max_seqlen: int):
        """head_dim <= 256: flash-attn's variable-length kernel, replacing
        cuDNN FusedAttention."""
        from flash_attn import flash_attn_varlen_func

        cu = cu_seqlens.to(torch.int32)
        out = flash_attn_varlen_func(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self._scale(query.shape[2]),
            causal=True,
            window_size=self.window_size,
        )
        return out.reshape(query.shape[0], -1)

    def _thd_sdpa_per_subseq(self, query, key, value, boundaries):
        """head_dim > 256 (gemma-4's global layers), which flash-attn cannot
        take.

        Loops sub-sequences rather than materialising a [T, T] block-diagonal
        mask.
        """
        n_q, head_dim = query.shape[1], query.shape[2]
        n_kv = key.shape[1]
        scale = self._scale(head_dim)
        out = torch.empty(query.shape[0], n_q * head_dim, dtype=query.dtype, device=query.device)
        for s, e in zip(boundaries, boundaries[1:]):
            o = F.scaled_dot_product_attention(
                query[s:e].unsqueeze(0).transpose(1, 2),
                key[s:e].unsqueeze(0).transpose(1, 2),
                value[s:e].unsqueeze(0).transpose(1, 2),
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=scale,
                is_causal=True,
                enable_gqa=(n_q != n_kv),
            )
            out[s:e] = o.transpose(1, 2).reshape(e - s, -1)
        return out

    # ---------------- entry ----------------

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        attn_mask_type=None,
        packed_seq_params=None,
        **kwargs,
    ):
        cp_size = getattr(self.config, "context_parallel_size", 1) or 1
        if cp_size > 1:
            raise NotImplementedError(
                "Gemma4CoreAttention does not implement context parallelism. slime's CP path "
                "relies on its own zig-zag CP layout, which is not Relax's convention; porting "
                "it unverified would silently compute the wrong attention. Run with "
                "--context-parallel-size 1, or implement CP against Relax's cp_utils first."
            )

        if query.dim() == 3:  # thd: [T, np, hn]
            cu_seqlens = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None
            if cu_seqlens is not None:
                if query.shape[2] <= 256:
                    max_seqlen = packed_seq_params.max_seqlen_q
                    if max_seqlen is None:
                        raise ValueError("packed_seq_params.max_seqlen_q is required for Gemma4 flash attention")
                    return self._thd_flash(query, key, value, cu_seqlens, max_seqlen)
                boundaries = getattr(packed_seq_params, "cu_seqlens_q_cpu", None)
                if boundaries is None:
                    # PackedSeqParams built outside Relax's data path. Copy once.
                    boundaries = cu_seqlens.detach().cpu().tolist()
                return self._thd_sdpa_per_subseq(query, key, value, boundaries)
            raise NotImplementedError(
                "Gemma4CoreAttention requires packed THD inputs with cu_seqlens. "
                "Treating THD input without sequence boundaries as one sequence would "
                "silently lose Gemma-4's sliding-window and padding-mask semantics."
            )

        raise NotImplementedError(
            "Gemma4CoreAttention currently supports packed THD attention only. "
            "The SBHD path needs an explicit causal sliding-window mask before it can be enabled safely."
        )
