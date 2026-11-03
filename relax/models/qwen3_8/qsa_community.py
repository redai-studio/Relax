# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Mixed Flash/Flex/Grouped adapter for Megatron's selected-ID QSA path."""

from __future__ import annotations

import os

import torch
from megatron.core.transformer.experimental_attention_variant import qsa
from torch import Tensor

from relax.utils.env import Envs


_SUPPORTED = ("mixed",)


def validate_backend() -> None:
    selected = Envs.RELAX_QWEN38_QSA_COMMUNITY_BACKEND.strip()
    if selected not in _SUPPORTED:
        raise ValueError(
            f"Unsupported QSA community backend {selected!r}; only 'mixed' is retained. "
            "Clear RELAX_QWEN38_QSA_COMMUNITY_BACKEND to use native or fused attention."
        )
    if os.environ.get("MCORE_QSA_SPARSE_BACKEND", "id_sparse") != "id_sparse":
        raise ValueError("Mixed QSA attention requires MCORE_QSA_SPARSE_BACKEND=id_sparse")


def _community_id_sparse_forward(
    self: qsa.QSACoreAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    selection: qsa.QSASelection,
    is_thd: bool,
) -> Tensor:
    if selection.selected_ids is None or not query.is_cuda:
        return qsa.QSACoreAttention._relax_native_id_sparse_forward(self, query, key, value, selection, is_thd)
    if (
        selection.compress_ratio != 4
        or query.dtype != torch.bfloat16
        or key.dtype != query.dtype
        or value.dtype != query.dtype
    ):
        raise NotImplementedError("Mixed QSA attention requires BF16 Q/K/V and 4-token blocks")
    if is_thd:
        q, k, v = (tensor.unsqueeze(0).contiguous() for tensor in (query, key, value))
    else:
        q, k, v = (tensor.permute(1, 0, 2, 3).contiguous() for tensor in (query, key, value))
    batch, sequence_length, heads, width = q.shape
    if selection.positions.shape != (batch, sequence_length) or k.shape[1] != sequence_length:
        raise ValueError("Mixed QSA attention requires full-sequence selected IDs and K/V")

    from relax.models.qwen3_8.ops.qsa_mixed_ops import attention_for_selection

    output = attention_for_selection(q, k, v, selection, self.softmax_scale, is_thd)

    if is_thd:
        return output.squeeze(0).reshape(sequence_length, heads * width)
    return output.permute(1, 0, 2, 3).reshape(sequence_length, batch, heads * width)
