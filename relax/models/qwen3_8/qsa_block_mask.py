# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in coarse QSA BlockMask construction from the selected R=4 bitset.

The original QSA mask_mod still decides every token pair inside partial blocks.
This patch only replaces the expensive token-pair scan used to discover which
128-token FlexAttention blocks are empty, partial, or full.
"""

from __future__ import annotations

import torch
from megatron.core.transformer.experimental_attention_variant import qsa
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from relax.models.qwen3_8.ops.qsa_block_mask import (
    _build_sparse_block_mask as _build_sparse_block_mask,
)
from relax.models.qwen3_8.ops.qsa_block_mask import (
    _classify_blocks as _classify_blocks,
)
from relax.models.qwen3_8.ops.qsa_block_mask import (
    _pack_block_indices as _pack_block_indices,
)
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_BLOCK_SIZE = 128


def _sparse_flex_forward(
    self: qsa.QSACoreAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    selection: qsa.QSASelection,
    is_thd: bool,
) -> Tensor:
    if (
        not query.is_cuda
        or selection.selected_bits is None
        or selection.compress_ratio != 4
        or selection.bits_per_row <= 0
        or not qsa.HAVE_FLEX_ATTENTION
    ):
        return qsa.QSACoreAttention._relax_native_flex_forward(self, query, key, value, selection, is_thd)
    if is_thd:
        q = query.unsqueeze(0).transpose(1, 2)
        k = key.unsqueeze(0).transpose(1, 2)
        v = value.unsqueeze(0).transpose(1, 2)
    else:
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)
    batch_size, heads, sequence_length, width = q.shape

    def build_mask() -> BlockMask:
        mask_mod = qsa._qsa_mask_mod_factory(selection)
        return _build_sparse_block_mask(selection, mask_mod, sequence_length)

    if Envs.RELAX_QWEN38_QSA_RECOMPUTE_REPLAY:
        # Import lazily: the CP patch also uses this module's mask builder.
        from relax.models.qwen3_8.qsa_cp import _cached

        block_mask = _cached("flex_block_mask", self, build_mask)
    else:
        block_mask = build_mask()
    try:
        out = qsa._compiled_flex_attention()(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            block_mask=block_mask,
            scale=self.softmax_scale,
            enable_gqa=k.shape[1] != heads,
        )
    except torch._dynamo.exc.TorchDynamoException:
        if sequence_length > 4096:
            raise
        return self._dense_masked_forward(query, key, value, selection, is_thd)
    if is_thd:
        return out.transpose(1, 2).squeeze(0).reshape(sequence_length, heads * width)
    return out.permute(2, 0, 1, 3).reshape(sequence_length, batch_size, heads * width)
