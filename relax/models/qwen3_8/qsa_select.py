# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in QSA selector fusion preserving FP32 GEMM, TopK and route bits."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from relax.models.qwen3_8 import qsa_dispatch
from relax.models.qwen3_8.ops.qsa_selector import (
    _pack_all_candidates_kernel as _pack_all_candidates_kernel,
)
from relax.models.qwen3_8.ops.qsa_selector import (
    _pack_kernel as _pack_kernel,
)
from relax.models.qwen3_8.ops.qsa_selector import (
    _score_kernel as _score_kernel,
)
from relax.models.qwen3_8.ops.qsa_selector import (
    _score_topk_pack_kernel as _score_topk_pack_kernel,
)
from relax.models.qwen3_8.ops.qsa_selector import (
    _topk_mask as _topk_mask,
)
from relax.models.qwen3_8.ops.qsa_selector import (
    pack_all_candidates as pack_all_candidates,
)
from relax.models.qwen3_8.ops.qsa_selector import (
    pack_topk as pack_topk,
)
from relax.models.qwen3_8.ops.qsa_selector import (
    score_postprocess as score_postprocess,
)
from relax.models.qwen3_8.ops.qsa_selector import (
    score_topk_pack as score_topk_pack,
)
from relax.utils.logging_utils import get_logger


if TYPE_CHECKING:
    from megatron.core.transformer.experimental_attention_variant import qsa

logger = get_logger(__name__)

_layout: ContextVar[tuple[int, ...] | None] = ContextVar("qsa_select_layout", default=None)
# Megatron's 64 MiB budget sizes a tile from the widest document, which now
# over-reserves by the narrowing factor and cuts every document into several
# times more GEMM, TopK and Triton launches than the budget needs. Raising it
# recovers most of that: measured over the real 128K routing, 64 -> 128 MiB
# takes the selector from 46.7 ms to 33.1 ms, and 256 MiB to 29.4 ms, with
# byte-identical routes at every budget. The tile is clamped per step anyway,
# so only the launch count changes.
_TILE_BYTES = 256 << 20


# Packed sample lengths vary within and across steps. Keep indexing sizes out of
# the compilation key; only tile geometry and algorithm constants specialize.


@torch.no_grad()
def select_blocks(
    self: qsa.QSAIndexer,
    q: Tensor,
    pooled_keys: Tensor,
    block_valid: Tensor,
    doc_ids: Tensor,
    positions: Tensor,
    uniform_doc_len: int | None = None,
    output_format: str = "bits",
) -> tuple[Tensor, bool]:
    from megatron.core.transformer.experimental_attention_variant import qsa

    total, heads, dim = q.shape
    n_docs, columns, _ = pooled_keys.shape
    offsets = _layout.get()
    if n_docs == 1:
        offsets = (0, total)
    elif uniform_doc_len is not None:
        offsets = tuple(i * uniform_doc_len for i in range(n_docs + 1))
    if (
        total == 0
        or n_docs == 0
        or output_format != "bits"
        or heads != 4
        or not q.is_cuda
        or not positions.is_contiguous()
        or offsets is None
        or not isinstance(offsets, tuple)
        or any(type(v) is not int for v in offsets)
        or any(b < a for a, b in zip(offsets, offsets[1:]))
        or len(offsets) != n_docs + 1
        or offsets[0] != 0
        or offsets[-1] != total
        or columns <= 1
        or max(b - a for a, b in zip(offsets, offsets[1:])) // self.compress_ratio <= self.block_topk
    ):
        return qsa_dispatch.cpu_select(
            self, q, pooled_keys, block_valid, doc_ids, positions, uniform_doc_len, output_format
        )
    nbytes = (columns + 8) // 8
    k = min(self.block_topk, columns)
    gathered_bytes = (
        columns * dim * (pooled_keys.element_size() + 4)
        + columns * (heads * 4 + 8)
        + heads * dim * 4
        + nbytes * 4
        + k * 32
    )
    gathered_tile = max(1, min(total, qsa._QSA_SELECT_TILE_BYTES // gathered_bytes))
    if n_docs > 1 and gathered_tile >= min(total, 32) and total <= n_docs * gathered_tile:
        return qsa_dispatch.cpu_select(
            self, q, pooled_keys, block_valid, doc_ids, positions, uniform_doc_len, output_format
        )
    # Like the original selector, rely on _pool_keys producing a valid prefix
    # of floor(doc_length / R) complete blocks and document-relative positions.
    # Causal visibility never exceeds that prefix, so invalid pooled-key tails
    # cannot be selected. block_valid is not an arbitrary attention/padding mask.
    selected = torch.empty((total, nbytes), dtype=torch.uint8, device=q.device)
    # SM90 and SM103 use the same kernel; wider rows benefit from 16 warps
    # on SM103. Retain ATen beyond each measured limit and on other devices.
    capability = torch.cuda.get_device_capability(q.device)
    fused_topk_limit = 32768 if capability == (10, 3) else (16384 if capability == (9, 0) else 0)
    # The tile only allocates as many columns as the tile can see, so sizing it
    # from the full ``columns`` bound leaves it far under budget and splits each
    # document into several times more GEMM, TopK and Triton launches than the
    # budget allows. Size from the widest narrow step instead; the loop below
    # clamps every tile to its own ``visible`` count anyway.
    widest = min(columns, max(1, max(b - a for a, b in zip(offsets, offsets[1:])) // self.compress_ratio))
    per_query = 4 * (widest * heads + heads * dim + nbytes)
    tile = max(1, min(total, _TILE_BYTES // max(1, per_query)))
    k = min(self.block_topk, columns)
    for start_doc, end_doc, doc in zip(offsets, offsets[1:], range(n_docs)):
        if start_doc == end_doc:
            continue
        key_t = pooled_keys[doc].float().T
        for start in range(start_doc, end_doc, tile):
            end = min(end_doc, start + tile)
            # Positions are document-relative, so the last row of the tile bounds
            # how many complete blocks any row can see. score_postprocess masks
            # every column past that bound to -inf, so narrowing the GEMM and the
            # TopK to it leaves every retained score and route bit unchanged.
            visible = min(columns, (end - start_doc) // self.compress_ratio)
            if visible == 0:
                selected[start:end].zero_()
                continue
            query = q[start:end].float().reshape(-1, dim)
            scores = (query @ key_t[:, :visible]).view(end - start, heads, visible)
            if visible <= k:
                pack_all_candidates(scores, positions[start:end], self.compress_ratio, dim, selected[start:end])
                continue
            if 0 < k and visible <= fused_topk_limit:
                score_topk_pack(scores, positions[start:end], self.compress_ratio, dim, k, selected[start:end])
                # Do not retain the H4 buffer while the next tile's GEMM allocates.
                del scores
                continue
            scores = score_postprocess(scores, positions[start:end], self.compress_ratio, dim)
            # Bits consume the selected set, not its rank order. CUDA TopK
            # gathers the same entries before the optional result sort.
            top = torch.topk(scores, k=min(k, visible), dim=-1, sorted=False)
            pack_topk(top.indices, top.values, selected[start:end])
    return selected, False


def indexer_forward(
    self: qsa.QSAIndexer,
    hidden_states: Tensor,
    rotary_pos_emb: Any,
    packed_seq_params: Any = None,
    output_format: str = "bits",
) -> qsa.QSASelection:
    offsets = getattr(packed_seq_params, "qsa_select_offsets_cpu", None)
    token = _layout.set(offsets)
    try:
        return qsa_dispatch.replay_forward(self, hidden_states, rotary_pos_emb, packed_seq_params, output_format)
    finally:
        _layout.reset(token)
