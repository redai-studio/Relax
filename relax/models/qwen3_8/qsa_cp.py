# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in local-query QSA and invocation-local MCore checkpoint replay.

The replay tape belongs to a CheckpointFunction invocation, never to a layer.
Only frozen indexer metadata and BlockMasks are retained; K/V stays
differentiable.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import torch
from megatron.core.ssm.mamba_context_parallel import _thd_partitioned_indices, _undo_attention_load_balancing
from megatron.core.tensor_parallel import random as tp_random
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.experimental_attention_variant import qsa
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from relax.models.qwen3_8 import qsa_block_mask, qsa_dispatch
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class _ReplayTape:
    entries: list[tuple[tuple[str, int], Any]] = field(default_factory=list)
    cursor: int = 0
    replay: bool = False

    def get(self, key: tuple[str, int], build: Callable[[], Any]) -> Any:
        if self.replay:
            if self.cursor >= len(self.entries) or self.entries[self.cursor][0] != key:
                raise RuntimeError("QSA checkpoint replay order differs from its forward")
            value = self.entries[self.cursor][1]
        else:
            value = build()
            self.entries.append((key, value))
        self.cursor += 1
        return value


_active_tape: ContextVar[_ReplayTape | None] = ContextVar("qsa_checkpoint_tape", default=None)


def _cached(kind: str, owner: object, build: Callable[[], Any]) -> Any:
    tape = _active_tape.get()
    return build() if tape is None else tape.get((kind, id(owner)), build)


def _with_replay(function: Callable) -> Callable:
    tape = _ReplayTape()

    @wraps(function)
    def run(*args: Any, **kwargs: Any) -> Any:
        tape.cursor = 0
        token = _active_tape.set(tape)
        try:
            result = function(*args, **kwargs)
            if tape.replay and tape.cursor != len(tape.entries):
                raise RuntimeError("QSA checkpoint replay did not consume all forward entries")
            tape.replay = True
            return result
        finally:
            _active_tape.reset(token)

    return run


def _checkpoint_forward(ctx: Any, function: Callable, distribute_saved_activations: bool, *args: Any) -> Any:
    return tp_random.CheckpointFunction._relax_native_forward(
        ctx, _with_replay(function), distribute_saved_activations, *args
    )


def _indexer_forward(self: qsa.QSAIndexer, *args: Any, **kwargs: Any) -> qsa.QSASelection:
    # Stage-2/trainable indexers must preserve their original autograd and RNG path.
    if self.config.qsa_indexer_loss_coeff != 0 or any(p.requires_grad for p in self.parameters()):
        return qsa_dispatch.tp_forward(self, *args, **kwargs)

    def build() -> qsa.QSASelection:
        with torch.no_grad():
            return qsa_dispatch.tp_forward(self, *args, **kwargs)

    return _cached("indexer", self, build)


def _local_mask(selection: qsa.QSASelection, query_ids: Tensor) -> BlockMask:
    global_mod = qsa._qsa_mask_mod_factory(selection)
    local_length = query_ids.numel()
    global_length = selection.positions.shape[1]

    def mask_mod(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
        # Flex may evaluate padded rows in a final partial query tile.
        global_query = query_ids[query.clamp(max=local_length - 1)]
        global_key = key.clamp(max=global_length - 1)
        return (query < local_length) & (key < global_length) & global_mod(batch, head, global_query, global_key)

    return qsa_block_mask._build_sparse_block_mask(
        selection, mask_mod, selection.positions.shape[1], query_ids=query_ids
    )


def _core_forward(
    self: qsa.QSACoreAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    attn_mask_type: Any = None,
    attention_bias: Tensor | None = None,
    packed_seq_params: Any = None,
    *,
    qsa_selection: qsa.QSASelection | None = None,
    qsa_index_query: Tensor | None = None,
    qsa_compressed_key: Tensor | None = None,
) -> Tensor:
    selection = qsa_selection if qsa_selection is not None else self._selection
    cp_group = self.pg_collection.cp
    cp_size = qsa.get_pg_size(cp_group)
    # Preserve dense/all-selected, stage-2, non-THD and alternate backend behavior.
    if (
        cp_size <= 1
        or packed_seq_params is None
        or packed_seq_params.qkv_format != "thd"
        or not query.is_cuda
        or selection is None
        or selection.selected_bits is None
        or selection.compress_ratio != 4
        or selection.bits_per_row <= 0
        or self.config.qsa_indexer_loss_coeff != 0
        or qsa_index_query is not None
        or qsa_compressed_key is not None
        or (selection.all_selected and not self.config.qsa_force_sparse)
        or self.sparse_backend != "flex"
        or not qsa.HAVE_FLEX_ATTENTION
    ):
        return qsa.QSACoreAttention._relax_native_forward(
            self,
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            attention_bias,
            packed_seq_params,
            qsa_selection=qsa_selection,
            qsa_index_query=qsa_index_query,
            qsa_compressed_key=qsa_compressed_key,
        )
    assert attention_bias is None, "QSA does not support attention bias"

    def local_ids() -> Tensor:
        cu = packed_seq_params.cu_seqlens_q_padded
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        with torch.no_grad():
            ids = _thd_partitioned_indices(
                cu, selection.positions.shape[1], cp_size, torch.distributed.get_rank(cp_group)
            )
        if ids.numel() != query.shape[0]:
            raise ValueError("QSA local query length does not match packed CP layout")
        return ids

    query_ids = _cached("local_ids", self, local_ids)
    # Explicit CP group; preserve remote K/V gradient summation in gather backward.
    kv_heads = key.shape[-2]
    kv = gather_from_sequence_parallel_region(
        torch.cat((key, value), dim=-2), tensor_parallel_output_grad=True, group=cp_group
    )
    kv = _undo_attention_load_balancing(kv, cp_size, packed_seq_params)
    key, value = kv.split(kv_heads, dim=-2)
    from relax.models.qwen3_8.qsa_fused import backend

    _fused_backend = backend()
    if _fused_backend is not None and _fused_backend.supports(selection, query, key, self.sparse_backend):
        plan = _cached(
            "fused_plan",
            self,
            lambda: _fused_backend.build_plan(selection, selection.positions.shape[1], query_ids=query_ids),
        )
        output = _fused_backend.qsa_fused_attention(query, key, value, selection, plan, query_ids, self.softmax_scale)
        return output.reshape(query.shape[0], -1)
    mask = _cached("block_mask", self, lambda: _local_mask(selection, query_ids))
    output = qsa._compiled_flex_attention()(
        query.unsqueeze(0).transpose(1, 2).contiguous(),
        key.unsqueeze(0).transpose(1, 2).contiguous(),
        value.unsqueeze(0).transpose(1, 2).contiguous(),
        block_mask=mask,
        scale=self.softmax_scale,
        enable_gqa=query.shape[-2] != kv_heads,
    )
    return output.transpose(1, 2).squeeze(0).reshape(query.shape[0], -1)
