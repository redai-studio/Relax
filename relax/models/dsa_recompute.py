# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reuse frozen DSA selection within one activation-checkpoint invocation."""

import copy
from dataclasses import dataclass, field
from functools import wraps
from types import MethodType
from typing import Any

import torch
from torch import Tensor


_SCOPE_ATTR = "_relax_dsa_checkpoint_scope"


@dataclass
class _Selection:
    signature: tuple
    indices: Tensor
    lengths: Tensor | None
    pool_tail: Tensor | None = None


@dataclass
class _Scope:
    selections: dict[int, _Selection] = field(default_factory=dict)
    consumed: set[int] = field(default_factory=set)


def with_selection_scope(packed: Any) -> Any:
    """Keep cache ownership with this forward's checkpoint closures."""
    packed = copy.copy(packed)
    setattr(packed, _SCOPE_ATTR, _Scope())
    return packed


def _is_checkpointing() -> bool:
    from megatron.core.tensor_parallel.random import is_checkpointing

    return is_checkpointing()


def configure_indexer_recompute_reuse(attention: Any, *, pool_size: int = 1) -> None:
    """Record native top-k once, then use Core's sharing path during replay.

    The owner belongs to the enclosing checkpoint closure, never to a module,
    step counter, shape cache, or the caller's original packed metadata.
    """
    if getattr(attention, "_relax_dsa_selection_reuse", False):
        return
    if (
        attention.config.dsa_kernel_backend != "tilelang"
        or attention.config.dsa_indexer_loss_coeff
        or attention.index_share
        or attention.skip_topk
        or attention.indexer is None
        or any(p.requires_grad for p in attention.indexer.parameters())
    ):
        raise ValueError("DSA selection reuse requires a frozen, independent TileLang indexer without auxiliary loss")
    original = attention.forward

    @wraps(original)
    def forward(
        self: Any,
        query: Tensor,
        key: Tensor,
        value: Tensor | None,
        attention_mask: Tensor | None,
        x: Tensor,
        qr: Tensor,
        position_ids: Tensor | None = None,
        attn_mask_type: Any = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: Any = None,
        up_v_weight: Tensor | None = None,
    ) -> Tensor:
        options = dict(
            x=x,
            qr=qr,
            position_ids=position_ids,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            up_v_weight=up_v_weight,
        )
        scope = getattr(packed_seq_params, _SCOPE_ATTR, None)
        if scope is None or not self.training or not _is_checkpointing() or self.index_share:
            return original.__func__(self, query, key, value, attention_mask, **options)
        if self.config.dsa_kernel_backend != "tilelang":
            raise ValueError("Selection reuse has only been validated for the TileLang backend")
        layer = self.layer_number
        signature = (tuple(query.shape), tuple(key.shape), query.dtype, query.device, self.index_topk)
        replay = torch.is_grad_enabled()
        selection = scope.selections.get(layer) if replay else None
        if replay and selection is None and layer not in scope.consumed:
            # Do not let a rank-local miss change the CP collective sequence.
            raise RuntimeError("Missing first-replay DSA selection in this checkpoint invocation")
        if not replay and (layer in scope.selections or layer in scope.consumed):
            raise RuntimeError("Repeated DSA layer in one checkpoint scope; refusing to overwrite selection")
        if selection is not None and selection.signature != signature:
            raise RuntimeError("DSA checkpoint replay shape/config differs from its original forward")
        delegated = copy.copy(self)
        delegated.index_share = True
        delegated.skip_topk = selection is not None
        delegated.source_layer = layer
        if pool_size > 1:
            # Share Core's per-pack position metadata across the shallow views.
            self._get_packed_cp_layout_cache(packed_seq_params)
        packed = copy.copy(packed_seq_params)
        holder = {}
        if selection is not None:
            indices = selection.indices
            if selection.pool_tail is not None:
                offsets = torch.arange(pool_size, device=indices.device, dtype=indices.dtype)
                tokens = indices.unsqueeze(-1) + offsets
                tokens.masked_fill_(indices.unsqueeze(-1) < 0, -1)
                indices = torch.cat((tokens.flatten(-2), selection.pool_tail), dim=-1)
            holder[layer] = indices
        lengths = {layer: selection.lengths} if selection is not None and selection.lengths is not None else {}
        setattr(packed, self._HOLDER_ATTR, holder)
        setattr(packed, self._LENGTH_HOLDER_ATTR, lengths)
        options["packed_seq_params"] = packed
        output = original.__func__(delegated, query, key, value, attention_mask, **options)
        if replay:
            scope.selections.pop(layer, None)
            scope.consumed.add(layer)
        else:
            indices = holder[layer].detach()
            length = lengths.get(layer)
            pool_tail = None
            if pool_size > 1:
                from megatron.core.transformer.enums import AttnMaskType

                # Causal THD KPool returns whole groups of consecutive tokens
                # (or all -1) followed by the incomplete causal tail. An
                # explicit additive mask may instead invalidate individual slots.
                if attn_mask_type == AttnMaskType.causal:
                    pool_tail = indices[..., self.index_topk :].clone()
                    indices = indices[..., : self.index_topk : pool_size].clone()
            scope.selections[layer] = _Selection(
                signature, indices, None if length is None else length.detach(), pool_tail
            )
        return output

    attention.forward = MethodType(forward, attention)
    attention._relax_dsa_selection_reuse = True
