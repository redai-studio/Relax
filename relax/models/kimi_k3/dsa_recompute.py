# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reuse frozen DSA selection within one activation-checkpoint invocation."""

import copy
import inspect
from dataclasses import dataclass, field
from functools import wraps
from types import MethodType
from typing import Any

import torch
from torch import Tensor


_SCOPE_ATTR = "_relax_k31_dsa_checkpoint_scope"


@dataclass
class _Selection:
    signature: tuple
    indices: Tensor
    lengths: Tensor | None


@dataclass
class _Scope:
    selections: dict[int, _Selection] = field(default_factory=dict)
    consumed: set[int] = field(default_factory=set)


def _is_checkpointing() -> bool:
    from megatron.core.tensor_parallel.random import is_checkpointing

    return is_checkpointing()


def configure_checkpoint_scope(block: Any) -> None:
    """Give each block invocation its own packed view, captured by Core's
    closures."""
    if getattr(block, "_relax_dsa_checkpoint_scope", False):
        return
    config = block.config
    if (
        config.recompute_granularity != "full"
        or config.recompute_method != "uniform"
        or config.recompute_num_layers != 1
        or getattr(config, "fp8", None)
        or getattr(config, "fp4", None)
        or getattr(config, "cuda_graph_impl", "none") not in (None, "none")
    ):
        raise ValueError("DSA selection reuse requires full/uniform/1 checkpointing without quantization or graphs")
    original = block._checkpointed_forward
    signature = inspect.signature(original)

    @wraps(original)
    def checkpointed_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        packed = bound.arguments.get("packed_seq_params")
        if packed is not None:
            packed = copy.copy(packed)
            setattr(packed, _SCOPE_ATTR, _Scope())
            bound.arguments["packed_seq_params"] = packed
        try:
            return original.__func__(self, *bound.args, **bound.kwargs)
        except BaseException:
            if packed is not None:
                getattr(packed, _SCOPE_ATTR).selections.clear()
            raise

    block._checkpointed_forward = MethodType(checkpointed_forward, block)
    block._relax_dsa_checkpoint_scope = True


def configure_indexer_recompute_reuse(attention: Any) -> None:
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
        packed = copy.copy(packed_seq_params)
        holder = {layer: selection.indices} if selection is not None else {}
        lengths = {layer: selection.lengths} if selection is not None and selection.lengths is not None else {}
        setattr(packed, self._HOLDER_ATTR, holder)
        setattr(packed, self._LENGTH_HOLDER_ATTR, lengths)
        options["packed_seq_params"] = packed
        output = original.__func__(delegated, query, key, value, attention_mask, **options)
        if replay:
            scope.selections.pop(layer, None)
            scope.consumed.add(layer)
        else:
            indices = holder[layer]
            length = lengths.get(layer)
            scope.selections[layer] = _Selection(
                signature, indices.detach(), None if length is None else length.detach()
            )
        return output

    attention.forward = MethodType(forward, attention)
    attention._relax_dsa_selection_reuse = True


def configure_model_indexer_reuse(model: Any, query_chunk_size: int = 8192) -> None:
    """Install only on K3.1 blocks that own DSA layers."""
    from megatron.core.transformer.transformer_block import TransformerBlock

    from relax.models.kimi_k3.dsa import KimiK31DSAAttention
    from relax.utils.logging_utils import get_logger

    if query_chunk_size not in (1024, 2048, 4096, 8192):
        raise ValueError("DSA query chunk size must be 1024, 2048, 4096 or 8192")
    count = 0
    for block in model.modules():
        if not isinstance(block, TransformerBlock):
            continue
        attentions = [module for module in block.modules() if isinstance(module, KimiK31DSAAttention)]
        if not attentions:
            continue
        configure_checkpoint_scope(block)
        for module in attentions:
            if module.core_attention.config.params_dtype != torch.bfloat16:
                raise ValueError("Model-level DSA selection reuse is currently limited to BF16")
            configure_indexer_recompute_reuse(module.core_attention)
            module.core_attention.fused_indexer_block_size = query_chunk_size
            count += 1
    if not count:
        # A K3.1 pipeline stage may contain only KDA layers.
        return
    get_logger(__name__).info("Enabled K3.1 checkpoint-scoped DSA selection reuse for %s layers.", count)
    get_logger(__name__).info("K3.1 DSA indexer query chunk size: %s.", query_chunk_size)
