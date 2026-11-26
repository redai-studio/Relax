# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Install checkpoint-scoped DSA selection reuse on Kimi K3.1."""

import inspect
from functools import wraps
from types import MethodType
from typing import Any

import torch

from relax.models.dsa_recompute import _SCOPE_ATTR, configure_indexer_recompute_reuse, with_selection_scope


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
            packed = with_selection_scope(packed)
            bound.arguments["packed_seq_params"] = packed
        try:
            return original.__func__(self, *bound.args, **bound.kwargs)
        except BaseException:
            if packed is not None:
                getattr(packed, _SCOPE_ATTR).selections.clear()
            raise

    block._checkpointed_forward = MethodType(checkpointed_forward, block)
    block._relax_dsa_checkpoint_scope = True


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
