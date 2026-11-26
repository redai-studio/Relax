# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from functools import wraps
from typing import Any

import torch

from relax.utils.logging_utils import get_logger


def configure_runtime(provider: Any, args: Namespace) -> None:
    """Configure the Flash Bridge input layout without changing CLI
    defaults."""
    pattern = getattr(args, "hybrid_layer_pattern", None)
    if pattern and "|" in pattern:
        if pattern.replace("|", "") != provider.hybrid_layer_pattern:
            raise ValueError("GLM Flash pipeline cuts must preserve the source model's physical layer pattern.")
        provider.hybrid_layer_pattern = pattern

    is_sft = getattr(args, "loss_type", None) == "sft"
    packed_cp = provider.context_parallel_size > 1
    packed = getattr(args, "qkv_format", "thd") == "thd"
    args.needs_cpu_image_grid = (
        getattr(provider, "vision_dp_when_tp", False) and provider.tensor_model_parallel_size > 1
    )
    if (packed_cp or is_sft) and not packed:
        raise ValueError("GLM-5.3-Flash SFT and CP require packed THD batches.")
    if getattr(args, "dynamic_context_parallel", False):
        raise ValueError("GLM-5.3-Flash Bridge requires static context parallelism.")
    if packed_cp:
        if getattr(args, "allgather_cp", False) or getattr(args, "cp_partition_mode", "zigzag") != "zigzag":
            raise ValueError("GLM-5.3-Flash Bridge requires per-document zigzag CP packing.")
        if provider.cp_comm_type != "allgather":
            raise ValueError("GLM-5.3-Flash DSA requires --cp-comm-type allgather with CP > 1.")
        if getattr(args, "enable_mtp_training", False):
            raise ValueError("GLM-5.3-Flash Bridge CP currently requires MTP training to be disabled.")
        args.global_packed_thd_cp = True
        args.uses_unsplit_forward = True

    if is_sft and not getattr(args, "enable_mtp_training", False):
        # The HF checkpoint contains a draft block, but ordinary SFT
        # consumes only the main model logits and supplies no MTP labels.
        provider.mtp_num_layers = None
        provider.mtp_hybrid_override_pattern = None
        provider.mtp_use_repeated_layer = False


def configure_dsa_selection_reuse(model: torch.nn.Module) -> None:
    """Install KPool selection reuse on the local GLM Flash language stack."""
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention

    from relax.models.dsa_recompute import configure_indexer_recompute_reuse, with_selection_scope

    stack = model.language_model.decoder
    attentions = [module for module in stack.modules() if isinstance(module, DSAttention)]
    if not attentions:
        # A physical pipeline stage can contain only KDA layers.
        return
    config = stack.config
    if (
        config.recompute_granularity != "full"
        or config.recompute_method != "uniform"
        or config.params_dtype != torch.bfloat16
        or config.fp8
        or config.fp4
        or config.cuda_graph_impl != "none"
        or getattr(stack._cp_layout_manager, "requires_conversion", False)
    ):
        raise ValueError(
            "GLM Flash DSA selection reuse requires BF16 full/uniform checkpointing with static CP layout"
        )
    for attention in attentions:
        indexer = attention.indexer
        if indexer is None or indexer.index_kpool <= 1:
            raise ValueError("GLM Flash DSA selection reuse requires an independent KPool indexer")
        configure_indexer_recompute_reuse(attention, pool_size=indexer.index_kpool)

    original = stack.forward

    @wraps(original)
    def forward(*args: Any, packed_seq_params: Any = None, **kwargs: Any) -> torch.Tensor:
        if stack.training and torch.is_grad_enabled() and getattr(packed_seq_params, "qkv_format", None) == "thd":
            packed_seq_params = with_selection_scope(packed_seq_params)
        return original(*args, packed_seq_params=packed_seq_params, **kwargs)

    stack.forward = forward
    get_logger(__name__).info("Enabled GLM Flash DSA selection reuse for %s local layers.", len(attentions))
