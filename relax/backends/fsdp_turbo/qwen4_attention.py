# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in QSA indexer and attention backends for Qwen4-Exp."""

import importlib
from contextlib import contextmanager
from types import MethodType

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def attention_block_mask(mask, query, key):
    """Translate HF allow/additive masks to the NPU blocked-mask convention."""
    import torch

    if mask is None:
        q_len, k_len = query.size(-2), key.size(-2)
        return torch.ones((q_len, k_len), dtype=torch.bool, device=query.device).triu(1 + k_len - q_len)
    if mask.dtype == torch.bool:
        return ~mask
    return mask < 0


def npu_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    import torch_npu

    if kwargs.get("output_attentions"):
        raise ValueError("Explicit NPU attention does not return attention weights")
    if getattr(module, "_relax_skip_attention_recompute", False) and dropout:
        raise ValueError("Attention recompute caching requires dropout=0")
    heads, kv_heads = query.size(1), key.size(1)
    if heads % kv_heads:
        raise ValueError("Query head count must be divisible by KV head count")
    if heads != kv_heads:
        key = key.repeat_interleave(heads // kv_heads, dim=1)
        value = value.repeat_interleave(heads // kv_heads, dim=1)
    blocked = attention_block_mask(attention_mask, query, key).contiguous()
    output = torch_npu.npu_fusion_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        heads,
        "BNSD",
        atten_mask=blocked,
        scale=scaling if scaling is not None else query.size(-1) ** -0.5,
        keep_prob=1.0 - dropout,
        sparse_mode=0,
        inner_precise=2,
    )[0]
    return output.transpose(1, 2).contiguous(), None


@contextmanager
def attention_recompute_policy(config, options):
    """Save NPU attention results while recomputing other layer operations."""
    if not options.skip_flash_attn_recompute:
        yield
        return
    if not config.memory.recompute:
        raise ValueError("skip_flash_attn_recompute requires memory.recompute=true")
    quantization_plan = config.quantization.quantization_plan
    if quantization_plan is not None and (quantization_plan.converters or quantization_plan.quant_recipe):
        raise ValueError("Selective attention recompute is not yet integrated with quantization contexts")
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

    recompute = importlib.import_module("fsdp_turbo.memory.recompute.recompute")
    original = recompute.recompute_wrapper

    def policy(ctx, op, *args, **kwargs):
        schema = getattr(op, "_schema", None)
        if schema is not None and schema.name.split("::")[-1] == "npu_fusion_attention":
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    def selective_wrapper(function, context_fn_factory=None, *args, **kwargs):
        def factory(module):
            contexts = [lambda: create_selective_checkpoint_contexts(policy)]
            base = context_fn_factory(module) if context_fn_factory is not None else None
            if base is not None:
                contexts.insert(0, base)
            return recompute.compose_checkpoint_contexts(contexts)

        return original(function, factory, *args, **kwargs)

    recompute.recompute_wrapper = selective_wrapper
    try:
        yield
    finally:
        recompute.recompute_wrapper = original


def configure(config, text_config) -> None:
    """Select the requested attention backend before model construction."""
    options = config.relax_performance
    distributed = config.distributed
    if options.qsa_indexer_implementation == "ascendc":
        try:
            from .qwen4_indexer import configure as configure_indexer
        except ImportError as exc:
            raise ValueError("AscendC QSA attention requires PR04 LightningIndexer") from exc
        configure_indexer(config, text_config)
    if options.attention_implementation == "ascendc_grouped":
        if distributed.tensor_parallel_size != 1:
            raise ValueError("Grouped QSA requires TP=1")
        from .performance_ops.qsa_grouped_ascendc import binding

        binding()
    if options.qsa_indexer_implementation == "ascendc" or options.attention_implementation != "native":
        if distributed.ulysses_parallel_size > 1 and getattr(text_config, "mtp_num_layers", 0):
            raise ValueError("Explicit QSA backends with Ulysses CP require MTP=0")
        if any(getattr(distributed, name, 1) != 1 for name in ("kv_allgather_parallel_size", "kcp_parallel_size")):
            raise ValueError("Explicit QSA backends support Ulysses CP only")
    if options.attention_implementation != "native":
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register("relax_qsa_npu", npu_attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register("relax_qsa_npu", ALL_MASK_ATTENTION_FUNCTIONS["eager"])
        text_config._attn_implementation = "relax_qsa_npu"


def install(model, config) -> None:
    """Install the selected indexer and CP/local attention paths."""
    options = config.relax_performance
    if options.qsa_indexer_implementation == "ascendc":
        from .qwen4_indexer import install as install_indexer

        install_indexer(model, config)
    for module in model.modules():
        name = type(module).__name__
        if name == "Qwen4ExpTextAttention":
            if options.attention_implementation != "native":
                module._relax_skip_attention_recompute = options.skip_flash_attn_recompute
            if config.distributed.ulysses_parallel_size > 1:
                if options.qsa_indexer_implementation == "ascendc" or options.attention_implementation != "native":
                    from .performance_ops.qsa_context_parallel import qsa_forward, qsa_grouped_forward

                    forward = (
                        qsa_grouped_forward if options.attention_implementation == "ascendc_grouped" else qsa_forward
                    )
                    module.context_parallel_forward = MethodType(forward, module)
                    module._relax_cp_npu_attention = options.attention_implementation == "npu"
            elif options.attention_implementation == "ascendc_grouped":
                from .performance_ops.qsa_grouped_local import qsa_grouped_local_forward

                module.forward = MethodType(qsa_grouped_local_forward, module)
