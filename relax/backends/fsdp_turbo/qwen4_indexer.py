# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from types import MethodType


def validate_lightning_mask(mask, query_length, kv_length):
    """Lightning routing supports causal right-padded prefixes and empty
    queries."""
    import torch

    if mask is None or mask.ndim != 4 or mask.shape[-2:] != (query_length, kv_length):
        raise ValueError("LightningIndexer requires a complete four-dimensional causal mask")
    allowed = mask if mask.dtype == torch.bool else mask >= 0
    causal = torch.ones((query_length, kv_length), dtype=torch.bool, device=mask.device).tril(kv_length - query_length)
    visible_keys = allowed.any(-2)
    key_prefix = torch.arange(kv_length, device=mask.device) < visible_keys.sum(-1, keepdim=True)
    expected = causal & visible_keys.unsqueeze(-2)
    valid = ((allowed == expected).all(-1) | ~allowed.any(-1)) & (visible_keys == key_prefix).all(-1, keepdim=True)
    # This opt-in validation has the same synchronization cost as Turbo's
    # prefix validation. Disable only for an already validated input pipeline.
    if not torch.equal(valid, torch.ones_like(valid)):
        raise ValueError("LightningIndexer supports causal-prefix masks with fully padded query rows only")


def configure(config, text_config) -> None:
    if config.relax_performance.qsa_indexer_implementation != "ascendc":
        return
    distributed = config.distributed
    if any(getattr(distributed, name, 1) != 1 for name in ("kv_allgather_parallel_size", "kcp_parallel_size")):
        raise ValueError("LightningIndexer supports Ulysses CP only")
    if distributed.ulysses_parallel_size > 1:
        if getattr(text_config, "mtp_num_layers", 0):
            raise ValueError("LightningIndexer with Ulysses CP requires MTP=0")
        try:
            from importlib import import_module

            import_module(".qwen4_attention", __package__)
        except ImportError as exc:
            raise ValueError("LightningIndexer with CP requires PR05 QSA attention integration") from exc


def install(model, config) -> None:
    options = config.relax_performance
    count = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen4ExpTextQSAIndexer":
            continue
        module.qsa_triton_validate_prefix_mask = options.qsa_triton_validate_prefix_mask
        if options.qsa_triton_query_chunk_size:
            module.qsa_triton_query_chunk_size = options.qsa_triton_query_chunk_size
        if options.qsa_indexer_implementation == "ascendc":
            from .performance_ops.qsa_indexer import qwen4_exp_qsa_indexer_forward_lightning

            module.qsa_id2mask_implementation = getattr(options, "qsa_id2mask_implementation", "eager")
            module.forward = MethodType(qwen4_exp_qsa_indexer_forward_lightning, module)
            count += 1
    if options.qsa_indexer_implementation == "ascendc" and not count:
        raise ValueError("Selected QSA indexer matched no model modules")
