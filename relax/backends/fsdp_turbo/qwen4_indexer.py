# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from types import MethodType


def validate_lightning_mask(mask, query_length, kv_length):
    """Lightning routing supports causal right-padded prefixes and empty
    queries."""
    import torch

    if mask is None or mask.ndim != 4 or mask.shape[-2:] != (query_length, kv_length):
        raise ValueError("LightningIndexer requires a complete four-dimensional causal mask")
    key = None if torch.is_inference(mask) else (mask._version, query_length, kv_length)
    if key is not None and getattr(mask, "_relax_prefix_validation", None) == key:
        return
    allowed = mask if mask.dtype == torch.bool else mask >= 0
    causal = torch.ones((query_length, kv_length), dtype=torch.bool, device=mask.device).tril(kv_length - query_length)
    visible_keys = allowed.any(-2)
    key_prefix = torch.arange(kv_length, device=mask.device) < visible_keys.sum(-1, keepdim=True)
    expected = causal & visible_keys.unsqueeze(-2)
    valid = ((allowed == expected).all(-1) | ~allowed.any(-1)) & (visible_keys == key_prefix).all(-1, keepdim=True)
    # Unknown device masks require a real check; NPU _assert_async falls back
    # to CPU too. Known SFT masks are certified at their construction boundary.
    if not bool(valid.all()):
        raise ValueError("LightningIndexer supports causal-prefix masks with fully padded query rows only")
    # The input owns this result: no retained device allocation or global mask cache.
    # In-place mutation invalidates it through Tensor._version.
    mask._relax_prefix_validation = key


def _prepare_lightning_mask(module, args, kwargs):
    import torch

    if not getattr(module, "_relax_validate_lightning_mask", True):
        return args, kwargs
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

    def value(name, position):
        return args[position] if len(args) > position else kwargs.get(name)

    mask = value("attention_mask", 1)
    embeds = value("inputs_embeds", 4)
    positions = value("position_ids", 2)
    cache = value("past_key_values", 3)
    use_cache = value("use_cache", 5)
    if use_cache is None:
        use_cache = getattr(module.config, "use_cache", False)
    if embeds is None or cache is not None or use_cache:
        return args, kwargs
    length = embeds.shape[1]
    if isinstance(mask, dict):
        validate_lightning_mask(mask.get("indexed_attention"), length, length)
        return args, kwargs
    if mask is not None and mask.ndim == 4:
        validate_lightning_mask(mask, length, length)
        return args, kwargs
    if mask is None or mask.ndim != 2:
        return args, kwargs
    # Match the pinned HF text model's mask inputs. Custom/packed positions
    # still require full validation, even when the 2D validity came from SFT.
    text_positions = positions
    if positions is not None and positions.ndim == 3:
        text_positions = positions[0] if positions.shape[0] in (1, 4) else None
    mask_args = dict(
        config=module.config,
        inputs_embeds=embeds,
        attention_mask=mask,
        past_key_values=None,
        position_ids=text_positions,
        allow_is_causal_skip=False,
    )
    causal = create_causal_mask(**mask_args)
    linear = create_recurrent_attention_mask(**mask_args)
    provenance = getattr(mask, "_relax_sft_prefix", None)
    trusted = (
        provenance is not None
        and not torch.is_inference(mask)
        and provenance[0] == mask._version
        and len(provenance[1]) == mask.shape[0]
        and mask.shape == embeds.shape[:2]
        and text_positions is None
    )
    if trusted and causal is not None and causal.ndim == 4 and not torch.is_inference(causal):
        causal._relax_prefix_validation = (causal._version, length, length)
    else:
        validate_lightning_mask(causal, length, length)
    mapping = {"indexed_attention": causal, "linear_attention": linear}
    if len(args) > 1:
        args = list(args)
        args[1] = mapping
        return tuple(args), kwargs
    return args, {**kwargs, "attention_mask": mapping}


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
        if type(module).__name__ == "Qwen4ExpTextModel":
            module._relax_validate_lightning_mask = options.qsa_triton_validate_prefix_mask
        if (
            type(module).__name__ == "Qwen4ExpTextModel"
            and options.qsa_indexer_implementation == "ascendc"
            and options.qsa_triton_validate_prefix_mask
            and config.distributed.ulysses_parallel_size == 1
            and not getattr(module, "_relax_lightning_mask_hook", False)
        ):
            module.register_forward_pre_hook(_prepare_lightning_mask, with_kwargs=True)
            module._relax_lightning_mask_hook = True
        if type(module).__name__ != "Qwen4ExpTextQSAIndexer":
            continue
        module.qsa_triton_validate_prefix_mask = options.qsa_triton_validate_prefix_mask
        module.qsa_triton_query_chunk_size = options.qsa_triton_query_chunk_size or None
        if options.qsa_indexer_implementation == "ascendc":
            from .performance_ops.qsa_indexer import qwen4_exp_qsa_indexer_forward_lightning

            module.qsa_id2mask_implementation = getattr(options, "qsa_id2mask_implementation", "eager")
            module.forward = MethodType(qwen4_exp_qsa_indexer_forward_lightning, module)
            count += 1
    if options.qsa_indexer_implementation == "ascendc" and not count:
        raise ValueError("Selected QSA indexer matched no model modules")
