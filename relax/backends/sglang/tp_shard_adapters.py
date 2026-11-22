# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Checkpoint names for native copy loaders; conversion stays in adapters."""

import inspect
import sys


def full_copy_module(module, param) -> bool:
    """Recognize native full-parameter copies; forward scaling stays native."""
    if (
        getattr(module, "weight", None) is not param
        or getattr(module, "use_presharded_weights", False)
        or getattr(param, "packed_dim", None) is not None
    ):
        return False
    embedding = sys.modules.get("sglang.srt.models.gemma3_causal")
    if type(module) is getattr(embedding, "Gemma3TextScaledWordEmbedding", None):
        return not hasattr(param, "weight_loader") and not hasattr(module, "quant_method")
    linear = sys.modules.get("sglang.srt.layers.linear")
    if type(module) is not getattr(linear, "ReplicatedLinear", None):
        return False
    loader = getattr(param, "weight_loader", None)
    unquant = sys.modules.get("sglang.srt.layers.quantization.unquant")
    return (
        getattr(loader, "__self__", None) is module
        and getattr(loader, "__func__", None) is type(module).weight_loader
        and type(getattr(module, "quant_method", None)) is getattr(unquant, "UnquantizedLinearMethod", None)
        and getattr(param, "input_dim", None) == 1
        and getattr(param, "output_dim", None) == 0
    )


def packed_copy_loader(module, param):
    """Qualify the native tuple-splitting wrapper without changing its
    loader."""
    loader = getattr(param, "weight_loader", None)
    if not inspect.isfunction(loader):
        return None
    closure = inspect.getclosurevars(loader).nonlocals
    cls, owner, original = (closure.get(key) for key in ("cls", "module", "original_weight_loader"))
    native = sys.modules.get("sglang.srt.models.qwen3_5")
    if cls is None or cls is not getattr(native, "Qwen3_5GatedDeltaNet", None) or owner is not module:
        return None
    reference = cls._make_packed_weight_loader(module, original)
    return original if loader.__code__ is reference.__code__ else None


def _native_copy_name(model, name: str) -> str:
    """Resolve a text-only namespace only when the native parameter exists."""
    if name.startswith("model.language_model."):
        candidate = "model." + name[len("model.language_model.") :]
        try:
            model.get_parameter(name)
        except AttributeError:
            try:
                model.get_parameter(candidate)
            except AttributeError:
                pass
            else:
                return candidate
    return name


def target_parameter(model, hf_name: str) -> tuple[str, str | int | tuple[int, ...] | None] | None:
    """Return the native parameter name and fused shard selected by
    load_weights."""
    name = hf_name
    remap = getattr(model, "remap_weight_name_to_dpsk_hf_format", None)
    if remap is not None:
        name = remap(name)
    model_name = type(model).__name__
    qwen = model_name.startswith("Qwen3") and type(model).__module__ in (
        "sglang.srt.models.qwen3",
        "sglang.srt.models.qwen3_5",
        "sglang.srt.models.qwen3_vl",
    )
    if not qwen and not (
        name.rpartition(".")[0].endswith(
            (
                ".gate_proj",
                ".up_proj",
                ".down_proj",
                ".o_proj",
                ".embed_tokens",
                ".embed_tokens_per_layer",
                ".per_layer_model_projection",
                ".lm_head",
            )
        )
        or name == "lm_head.weight"
    ):
        # MLA projections and AttnRes weights feed derived native caches.
        # Packed/quantized families and other transforms remain native.
        return None
    if qwen and name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model.") :]
    elif qwen and name.startswith("model.visual."):
        name = "visual." + name[len("model.visual.") :]
    if qwen and "Qwen3_5" in model_name and ".self_attn." in name:
        name = name.replace(".self_attn.", ".")
    if qwen and name.startswith("visual."):
        return name.replace(".attn.qkv.", ".attn.qkv_proj."), None
    if ".experts." in name:
        return (name, None) if qwen else None
    if qwen and ".linear_attn." in name:
        for original, fused, shard in (
            (".in_proj_qkv.", ".in_proj_qkvz.", (0, 1, 2)),
            (".in_proj_z.", ".in_proj_qkvz.", 3),
            (".in_proj_b.", ".in_proj_ba.", 0),
            (".in_proj_a.", ".in_proj_ba.", 1),
        ):
            if original in name:
                return name.replace(original, fused), shard
    for original, fused, shard in (
        (".gate_proj.", ".gate_up_proj.", 0),
        (".up_proj.", ".gate_up_proj.", 1),
        (".q_proj.", ".qkv_proj.", "q"),
        (".k_proj.", ".qkv_proj.", "k"),
        (".v_proj.", ".qkv_proj.", "v"),
    ):
        if original in name:
            # Other Q/K/V fusions have additional gates or packed layouts.
            if not qwen and shard in ("q", "k", "v"):
                return None
            return _native_copy_name(model, name.replace(original, fused)), shard
    return _native_copy_name(model, name), None


def needs_native_refresh(model) -> bool:
    """Qwen3.5 MoE rebuilds its routed-expert LazyValue on an empty load."""
    return (
        type(model).__module__ == "sglang.srt.models.qwen3_5"
        and type(model).__name__ == "Qwen3_5MoeForConditionalGeneration"
        and getattr(model, "_routed_experts_weights_of_layer", None) is not None
    )
