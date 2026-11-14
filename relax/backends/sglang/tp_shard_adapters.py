# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Checkpoint-name mapping used by Qwen SGLang weight loaders."""


def target_parameter(model, hf_name: str) -> tuple[str, str | int | None] | None:
    """Return the native parameter name and fused shard selected by
    load_weights."""
    name = hf_name
    model_name = type(model).__name__
    if not model_name.startswith("Qwen3") or type(model).__module__ not in (
        "sglang.srt.models.qwen3",
        "sglang.srt.models.qwen3_5",
        "sglang.srt.models.qwen3_vl",
    ):
        return None
    if name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model.") :]
    elif name.startswith("model.visual."):
        name = "visual." + name[len("model.visual.") :]
    if "Qwen3_5" in model_name and ".self_attn." in name:
        name = name.replace(".self_attn.", ".")
    if name.startswith("visual."):
        return name.replace(".attn.qkv.", ".attn.qkv_proj."), None
    if ".experts." in name:
        return name, None
    for original, fused, shard in (
        (".gate_proj.", ".gate_up_proj.", 0),
        (".up_proj.", ".gate_up_proj.", 1),
        (".q_proj.", ".qkv_proj.", "q"),
        (".k_proj.", ".qkv_proj.", "k"),
        (".v_proj.", ".qkv_proj.", "v"),
    ):
        if original in name:
            return name.replace(original, fused), shard
    # Qwen3.5 GDN packed projections need their own split semantics. They
    # remain on the native loader until that adapter is explicitly qualified.
    return name, None


def needs_native_refresh(model) -> bool:
    """Qwen3.5 MoE rebuilds its routed-expert LazyValue on an empty load."""
    return (
        type(model).__module__ == "sglang.srt.models.qwen3_5"
        and type(model).__name__ == "Qwen3_5MoeForConditionalGeneration"
        and getattr(model, "_routed_experts_weights_of_layer", None) is not None
    )
