# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Architecture checks and checkpoint names, independent of Megatron
imports."""

from collections.abc import Collection
from typing import Any


def validate_qwen2_vl_config(config: Any) -> tuple[Any, list[int]]:
    """Reject nearby architectures rather than treating Qwen2.5-VL as
    Qwen2-VL."""
    if getattr(config, "model_type", None) != "qwen2_vl":
        raise ValueError("Qwen2VLBridge requires model_type=qwen2_vl")
    text = getattr(config, "text_config", config)
    if getattr(text, "use_sliding_window", False):
        raise ValueError("Qwen2-VL sliding-window attention is not implemented by this integration")
    rope = getattr(text, "rope_parameters", None) or getattr(text, "rope_scaling", None) or {}
    sections = rope.get("mrope_section")
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
    if sections is None or len(sections) != 3 or any(int(value) <= 0 for value in sections):
        raise ValueError("Qwen2-VL requires three positive mrope_section values from its own HF config")
    if sum(sections) * 2 != head_dim:
        raise ValueError("Qwen2-VL mrope_section must cover the full attention head dimension")
    if getattr(config, "vision_config", None) is None:
        raise ValueError("Qwen2-VL requires its original vision_config")
    return text, list(sections)


def checkpoint_prefixes(names: Collection[str]) -> tuple[str, str]:
    """Support original checkpoints and HF's newer nested serialization
    layout."""
    if "model.language_model.embed_tokens.weight" in names:
        return "model.language_model", "model.visual"
    if "model.embed_tokens.weight" in names:
        return "model", "visual"
    raise ValueError("Qwen2-VL checkpoint does not contain a recognized language embedding key")


def parameter_mapping_specs(text: str, vision: str) -> list[tuple[str, str, Any]]:
    """Describe all language transforms and the unchanged Qwen2-VL vision
    tower."""
    layer = "language_model.decoder.layers.*"
    hf_layer = f"{text}.layers.*"
    simple = {
        "language_model.embedding.word_embeddings.weight": f"{text}.embed_tokens.weight",
        "language_model.output_layer.weight": "lm_head.weight",
        "language_model.decoder.final_layernorm.weight": f"{text}.norm.weight",
        f"{layer}.self_attention.linear_qkv.layer_norm_weight": f"{hf_layer}.input_layernorm.weight",
        f"{layer}.mlp.linear_fc1.layer_norm_weight": f"{hf_layer}.post_attention_layernorm.weight",
        f"{layer}.self_attention.linear_proj.weight": f"{hf_layer}.self_attn.o_proj.weight",
        f"{layer}.mlp.linear_fc2.weight": f"{hf_layer}.mlp.down_proj.weight",
    }
    specs = [("auto", megatron, hf) for megatron, hf in simple.items()]
    specs.append(("replicated", "visual.**", f"{vision}.**"))
    for suffix in ("weight", "bias"):
        specs.append(
            (
                "qkv",
                f"{layer}.self_attention.linear_qkv.{suffix}",
                {name: f"{hf_layer}.self_attn.{name}_proj.{suffix}" for name in ("q", "k", "v")},
            )
        )
    specs.append(
        (
            "gated_mlp",
            f"{layer}.mlp.linear_fc1.weight",
            {name: f"{hf_layer}.mlp.{name}_proj.weight" for name in ("gate", "up")},
        )
    )
    return specs
