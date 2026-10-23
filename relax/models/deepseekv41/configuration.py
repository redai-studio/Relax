# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""HF configuration compatibility for the V4.1 text-training architecture."""

from argparse import Namespace
from copy import deepcopy
from typing import Any


DeepSeekV41Config: type[Any]
DeepSeekV41TextConfig: type[Any]


V41_CONFIG_FIELDS = (
    "kv_source_layer_ids",
    "index_source_layer_ids",
    "candidate_source_layer_id",
    "candidate_topk_blocks",
    "candidate_block_size",
    "engram_layer_ids",
    "engram_num_embeddings",
    "engram_max_ngram_size",
    "engram_vocab_size",
    "engram_n_heads",
    "engram_head_dim",
    "engram_pad_token_id",
    "engram_compressed_vocab_size",
)

# CLI defaults must not replace the architecture obtained from text_config.
ARCHITECTURE_KEYS = {
    "num_layers",
    "moe_layer_freq",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_head_dim",
    "qk_pos_emb_head_dim",
    "v_head_dim",
    "rotary_scaling_factor",
    "rotary_base",
    "moe_router_pre_softmax",
    "moe_router_enable_expert_bias",
    "moe_router_bias_update_rate",
    "moe_router_dtype",
    "moe_router_load_balancing_type",
    "moe_aux_loss_coeff",
    "moe_shared_expert_intermediate_size",
    "moe_router_topk",
    "moe_router_num_groups",
    "moe_router_group_topk",
    "moe_router_topk_scaling_factor",
    "moe_router_score_function",
    "moe_ffn_hidden_size",
    "activation_func_clamp_shared_expert",
    "mtp_num_layers",
}


def hf_config_dict(config: Any) -> dict[str, Any]:
    """Copy a config without changing the source used by subsequent AutoBridge
    calls."""
    if isinstance(config, dict):
        return deepcopy(config)
    if hasattr(config, "to_dict"):
        return deepcopy(config.to_dict())
    return deepcopy(vars(config))


def normalize_text_config(config: Any) -> dict[str, Any]:
    """Accept both composite checkpoints and flattened text checkpoints."""
    original = hf_config_dict(config)
    if original.get("model_type") not in ("deepseek_v41", "deepseek_v41_text"):
        raise ValueError("DeepSeek V4.1 Bridge requires a deepseek_v41 checkpoint configuration.")
    text = hf_config_dict(original["text_config"]) if original.get("text_config") is not None else original
    for key in ("dtype", "torch_dtype", "bos_token_id", "eos_token_id", "pad_token_id", "quantization_config"):
        if key in original and (key not in text or text[key] is None):
            text[key] = deepcopy(original[key])
    # This private view is for provider construction, not a rewrite of config.json.
    text["model_type"] = "deepseek_v41"
    text["architectures"] = ["DeepseekV41ForCausalLM"]
    return text


def _init_hf_config(self: Any, text_config: Any = None, **kwargs: Any) -> None:
    from transformers import PretrainedConfig

    kwargs.setdefault("architectures", ["DeepseekV41ForCausalLM"])
    # Transformers 5 resolves RoPE before retaining unknown architecture fields.
    # Its configuration-only fallback therefore needs those fields beforehand.
    for name, value in kwargs.items():
        setattr(self, name, deepcopy(value))
    PretrainedConfig.__init__(self, **kwargs)
    if text_config is not None:
        config_class = globals()["DeepSeekV41TextConfig"]
        self.text_config = config_class(**hf_config_dict(text_config))


def register_hf_config(*, for_training: bool = False) -> None:
    """Register configuration-only fallbacks; no HF model implementation is
    needed.

    Keep Transformers optional until this model package is imported. Binding
    the fallback classes at module scope also makes their instances pickleable.
    SGLang registers its own flattened config for rollout; only training entry
    points explicitly restore our composite-aware registration after those
    imports.
    """
    from transformers import AutoConfig, PretrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    for model_type, name in (
        ("deepseek_v41_text", "DeepSeekV41TextConfig"),
        ("deepseek_v41", "DeepSeekV41Config"),
    ):
        config_class = globals().get(name)
        if config_class is None and model_type in CONFIG_MAPPING:
            candidate = CONFIG_MAPPING[model_type]
            if candidate.__module__.startswith("transformers.models."):
                config_class = candidate
        if config_class is None:
            attributes: dict[str, Any] = {"model_type": model_type, "__module__": __name__}
            attributes["__init__"] = _init_hf_config
            config_class = type(name, (PretrainedConfig,), attributes)
        globals()[name] = config_class
        if for_training or model_type not in CONFIG_MAPPING:
            AutoConfig.register(model_type, config_class, exist_ok=True)


def configure_runtime(provider: Any, args: Namespace) -> None:
    """Keep unsupported auxiliary architectures out of the text-only
    provider."""
    from .execution_config import configure_execution_runtime

    if getattr(args, "mtp_num_layers", None):
        raise ValueError("DeepSeek V4.1 supports text-only training; MTP/DSpark training is unsupported.")
    provider.mtp_num_layers = None
    configure_execution_runtime(provider, args)
