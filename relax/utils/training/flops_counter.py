# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# FLOPS estimation adapted from verl (volcengine/verl) under Apache License 2.0.

import inspect

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# BF16 theoretical peak FLOPS per device (raw FLOPS).
_DEVICE_FLOPS = {
    "CPU": 448e9,
    "GB200": 2.5e15,
    "B200": 2.25e15,
    "GB300": 2.5e15,
    "B300": 2.25e15,
    "MI300X": 1307e12,
    "MI350X": 2300e12,
    "MI355X": 2500e12,
    "H100": 989e12,
    "H800": 989e12,
    "L20Y": 989e12,
    "H200": 989e12,
    "A100": 312e12,
    "A800": 312e12,
    "L40S": 362.05e12,
    "L40": 181.05e12,
    "A40": 149.7e12,
    "L20": 119.5e12,
    "H20": 148e12,
    "910B": 354e12,
    "Ascend910": 354e12,
    "RTX 3070 Ti": 43.5e12,
}


def _unit_convert(value: float, target_unit: str) -> float:
    units = ["B", "K", "M", "G", "T", "P"]
    if value <= 0 or value == float("inf"):
        return value
    ptr = 0
    while ptr < len(units) and units[ptr] != target_unit:
        value /= 1000
        ptr += 1
    return value


def get_device_peak_flops(unit: str = "T", device_name: str | None = None) -> float:
    """Get theoretical BF16 peak FLOPS for the current GPU.

    Returns ``float('inf')`` for unknown GPUs (MFU silently skipped).
    """
    if device_name is None:
        try:
            from relax.utils.device import get_device_properties

            device_name = get_device_properties().name
        except (AttributeError, ImportError):
            device_name = "CPU"

    flops = float("inf")
    for key, value in sorted(_DEVICE_FLOPS.items(), key=lambda x: len(x[0]), reverse=True):
        if key in device_name:
            flops = value
            break

    if flops == float("inf"):
        logger.warning("Unknown GPU '%s' — MFU will not be reported.", device_name)

    return _unit_convert(flops, unit)


# ---------------------------------------------------------------------------
# Per-model-type FLOPS estimators (6N formula, fwd+bwd inclusive)
#
# Each function returns achieved TFLOPS = total_flops / delta_time / 1e12.
# The "6" factor = 2 (matmul multiply+add) × 3 (1x fwd + 2x bwd).
# Attention uses "6" with causal mask (/2 for QK^T and /2 for A*V, cancels
# one factor of 2), giving 6 * S^2 * D * H per layer for causal models,
# or 12 for full-attention (ViT).
# ---------------------------------------------------------------------------


def _estimate_qwen2_flops(config, tokens_sum, batch_seqlens, delta_time):
    """Dense transformer: Qwen2, Qwen3, LLaMA, Mistral, etc."""
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size

    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_N_flops = 6 * dense_N * tokens_sum

    seqlen_square_sum = sum(s * s for s in batch_seqlens)
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    return (dense_N_flops + attn_qkv_flops) / delta_time / 1e12


def _estimate_qwen2_moe_flops(config, tokens_sum, batch_seqlens, delta_time):
    """MoE transformer: Qwen2-MoE, Qwen3-MoE."""
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    moe_intermediate_size = config.moe_intermediate_size
    moe_topk = config.num_experts_per_tok
    num_experts = config.num_experts

    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    moe_mlp_N = hidden_size * moe_topk * moe_intermediate_size * 3 + hidden_size * num_experts
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N = (moe_mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_N_flops = 6 * dense_N * tokens_sum

    seqlen_square_sum = sum(s * s for s in batch_seqlens)
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    return (dense_N_flops + attn_qkv_flops) / delta_time / 1e12


def _estimate_qwen3_vl_flops(config, tokens_sum, batch_seqlens, delta_time, **kargs):
    """Qwen3-VL (dense text + ViT)."""
    hidden_size = config.text_config.hidden_size
    vocab_size = config.text_config.vocab_size
    num_hidden_layers = config.text_config.num_hidden_layers
    num_key_value_heads = config.text_config.num_key_value_heads
    num_attention_heads = config.text_config.num_attention_heads
    intermediate_size = config.text_config.intermediate_size

    head_dim = hidden_size // num_attention_heads
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_N_flops = 6 * dense_N * tokens_sum

    seqlen_square_sum = sum(s * s for s in batch_seqlens)
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    images_seqlens = kargs.get("images_seqlens", None)
    vit_flops = _estimate_qwen3_vit_flop(images_seqlens, config.vision_config) if images_seqlens is not None else 0

    return (dense_N_flops + attn_qkv_flops + vit_flops) / delta_time / 1e12


def _estimate_qwen3_vl_moe_flops(config, tokens_sum, batch_seqlens, delta_time, **kargs):
    """Qwen3-VL-MoE (MoE text + ViT)."""
    hidden_size = config.text_config.hidden_size
    vocab_size = config.text_config.vocab_size
    num_hidden_layers = config.text_config.num_hidden_layers
    num_key_value_heads = config.text_config.num_key_value_heads
    num_attention_heads = config.text_config.num_attention_heads
    moe_intermediate_size = config.text_config.moe_intermediate_size
    moe_num_expert = config.text_config.num_experts
    moe_topk = config.text_config.num_experts_per_tok

    head_dim = getattr(
        config.text_config, "head_dim", config.text_config.hidden_size // config.text_config.num_attention_heads
    )
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    moe_gate_N = hidden_size * moe_num_expert
    moe_expertmlp_N = hidden_size * moe_intermediate_size * moe_topk * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    moe_N = (moe_gate_N + moe_expertmlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_N_flops = 6 * moe_N * tokens_sum

    seqlen_square_sum = sum(s * s for s in batch_seqlens)
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    images_seqlens = kargs.get("images_seqlens", None)
    vit_flops = _estimate_qwen3_vit_flop(images_seqlens, config.vision_config) if images_seqlens is not None else 0

    return (dense_N_flops + attn_qkv_flops + vit_flops) / delta_time / 1e12


def _estimate_qwen3_omni_moe_flops(config, tokens_sum, batch_seqlens, delta_time, **kargs):
    """Qwen3-Omni-MoE (MoE text + ViT + Audio encoder)."""
    thinker = config.thinker_config if hasattr(config, "thinker_config") else config
    text_config = thinker.text_config if hasattr(thinker, "text_config") else thinker

    hidden_size = text_config.hidden_size
    vocab_size = text_config.vocab_size
    num_hidden_layers = text_config.num_hidden_layers
    num_key_value_heads = text_config.num_key_value_heads
    num_attention_heads = text_config.num_attention_heads
    moe_intermediate_size = text_config.moe_intermediate_size
    moe_num_expert = text_config.num_experts
    moe_topk = text_config.num_experts_per_tok

    head_dim = getattr(text_config, "head_dim", hidden_size // num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    moe_gate_N = hidden_size * moe_num_expert
    moe_expertmlp_N = hidden_size * moe_intermediate_size * moe_topk * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    moe_N = (moe_gate_N + moe_expertmlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_N_flops = 6 * moe_N * tokens_sum

    seqlen_square_sum = sum(s * s for s in batch_seqlens)
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    images_seqlens = kargs.get("images_seqlens", None)
    vision_config = getattr(thinker, "vision_config", None)
    vit_flops = _estimate_qwen3_vit_flop(images_seqlens, vision_config) if images_seqlens else 0

    audio_seqlens = kargs.get("audio_seqlens", None)
    audio_config = getattr(thinker, "audio_config", None)
    audio_flops = _estimate_qwen3_audio_flop(audio_seqlens, audio_config) if audio_seqlens else 0

    return (dense_N_flops + attn_qkv_flops + vit_flops + audio_flops) / delta_time / 1e12


def _get_audio_encoder_seqlens(feature_lengths, n_window=100):
    """Convert raw mel feature lengths to audio encoder transformer seq
    lengths.

    Mirrors ``_get_feat_extract_output_lengths`` in the Qwen3-Omni codebase.
    """
    result = []
    for fl in feature_lengths:
        leave = fl % n_window
        feat = (leave - 1) // 2 + 1
        out = ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (fl // n_window) * 13
        result.append(out)
    return result


def _estimate_qwen3_audio_flop(audio_seqlens, config):
    """Whisper-style audio encoder FLOPS (Qwen3-Omni).

    ``audio_seqlens`` are raw mel feature lengths (pre-conv). They are converted
    to transformer sequence lengths internally.

    Architecture: 2x Conv1d stem -> N transformer layers (windowed attention) ->
    downsampling conv -> output projection.
    """
    if config is None or not audio_seqlens:
        return 0

    d_model = config.d_model
    encoder_layers = getattr(config, "encoder_layers", config.num_hidden_layers)
    num_heads = config.encoder_attention_heads
    ffn_dim = config.encoder_ffn_dim
    num_mel_bins = config.num_mel_bins
    output_dim = getattr(config, "output_dim", d_model)
    n_window = getattr(config, "n_window", 100)
    head_dim = d_model // num_heads

    # Convert raw mel lengths to transformer seq lengths
    encoder_seqlens = _get_audio_encoder_seqlens(audio_seqlens, n_window)
    tokens_sum = sum(encoder_seqlens)

    # Conv stem: conv1 (mel->d_model, k=3) + conv2 (d_model->d_model, k=3, stride=2)
    conv_N = num_mel_bins * d_model * 3 + d_model * d_model * 3

    # Transformer layers: self-attention (windowed) + FFN (GELU, not GLU -> 2 linear layers)
    attn_linear_N = d_model * (4 * d_model)
    ffn_N = d_model * ffn_dim * 2
    transformer_N = (attn_linear_N + ffn_N) * encoder_layers

    # Output projection: d_model -> output_dim
    output_proj_N = d_model * output_dim

    dense_N = conv_N + transformer_N + output_proj_N
    dense_N_flops = 6 * dense_N * tokens_sum

    # Windowed full attention (no causal mask -> coefficient 12)
    # Each token attends to at most n_window tokens
    effective_seqlen_sq_sum = sum(s * min(s, n_window) for s in encoder_seqlens)
    attn_qkv_flops = 12 * effective_seqlen_sq_sum * head_dim * num_heads * encoder_layers

    return dense_N_flops + attn_qkv_flops


def _estimate_qwen3_vit_flop(images_seqlens, config):
    if config is None:
        return 0
    tokens_sum = sum(images_seqlens)

    num_heads = config.num_heads
    depth = config.depth
    dim = config.hidden_size
    mlp_hidden_dim = config.intermediate_size
    out_hidden_size = config.out_hidden_size
    spatial_merge_size = config.spatial_merge_size
    head_dim = dim // num_heads

    patch_embed_N = dim * config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size
    mlp_N = dim * mlp_hidden_dim * 2  # no GLU in ViT
    attn_linear_N = dim * (4 * dim)
    merger_N = (out_hidden_size + (dim * (spatial_merge_size**2))) * (dim * (spatial_merge_size**2))
    deepstack_visual_indexes = getattr(config, "deepstack_visual_indexes", None)
    deepstack_merger_N = merger_N * len(deepstack_visual_indexes) if deepstack_visual_indexes is not None else 0
    dense_N = patch_embed_N + (mlp_N + attn_linear_N) * depth + deepstack_merger_N + merger_N
    dense_N_flops = 6 * dense_N * tokens_sum

    # ViT uses full attention (no causal mask) -> coefficient 12 instead of 6
    seqlen_square_sum = sum(s * s for s in images_seqlens)
    attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_heads * depth

    return dense_N_flops + attn_qkv_flops


def _count_qwen3_5_layer_types(config):
    layer_types = getattr(config, "layer_types", None)
    if layer_types:
        num_full_attn_layers = sum(layer_type == "full_attention" for layer_type in layer_types)
        num_linear_attn_layers = sum(layer_type == "linear_attention" for layer_type in layer_types)
        return num_full_attn_layers, num_linear_attn_layers

    full_attention_interval = getattr(config, "full_attention_interval", 4)
    num_full_attn_layers = sum(
        not bool((layer_idx + 1) % full_attention_interval) for layer_idx in range(config.num_hidden_layers)
    )
    return num_full_attn_layers, config.num_hidden_layers - num_full_attn_layers


def _compute_qwen3_5_hybrid_attn_params(config):
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    num_full_attn_layers, num_linear_attn_layers = _count_qwen3_5_layer_types(config)

    full_attn_linear_N = hidden_size * (2 * q_size + k_size + v_size + q_size)

    linear_k_size = config.linear_num_key_heads * config.linear_key_head_dim
    linear_v_size = config.linear_num_value_heads * config.linear_value_head_dim
    linear_attn_linear_N = hidden_size * (2 * linear_k_size + 3 * linear_v_size + 2 * config.linear_num_value_heads)
    conv_N = config.linear_conv_kernel_dim * (2 * linear_k_size + linear_v_size)

    attn_linear_N = full_attn_linear_N * num_full_attn_layers
    attn_linear_N += (linear_attn_linear_N + conv_N) * num_linear_attn_layers

    return attn_linear_N, num_full_attn_layers, num_linear_attn_layers, head_dim, num_attention_heads


def _compute_qwen3_5_gdn_recurrence_flops(config, tokens_sum, num_linear_attn_layers):
    return (
        15
        * config.linear_key_head_dim
        * config.linear_value_head_dim
        * config.linear_num_value_heads
        * tokens_sum
        * num_linear_attn_layers
    )


def _estimate_qwen3_5_flops(config, tokens_sum, batch_seqlens, delta_time, **kargs):
    """Qwen3.5 hybrid attention (full attention + GatedDeltaNet linear
    attention)."""
    text_config = config.text_config if hasattr(config, "text_config") else config
    hidden_size = text_config.hidden_size
    vocab_size = text_config.vocab_size
    num_hidden_layers = text_config.num_hidden_layers

    attn_linear_N, num_full_attn_layers, num_linear_attn_layers, head_dim, num_attention_heads = (
        _compute_qwen3_5_hybrid_attn_params(text_config)
    )

    if hasattr(text_config, "num_experts"):
        moe_gate_N = hidden_size * text_config.num_experts
        moe_expertmlp_N = hidden_size * text_config.moe_intermediate_size * text_config.num_experts_per_tok * 3
        moe_sharedexpertmlp_N = hidden_size * text_config.shared_expert_intermediate_size * 3
        moe_sharedexpert_gate_N = hidden_size
        mlp_N = (moe_gate_N + moe_expertmlp_N + moe_sharedexpertmlp_N + moe_sharedexpert_gate_N) * num_hidden_layers
    else:
        mlp_N = hidden_size * text_config.intermediate_size * 3 * num_hidden_layers

    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N_flops = 6 * (mlp_N + attn_linear_N + emd_and_lm_head_N) * tokens_sum

    seqlen_square_sum = sum(s * s for s in batch_seqlens)
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_full_attn_layers

    gdn_recurrence_flops = _compute_qwen3_5_gdn_recurrence_flops(text_config, tokens_sum, num_linear_attn_layers)

    images_seqlens = kargs.get("images_seqlens", None)
    if images_seqlens is not None and hasattr(config, "vision_config"):
        vit_flops = _estimate_qwen3_vit_flop(images_seqlens, config.vision_config)
    else:
        vit_flops = 0

    flops_all_token = dense_N_flops + attn_qkv_flops + gdn_recurrence_flops + vit_flops
    return flops_all_token / delta_time / 1e12


def _estimate_deepseek_v3_flops(config, tokens_sum, batch_seqlens, delta_time):
    """DeepSeek-V3 (MLA attention + MoE)."""
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    moe_intermediate_size = config.moe_intermediate_size
    num_hidden_layers = config.num_hidden_layers
    first_k_dense_replace = config.first_k_dense_replace
    num_query_heads = config.num_attention_heads
    moe_num_expert = config.n_routed_experts
    moe_topk = config.num_experts_per_tok
    share_expert_num = config.n_shared_experts

    moe_gate_N = hidden_size * moe_num_expert
    moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 3

    # MLA attention linear params
    attn_linear_N = 0
    q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    if config.q_lora_rank is None:
        attn_linear_N += hidden_size * num_query_heads * q_head_dim
    else:
        attn_linear_N += hidden_size * config.q_lora_rank
        attn_linear_N += num_query_heads * q_head_dim * config.q_lora_rank

    attn_linear_N += hidden_size * (config.kv_lora_rank + config.qk_rope_head_dim)
    attn_linear_N += num_query_heads * (q_head_dim - config.qk_rope_head_dim + config.v_head_dim) * config.kv_lora_rank
    attn_linear_N += num_query_heads * config.v_head_dim * hidden_size

    emd_and_lm_head_N = vocab_size * hidden_size * 2
    moe_N = (
        (moe_gate_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers - first_k_dense_replace)
        + (hidden_size * config.intermediate_size * 3 + attn_linear_N) * first_k_dense_replace
        + emd_and_lm_head_N
    )
    dense_N_flops = 6 * moe_N * tokens_sum

    seqlen_square_sum = sum(s * s * num_hidden_layers for s in batch_seqlens)
    # MLA causal: 3 * 2 * seq^2 * dim / 2 = 3 * seq^2 * dim per component
    attn_qkv_flops = 3 * seqlen_square_sum * (q_head_dim + config.v_head_dim) * num_query_heads

    return (dense_N_flops + attn_qkv_flops) / delta_time / 1e12


def _estimate_fallback_flops(config, tokens_sum, batch_seqlens, delta_time):
    """Fallback for unknown model types using standard dense transformer
    formula."""
    text_config = config.text_config if hasattr(config, "text_config") else config
    required = ("hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads", "intermediate_size")
    if not all(hasattr(text_config, attr) for attr in required):
        return 0

    hidden_size = text_config.hidden_size
    vocab_size = text_config.vocab_size
    num_hidden_layers = text_config.num_hidden_layers
    num_attention_heads = text_config.num_attention_heads
    intermediate_size = text_config.intermediate_size
    num_key_value_heads = getattr(text_config, "num_key_value_heads", num_attention_heads)
    head_dim = getattr(text_config, "head_dim", hidden_size // num_attention_heads)

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_N_flops = 6 * dense_N * tokens_sum

    seqlen_square_sum = sum(s * s for s in batch_seqlens)
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    return (dense_N_flops + attn_qkv_flops) / delta_time / 1e12


_ESTIMATE_FUNC = {
    # Dense transformers (SwiGLU + GQA)
    "qwen2": _estimate_qwen2_flops,
    "qwen3": _estimate_qwen2_flops,
    "qwen3_5": _estimate_qwen3_5_flops,
    "llama": _estimate_qwen2_flops,
    "mistral": _estimate_qwen2_flops,
    "glm4": _estimate_qwen2_flops,
    "glm4v": _estimate_qwen2_flops,
    "glm46v": _estimate_qwen2_flops,
    "glm": _estimate_qwen2_flops,
    "minicpmv": _estimate_qwen2_flops,
    "minicpmo": _estimate_qwen2_flops,
    "seed_oss": _estimate_qwen2_flops,
    "mimo": _estimate_qwen2_flops,
    # MoE transformers (SwiGLU + GQA + router gate + topk experts)
    "qwen2_moe": _estimate_qwen2_moe_flops,
    "qwen3_moe": _estimate_qwen2_moe_flops,
    "qwen3_5_moe": _estimate_qwen3_5_flops,
    "qwen3_next": _estimate_qwen2_moe_flops,
    "qwen3_omni_moe": _estimate_qwen3_omni_moe_flops,
    "glm4_moe": _estimate_qwen2_moe_flops,
    "glm4v_moe": _estimate_qwen2_moe_flops,
    # MLA + MoE (DeepSeek-V3 style: MLA attention + shared experts + dense-replace)
    "deepseek_v3": _estimate_deepseek_v3_flops,
    "glm4_moe_lite": _estimate_deepseek_v3_flops,
    "glm_moe_dsa": _estimate_deepseek_v3_flops,
    # Vision-language
    "qwen2_vl": _estimate_qwen2_flops,
    "qwen2_5_vl": _estimate_qwen2_flops,
    "qwen3_vl": _estimate_qwen3_vl_flops,
    "qwen3_vl_moe": _estimate_qwen3_vl_moe_flops,
}


class FlopsCounter:
    """Estimate training FLOPS and MFU based on HuggingFace model config.

    Example::

        counter = FlopsCounter(hf_config)
        estimated_tflops, peak_tflops = counter.estimate(batch_seqlens, delta_time)
        mfu = estimated_tflops / peak_tflops / world_size
    """

    def __init__(self, config):
        if hasattr(config, "model_type") and config.model_type not in _ESTIMATE_FUNC:
            logger.warning(
                "Unsupported model_type '%s' for FLOPS estimation — falling back to dense transformer formula. "
                "Supported: %s.",
                config.model_type,
                list(_ESTIMATE_FUNC.keys()),
            )
        self.config = config

    def estimate(self, batch_seqlens, delta_time, **kwargs):
        """Return (estimated_tflops, peak_device_tflops).

        ``estimated_tflops`` includes fwd+bwd (6N formula).
        """
        tokens_sum = sum(batch_seqlens)
        model_type = getattr(self.config, "model_type", None)
        func = _ESTIMATE_FUNC.get(model_type, _estimate_fallback_flops)
        sig = inspect.signature(func)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            estimated_tflops = func(self.config, tokens_sum, batch_seqlens, delta_time, **kwargs)
        else:
            estimated_tflops = func(self.config, tokens_sum, batch_seqlens, delta_time)
        peak_tflops = get_device_peak_flops(unit="T")
        return estimated_tflops, peak_tflops
