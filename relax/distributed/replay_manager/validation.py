# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validation for frozen visual-embedding replay."""

from typing import Any


def validate_mm_embedding_replay_args(args: Any, *, is_offline: bool) -> None:
    """Validate frozen CP1 visual consumers before launching workers."""
    from relax.utils.misc import get_hf_config

    config = get_hf_config(args.hf_checkpoint)
    vision = getattr(config, "vision_config", None)
    text = getattr(config, "text_config", config)
    unsupported = []
    hidden_size = {"qwen4_exp": 2560, "qwen3_5_moe": 2048}.get(getattr(config, "model_type", None))
    if hidden_size is None or vision is None:
        unsupported.append("requires a supported multimodal Qwen3.5 or Qwen3.8 checkpoint")
    if vision is not None and (
        getattr(vision, "out_hidden_size", None) != hidden_size
        or getattr(vision, "spatial_merge_size", None) != 2
        or getattr(vision, "deepstack_visual_indexes", None)
        or getattr(text, "hidden_size", None) != hidden_size
    ):
        unsupported.append("requires the model visual width and R4 embeddings without DeepStack")
    asynchronous = getattr(args, "fully_async", False) or getattr(args, "hybrid", False)
    if (
        is_offline
        or getattr(args, "train_backend", None) != "megatron"
        or not (getattr(args, "colocate", False) or asynchronous)
    ):
        unsupported.append("requires online Megatron training")
    if asynchronous and (
        getattr(args, "pipeline_model_parallel_size", 1) != 1
        or getattr(args, "virtual_pipeline_model_parallel_size", None) not in (None, 1)
    ):
        unsupported.append("asynchronous MM replay requires PP1 without interleaving")
    if getattr(args, "fully_async", False) and not getattr(args, "hybrid", False):
        if not getattr(args, "use_dynamic_batch_size", False):
            unsupported.append("fully-async MM replay requires dynamic token batching")
        inline_forward = getattr(args, "true_on_policy_mode", False) or (
            getattr(args, "rollout_batch_size", 0) * getattr(args, "n_samples_per_prompt", 1)
            == getattr(args, "global_batch_size", -1)
        )
        if not inline_forward:
            unsupported.append("fully-async MM replay requires inline actor log-probabilities")
        if getattr(args, "kl_coef", 0.0) != 0 or getattr(args, "use_kl_loss", False):
            unsupported.append("fully-async MM replay does not support a separate reference policy")
    if getattr(args, "context_parallel_size", 1) != 1 or getattr(args, "dynamic_context_parallel", False):
        unsupported.append("requires CP1 without dynamic CP")
    if not getattr(args, "freeze_vision_model", False) or not getattr(args, "freeze_vision_projection", False):
        unsupported.append("requires frozen vision model and projection/merger")
    if not getattr(args, "bf16", True):
        unsupported.append("requires BF16 visual embeddings")
    if getattr(args, "multimodal_keys", None) is None:
        unsupported.append("requires multimodal input mapping")
    from relax.algorithms import algorithm_needs_critic

    if algorithm_needs_critic(args):
        unsupported.append("critic-based algorithms do not yet consume MM replay")
    for field in (
        "partial_rollout",
        "enable_mtp_training",
        "sglang_speculative_algorithm",
        "sglang_enable_hierarchical_cache",
        "use_opd",
    ):
        if getattr(args, field, False):
            unsupported.append(field)
    if (
        getattr(args, "rollout_function_path", "relax.engine.rollout.sglang_rollout.generate_rollout")
        not in ("relax.engine.rollout.sglang_rollout.generate_rollout", "relax.agentic.rollout.generate_rollout")
        or getattr(args, "custom_generate_function_path", None) is not None
    ):
        unsupported.append("custom rollout/generation function")
    if getattr(args, "qkv_format", "thd") != "thd":
        unsupported.append("requires packed THD")
    if unsupported:
        raise ValueError("--use-rollout-mm-embedding-replay: " + ", ".join(unsupported))
