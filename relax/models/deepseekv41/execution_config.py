# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Execution limits of the V4.1 text model, not HF architecture overrides."""

from argparse import Namespace
from typing import Any


def _validate_runtime_layout(provider: Any) -> None:
    if getattr(provider, "cuda_graph_impl", "none") not in (None, "none"):
        raise NotImplementedError("DeepSeek V4.1 training CUDA-graph capture/replay is not supported yet.")
    qkv_format = getattr(provider, "v41_qkv_format", "bshd")
    if qkv_format not in ("bshd", "thd"):
        raise ValueError("DeepSeek V4.1 execution requires --qkv-format bshd or thd.")
    if qkv_format == "thd" and getattr(provider, "v41_indexer_replay", False):
        raise ValueError("DeepSeek V4.1 THD does not support attention indexer replay.")
    if getattr(provider, "dynamic_context_parallel", False):
        raise ValueError("DeepSeek V4.1 currently supports fixed context-parallel groups only.")
    cp = getattr(provider, "context_parallel_size", 1)
    if cp < 1:
        raise ValueError("DeepSeek V4.1 context_parallel_size must be positive.")
    # Both layouts use Relax's contiguous allgather-CP ingestion. BSHD does
    # not use Core's native cp_partition_mode (whose contiguous path is THD).
    if cp > 1 and not getattr(provider, "v41_allgather_cp", False):
        raise ValueError("DeepSeek V4.1 CP>1 requires --allgather-cp for contiguous input.")

    pp = getattr(provider, "pipeline_model_parallel_size", 1)
    if pp < 1:
        raise ValueError("DeepSeek V4.1 pipeline_model_parallel_size must be positive.")
    if getattr(provider, "virtual_pipeline_model_parallel_size", None) not in (None, 1):
        raise ValueError("DeepSeek V4.1 does not support virtual pipeline parallelism.")
    if getattr(provider, "pipeline_model_parallel_layout", None) is not None:
        raise ValueError("DeepSeek V4.1 does not support custom pipeline_model_parallel_layout.")
    if pp > 1:
        if provider.num_layers % pp:
            raise ValueError("DeepSeek V4.1 num_layers must be divisible by PP size for uniform pipeline stages.")
        if any(
            getattr(provider, field, None) is not None
            for field in ("num_layers_in_first_pipeline_stage", "num_layers_in_last_pipeline_stage")
        ):
            raise ValueError(
                "DeepSeek V4.1 supports uniformly divided PP stages, not first/last-stage layer overrides."
            )
        if not getattr(provider, "variable_seq_lengths", False):
            raise ValueError("DeepSeek V4.1 PP requires variable_seq_lengths=True to transmit its cross-layer carry.")
        if getattr(provider, "overlap_moe_expert_parallel_comm", False):
            raise ValueError(
                "DeepSeek V4.1 PP does not support overlap_moe_expert_parallel_comm: "
                "that schedule requires VPP, which this model does not support."
            )

    if getattr(provider, "recompute_granularity", None) == "selective":
        unsupported = {"core_attn", "mhc"}.intersection(getattr(provider, "recompute_modules", None) or ())
        if unsupported:
            raise ValueError(
                f"DeepSeek V4.1 selective recompute does not support {sorted(unsupported)}; "
                "use full recompute or selective moe/mlp recompute."
            )
    if qkv_format == "thd" and getattr(provider, "recompute_granularity", None) == "full":
        if (
            getattr(provider, "recompute_method", None) != "uniform"
            or getattr(provider, "recompute_num_layers", 0) != 1
        ):
            raise ValueError("DeepSeek V4.1 THD full recompute requires uniform recompute with one layer per region.")


def configure_execution_runtime(provider: Any, args: Namespace) -> None:
    """Carry the existing runtime options into the model without adding CLI
    arguments.

    Kernel dimensions and the final dtype are checked at model construction,
    after Relax's generic provider overrides. Configuration-only Bridge
    conversion remains usable for differently sized checkpoint fixtures.
    """
    for source, target, default in (
        ("hf_checkpoint", "v41_hf_checkpoint", None),
        ("qkv_format", "v41_qkv_format", "bshd"),
        ("allgather_cp", "v41_allgather_cp", False),
        ("use_rollout_indexer_replay", "v41_indexer_replay", False),
    ):
        setattr(provider, target, getattr(args, source, getattr(provider, target, default)))
    # Generic Bridge mode currently clears this flag before model construction;
    # this attention implementation caches fixed CP groups and cannot follow it.
    if getattr(args, "dynamic_context_parallel", False):
        raise ValueError("DeepSeek V4.1 currently supports fixed context-parallel groups only.")
    if hasattr(args, "overlap_moe_expert_parallel_comm"):
        provider.overlap_moe_expert_parallel_comm = args.overlap_moe_expert_parallel_comm
    provider.variable_seq_lengths = True
    _validate_runtime_layout(provider)


def validate_execution_config(provider: Any) -> None:
    """Reject unsupported execution paths before allocating model weights or
    launching kernels."""
    _validate_runtime_layout(provider)
    dtype = str(provider.params_dtype).removeprefix("torch.")
    if dtype != "bfloat16" or getattr(provider, "fp8", None) or getattr(provider, "fp4", None):
        raise ValueError("DeepSeek V4.1 requires BF16 training; FP16/FP32/FP8/FP4 compute is unsupported.")
    for field, expected in (("kv_lora_rank", 512), ("qk_pos_emb_head_dim", 64), ("csa_window_size", 128)):
        actual = getattr(provider, field)
        if actual != expected:
            raise ValueError(f"DeepSeek V4.1 kernels require {field}={expected}, got {actual}.")
    for field in ("hidden_dropout", "attention_dropout"):
        if getattr(provider, field, 0.0) != 0.0:
            raise ValueError(f"DeepSeek V4.1 execution currently requires {field}=0.0.")
    tp = provider.tensor_model_parallel_size
    if tp < 1:
        raise ValueError("DeepSeek V4.1 tensor_model_parallel_size must be positive.")
    for field in ("num_attention_heads", "o_groups"):
        count = getattr(provider, field)
        if count < 1 or count % tp:
            raise ValueError(f"DeepSeek V4.1 {field}={count} must be positive and divisible by TP size {tp}.")
    if provider.num_attention_heads % provider.o_groups:
        raise ValueError("DeepSeek V4.1 num_attention_heads must be divisible by o_groups.")
