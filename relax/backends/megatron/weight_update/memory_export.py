# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Megatron export capabilities, separate from the memory wire protocol."""

from typing import Any

from relax.distributed.weight_sync.format import WeightSyncError


def validate_memory_export_args(args: Any, strategy: str) -> None:
    """Reject unimplemented conversion contracts, without model size gates."""
    mode = getattr(args, "megatron_to_hf_mode", "raw")
    if mode not in ("bridge", "raw"):
        raise WeightSyncError("Memory synchronization requires a Bridge or raw HF exporter")
    if getattr(args, "use_lora", False) or getattr(args, "lora_rank", 0) > 0:
        raise WeightSyncError("The memory exporter has not integrated LoRA merge/adapter lifecycles")
    if getattr(args, "virtual_pipeline_model_parallel_size", None) not in (None, 1):
        raise WeightSyncError("Memory export requires VPP1")
    if strategy == "native_gated_moe":
        if (
            mode != "bridge"
            or not getattr(args, "num_experts", None)
            or getattr(args, "expert_model_parallel_size", 1) <= 1
            or getattr(args, "expert_tensor_parallel_size", 1) != 1
            or getattr(args, "pipeline_model_parallel_size", 1) != 1
            or getattr(args, "context_parallel_size", 1) != 1
        ):
            raise WeightSyncError("Native gated-MoE export requires Bridge, EP>1, ETP1 and PP/CP1")
        return
    if strategy == "dense":
        if getattr(args, "num_experts", None):
            raise WeightSyncError("The direct-copy Dense adapter supports Dense models only")
        if getattr(args, "rollout_num_gpus_per_engine", 1) != 1:
            raise WeightSyncError("The direct-copy Dense adapter requires one inference worker per engine")

    dense_bridge = strategy == "canonical_hf" and mode == "bridge" and not getattr(args, "num_experts", None)
    # Only Dense Bridge has a callback capture path that gates conversion and
    # encoder failures at the same fixed parameter boundary on every TP/DP rank.
    if dense_bridge:
        if any(
            getattr(args, name, 1) != 1
            for name in ("pipeline_model_parallel_size", "context_parallel_size", "expert_model_parallel_size")
        ):
            raise WeightSyncError("Gated Dense Bridge memory export requires training PP/CP/EP=1")
        return

    # Raw and ordinary MoE conversion retain their existing TP1 scope.
    if any(
        getattr(args, name, 1) != 1
        for name in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
        )
    ):
        raise WeightSyncError("Canonical memory export currently requires training TP/PP/CP/EP=1")
