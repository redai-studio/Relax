# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Configuration and topology checks for opt-in TCP memory updates."""

from dataclasses import dataclass, fields
from pathlib import Path

from .format import WeightSyncError, integer, parse_json


def _validate_sync_execution(args, label: str, *, sharded: bool = False) -> None:
    unsupported = [
        name
        for name in (
            "fully_async",
            "hybrid",
            "colocate",
            "offload_train",
            "offload_rollout",
            "use_fault_tolerance",
            "keep_old_actor",
            "use_lora",
            "debug_train_only",
            "debug_rollout_only",
            "partial_rollout",
            "use_agentic_rollout",
            "rollout_external",
            "sglang_speculative_algorithm",
            "enable_mtp_training",
            "mtp_num_layers",
        )
        if getattr(args, name, False)
    ]
    if unsupported:
        raise WeightSyncError(f"{label} weight sync does not support: {', '.join(unsupported)}")
    if getattr(args, "lora_rank", 0) > 0 or getattr(args, "save_hf_dtype", "bf16") != "bf16":
        raise WeightSyncError(f"{label} weight sync requires full-parameter, nonquantized HF export")
    if getattr(args, "num_experts", None) and not sharded:
        raise WeightSyncError(f"{label} weight sync currently supports Dense models only")
    if not getattr(args, "sglang_disable_cuda_graph", False):
        raise WeightSyncError(f"{label} weight sync currently requires --sglang-disable-cuda-graph")
    if not sharded and any(
        getattr(args, name, 1) != 1 for name in ("sglang_pp_size", "sglang_dp_size", "rollout_num_gpus_per_engine")
    ):
        raise WeightSyncError(f"The {label.lower()} MVP currently requires single-GPU TP/PP/DP=1 inference engines")
    if getattr(args, "train_backend", "megatron") != "megatron":
        raise WeightSyncError(f"{label} weight sync requires the Megatron backend")


@dataclass(frozen=True)
class MemorySyncConfig:
    stream: str
    layout: str = "dense"
    bucket_bytes: int = 512 << 20
    gpu_budget_bytes: int = 16 << 30
    host_budget_bytes: int = 8 << 30
    sync_timeout_s: int = 120
    anchor_interval: int = 20
    full_only: bool = False
    receiver_gpu_budget_bytes: int = 12 << 30

    @classmethod
    def load(cls, path: str | Path) -> "MemorySyncConfig":
        data = parse_json(Path(path).read_bytes())
        if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
            raise WeightSyncError("Unknown or invalid memory weight sync settings")
        if not isinstance(data.get("stream"), str) or not data["stream"]:
            raise WeightSyncError("Memory weight sync requires a nonempty stream")
        config = cls(**data)
        for name in (
            "bucket_bytes",
            "gpu_budget_bytes",
            "host_budget_bytes",
            "sync_timeout_s",
            "anchor_interval",
            "receiver_gpu_budget_bytes",
        ):
            integer(getattr(config, name), 1)
        if type(config.full_only) is not bool:
            raise WeightSyncError("full_only must be a boolean")
        if config.layout not in ("dense", "qwen35_owner_local"):
            raise WeightSyncError("Unsupported memory shard layout")
        return config


def validate_memory_weight_sync_args(args) -> None:
    path = getattr(args, "memory_weight_sync_config", None)
    if path is None:
        return
    config = MemorySyncConfig.load(path)
    sharded = config.layout == "qwen35_owner_local"
    _validate_sync_execution(args, "Memory", sharded=sharded)
    if sharded:
        if (
            getattr(args, "megatron_to_hf_mode", None) != "bridge"
            or getattr(args, "num_experts", None) != 256
            or getattr(args, "num_layers", None) != 40
            or getattr(args, "hidden_size", None) != 2048
            or getattr(args, "expert_tensor_parallel_size", 1) != 1
            or getattr(args, "expert_model_parallel_size", 1) != 8
            or getattr(args, "tensor_model_parallel_size", 1) != 4
            or any(
                getattr(args, name, 1) != 1
                for name in (
                    "pipeline_model_parallel_size",
                    "context_parallel_size",
                    "sglang_pp_size",
                    "sglang_dp_size",
                )
            )
            or getattr(args, "rollout_num_gpus_per_engine", 1) != 2
            or getattr(args, "sglang_moe_runner_backend", None) != "triton"
            or not getattr(args, "sglang_disable_shared_experts_fusion", False)
        ):
            raise WeightSyncError("Owner-local memory sync requires the explicit Qwen3.5 TP4/EP8 to TP2 layout")
        return
    if any(
        getattr(args, name, 1) != 1
        for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size")
    ):
        raise WeightSyncError("Memory sync currently requires training TP/PP/CP=1")
