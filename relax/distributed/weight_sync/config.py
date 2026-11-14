# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Configuration and topology checks for opt-in TCP memory updates."""

from dataclasses import dataclass, fields, replace
from pathlib import Path

from .format import WeightSyncError, integer, parse_json


def _validate_sync_execution(args) -> None:
    """Current coordinator envelope; model/export capability is checked
    separately."""
    unavailable = [
        name
        for name in (
            "fully_async",
            "hybrid",
            "colocate",
            "offload_train",
            "offload_rollout",
            "use_fault_tolerance",
            "keep_old_actor",
            "rollout_external",
        )
        if getattr(args, name, False)
    ]
    if unavailable:
        raise WeightSyncError(f"Memory coordinator does not support these lifecycles yet: {', '.join(unavailable)}")
    unvalidated = [name for name in ("partial_rollout", "use_agentic_rollout") if getattr(args, name, False)]
    if unvalidated:
        raise WeightSyncError(
            f"Memory synchronization has not validated these request boundaries: {', '.join(unvalidated)}"
        )
    if getattr(args, "debug_train_only", False) or getattr(args, "debug_rollout_only", False):
        raise WeightSyncError("Debug-only execution does not perform training-to-rollout synchronization")
    if (
        getattr(args, "sglang_speculative_algorithm", None)
        or getattr(args, "enable_mtp_training", False)
        or getattr(args, "mtp_num_layers", 0)
    ):
        raise WeightSyncError("Memory synchronization has no draft/MTP-model update contract")
    if not getattr(args, "sglang_disable_cuda_graph", False):
        raise WeightSyncError(
            "Memory synchronization currently requires --sglang-disable-cuda-graph until replay validation"
        )
    if any(getattr(args, name, 1) != 1 for name in ("sglang_pp_size", "sglang_dp_size")):
        raise WeightSyncError("Memory synchronization has not integrated inference PP/DP routing")
    if getattr(args, "train_backend", "megatron") != "megatron":
        raise WeightSyncError("The memory source adapter is currently integrated with Megatron")


@dataclass(frozen=True)
class MemorySyncConfig:
    stream: str
    layout: str = "canonical_hf"
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
        if config.layout == "qwen35_owner_local":
            config = replace(config, layout="native_gated_moe")
        if config.layout not in ("dense", "canonical_hf", "native_gated_moe"):
            raise WeightSyncError("Unsupported memory application strategy")
        return config


def validate_memory_weight_sync_args(args) -> None:
    path = getattr(args, "memory_weight_sync_config", None)
    if path is None:
        return
    from relax.backends.megatron.weight_update.memory_export import validate_memory_export_args

    config = MemorySyncConfig.load(path)
    _validate_sync_execution(args)
    validate_memory_export_args(args, config.layout)
