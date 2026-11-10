# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One explicit configuration object for the independent publication path."""

from dataclasses import dataclass, fields
from pathlib import Path

from .format import MAX_BLOCK_BYTES, WeightSyncError, integer, parse_json


@dataclass(frozen=True)
class WeightSyncConfig:
    mode: str = "existing"
    stream: str = ""
    store_root: str = ""
    canonical_root: str = ""
    chunk_bytes: int = MAX_BLOCK_BYTES
    compression: bool = True
    anchor_interval: int = 20
    host_budget_bytes: int = 16 * 1024**3
    disk_budget_bytes: int = 128 * 1024**3
    sync_timeout_s: int = 600
    full_only: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "WeightSyncConfig":
        path = Path(path).resolve()
        data = parse_json(path.read_bytes())
        if isinstance(data, dict):
            for name in ("store_root", "canonical_root"):
                value = data.get(name)
                if isinstance(value, str) and value and not Path(value).is_absolute():
                    data[name] = str(path.parent / value)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "WeightSyncConfig":
        if not isinstance(data, dict):
            raise WeightSyncError("Weight sync configuration must be an object")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise WeightSyncError(f"Unknown weight sync settings: {sorted(unknown)}")
        config = cls(**data)
        if config.mode not in {"existing", "delta"}:
            raise WeightSyncError("mode must be existing or delta")
        if type(config.compression) is not bool or type(config.full_only) is not bool:
            raise WeightSyncError("compression and full_only must be booleans")
        integer(config.chunk_bytes, 8, MAX_BLOCK_BYTES)
        for name in ("anchor_interval", "host_budget_bytes", "disk_budget_bytes", "sync_timeout_s"):
            integer(getattr(config, name), 1)
        for name in ("stream", "store_root", "canonical_root"):
            if not isinstance(getattr(config, name), str):
                raise WeightSyncError(f"{name} must be a string")
            if config.mode == "delta" and not getattr(config, name):
                raise WeightSyncError(f"delta mode requires {name}")
        if config.mode == "delta" and Path(config.canonical_root).resolve().is_relative_to(
            Path(config.store_root).resolve()
        ):
            raise WeightSyncError(
                "canonical_root must be outside store_root; keep local recovery bytes off the delivery path"
            )
        return config

    def publisher_kwargs(self) -> dict:
        return {
            name: getattr(self, name)
            for name in (
                "chunk_bytes",
                "compression",
                "anchor_interval",
                "host_budget_bytes",
                "full_only",
                "sync_timeout_s",
            )
        }


def validate_disk_weight_sync_args(args) -> None:
    """Reject execution modes whose lifecycle is outside the disk MVP."""
    path = getattr(args, "disk_weight_sync_config", None)
    if path is None:
        return
    config = WeightSyncConfig.load(path)
    if config.mode != "delta":
        raise WeightSyncError("--disk-weight-sync-config requires mode=delta (full_only selects the FULL control)")
    _validate_sync_execution(args, "Disk")


def _validate_sync_execution(args, label: str) -> None:
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
    if getattr(args, "num_experts", None):
        raise WeightSyncError(f"{label} weight sync currently supports Dense models only")
    if not getattr(args, "sglang_disable_cuda_graph", False):
        raise WeightSyncError(f"{label} weight sync currently requires --sglang-disable-cuda-graph")
    if any(
        getattr(args, name, 1) != 1 for name in ("sglang_pp_size", "sglang_dp_size", "rollout_num_gpus_per_engine")
    ):
        raise WeightSyncError(f"The {label.lower()} MVP currently requires single-GPU TP/PP/DP=1 inference engines")
    if getattr(args, "train_backend", "megatron") != "megatron":
        raise WeightSyncError(f"{label} weight sync requires the Megatron backend")


@dataclass(frozen=True)
class MemorySyncConfig:
    stream: str
    bucket_bytes: int = 512 << 20
    gpu_budget_bytes: int = 16 << 30
    host_budget_bytes: int = 8 << 30
    sync_timeout_s: int = 120
    anchor_interval: int = 20
    full_only: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "MemorySyncConfig":
        data = parse_json(Path(path).read_bytes())
        if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
            raise WeightSyncError("Unknown or invalid memory weight sync settings")
        if not isinstance(data.get("stream"), str) or not data["stream"]:
            raise WeightSyncError("Memory weight sync requires a nonempty stream")
        config = cls(**data)
        for name in ("bucket_bytes", "gpu_budget_bytes", "host_budget_bytes", "sync_timeout_s", "anchor_interval"):
            integer(getattr(config, name), 1)
        if type(config.full_only) is not bool:
            raise WeightSyncError("full_only must be a boolean")
        return config


def validate_memory_weight_sync_args(args) -> None:
    path = getattr(args, "memory_weight_sync_config", None)
    if path is None:
        return
    if getattr(args, "disk_weight_sync_config", None):
        raise WeightSyncError("Select one weight sync configuration")
    MemorySyncConfig.load(path)
    _validate_sync_execution(args, "Memory")
    if any(
        getattr(args, name, 1) != 1
        for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size")
    ):
        raise WeightSyncError("Memory sync currently requires training TP/PP/CP=1")
