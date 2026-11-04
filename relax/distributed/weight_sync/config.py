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
        data = parse_json(Path(path).read_bytes())
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
            for name in ("chunk_bytes", "compression", "anchor_interval", "host_budget_bytes", "full_only")
        }
