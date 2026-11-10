# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One delivery configuration; references and compression are separate
choices."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Sequence

from relax.utils.env import Envs

from .ref_data import FIELDS, REFERENCE_FIELD


@dataclass(frozen=True)
class ReplayConfig:
    """Transport and bounded storage policy; independent of model
    parallelism."""

    delivery: Literal["full", "reference"] = "full"
    compression: Literal["none"] = "none"
    run_id: str = ""
    capacity_bytes: int = 16 << 30
    chunk_rows: int = 1024
    publication_bytes: int = 64 << 20
    materialization_bytes: int = 256 << 20

    def __post_init__(self) -> None:
        if self.delivery not in {"full", "reference"}:
            raise ValueError(f"Unsupported replay delivery: {self.delivery}")
        if self.compression != "none":
            raise ValueError("Replay compression is not implemented; use 'none'")
        if min(self.capacity_bytes, self.chunk_rows, self.publication_bytes) <= 0 or self.materialization_bytes < 0:
            raise ValueError(
                "Replay capacity/chunk/publication limits must be positive; materialization budget nonnegative"
            )

    @classmethod
    def from_env(cls) -> "ReplayConfig":
        return cls(
            delivery="reference" if references_enabled() else "full",
            run_id=Envs.RELAX_REPLAY_RUN_ID,
            capacity_bytes=Envs.RELAX_REPLAY_CAPACITY_MB << 20,
        )

    def fields(self, fields: Sequence[str]) -> list[str]:
        result = list(dict.fromkeys(fields))
        if self.delivery == "reference":
            has_replay = bool(set(FIELDS.values()).intersection(result))
            result = [field for field in result if field not in FIELDS.values()]
            if has_replay and REFERENCE_FIELD not in result:
                result.append(REFERENCE_FIELD)
        return result


def references_enabled() -> bool:
    return Envs.RELAX_REPLAY_REFERENCES


def reference_environment() -> dict[str, str]:
    names = ("RELAX_REPLAY_REFERENCES", "RELAX_REPLAY_RUN_ID", "RELAX_REPLAY_CAPACITY_MB")
    return {name: os.environ[name] for name in names if name in os.environ}


def rollout_kinds(args: Any) -> frozenset[str]:
    return frozenset(
        kind
        for kind, flag in (("moe", "use_rollout_routing_replay"), ("indexer", "use_rollout_indexer_replay"))
        if getattr(args, flag, False)
    )


def replay_fields(fields: list[str]) -> list[str]:
    return ReplayConfig.from_env().fields(fields)


if TYPE_CHECKING:
    from .storage import TQReplayStorage


def get_store(config: ReplayConfig) -> "TQReplayStorage":
    from .storage import TQReplayStorage

    if config.delivery != "reference":
        raise RuntimeError("Full replay delivery does not use a body store")
    return TQReplayStorage(config)
