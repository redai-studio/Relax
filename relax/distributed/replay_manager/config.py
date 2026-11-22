# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One delivery configuration; references and compression are separate
choices."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Literal, Sequence

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

    def fields(self, fields: Sequence[str], *, kinds: Iterable[str] = ()) -> list[str]:
        replay = [FIELDS[kind] for kind in sorted(kinds)]
        result = list(dict.fromkeys([*fields, *replay]))
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
        for kind, flag in (
            ("moe", "use_rollout_routing_replay"),
            ("mm_embedding", "use_rollout_mm_embedding_replay"),
        )
        if getattr(args, flag, False)
    )


def _supports_reference_training(args: Any, kinds: frozenset[str]) -> bool:
    if getattr(args, "train_backend", None) != "megatron":
        return False
    if getattr(args, "fully_async", False) or getattr(args, "hybrid", False):
        return kinds.issubset({"moe", "mm_embedding"})
    return getattr(args, "colocate", False)


def _r3_unsupported_layout(args: Any) -> str | None:
    if getattr(args, "qkv_format", "thd") != "thd":
        return "QKV layouts other than packed THD"
    if getattr(args, "dynamic_context_parallel", False):
        return "dynamic context parallelism"
    if getattr(args, "overlap_moe_expert_parallel_comm", False):
        return "combined 1F1B MoE overlap"
    return None


def default_reference_r3(args: Any) -> bool:
    """Select reference delivery for online R3 when the user leaves it
    unset."""
    from relax.engine.sft.runtime import is_offline_mode

    return (
        getattr(args, "use_rollout_routing_replay", False)
        and not os.environ.get("RELAX_REPLAY_REFERENCES", "").strip()
        and _supports_reference_training(args, rollout_kinds(args))
        and _r3_unsupported_layout(args) is None
        and not is_offline_mode(args)
        and not getattr(args, "debug_train_only", False)
        and not getattr(args, "load_forge_rollout_data", None)
        and getattr(args, "rollout_function_path", "relax.engine.rollout.sglang_rollout.generate_rollout")
        in {"relax.engine.rollout.sglang_rollout.generate_rollout", "relax.agentic.rollout.generate_rollout"}
        and not getattr(args, "custom_generate_function_path", None)
        and not getattr(args, "custom_convert_samples_to_train_data_path", None)
        and (getattr(args, "rollout_external", False) or getattr(args, "sglang_pp_size", 1) == 1)
    )


def validate_reference_training(args: Any) -> None:
    """Reject deployment paths whose reference consumers are not wired."""
    kinds = rollout_kinds(args)
    if not references_enabled():
        if "moe" in kinds and getattr(args, "fully_async", False) and not getattr(args, "hybrid", False):
            if (layout := _r3_unsupported_layout(args)) is not None:
                raise ValueError(f"Fully async R3 does not support {layout}")
        return
    if not kinds:
        return
    if getattr(args, "load_debug_rollout_data", None) is not None:
        raise ValueError(
            "--load-debug-rollout-data cannot restore live replay references. "
            "For a dump containing full routes, set RELAX_REPLAY_REFERENCES=0 "
            "and disable --use-rollout-mm-embedding-replay."
        )
    if not _supports_reference_training(args, kinds):
        raise ValueError(
            "Reference replay requires colocated Megatron training, or hybrid/fully-async Megatron for R3 and frozen visual embeddings"
        )
    if "moe" in kinds and (layout := _r3_unsupported_layout(args)) is not None:
        raise ValueError(f"Reference R3 does not support {layout}")


def replay_fields(fields: list[str], *, kinds: Iterable[str] = ()) -> list[str]:
    return ReplayConfig.from_env().fields(fields, kinds=kinds)


if TYPE_CHECKING:
    from .storage import TQReplayStorage


def get_store(config: ReplayConfig) -> "TQReplayStorage":
    from .storage import TQReplayStorage

    if config.delivery != "reference":
        raise RuntimeError("Full replay delivery does not use a body store")
    return TQReplayStorage(config)
