# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Resolve the agentic observability output layout from parsed args.

Disabled unless ``--dump-agentic-obs`` names a directory, so every producer
starts with one ``None`` check and does no further work. The path doubles as
the switch, matching ``--dump-details``.

``run_id`` is stamped once during argument post-processing and travels inside
the args Namespace to every Ray actor, so the SessionShards, the rollout driver
and the actors all agree on one directory without coordinating.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class AgenticObsConfig:
    """Resolved output roots for one run."""

    root: Path
    run_id: str
    dump_media: bool

    @property
    def trajectories_dir(self) -> Path:
        return self.root / "trajectories"

    @property
    def stages_dir(self) -> Path:
        return self.root / "stages"

    @property
    def train_debug_dir(self) -> Path:
        return self.root / "train"

    @property
    def media_dir(self) -> Path:
        return self.root / "media"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def trajectory_path(self, shard_ordinal: int) -> Path:
        """One file per shard: a single writer, no concurrent appends.

        Not keyed by ``rollout_id`` on purpose — that identifier means three
        different things in this repo (driver step counter, per-node generation
        stamp, fully-async backfill index), and one step can write into two
        TransferQueue partitions.
        """

        return self.trajectories_dir / f"shard_{shard_ordinal:02d}.jsonl"

    def inventory_path(self, stream: str, producer_id: str) -> Path:
        return self.root / stream / f"{producer_id}.jsonl"

    def stage_path(self, rollout_id: int, *, rollout_mode: str = "train") -> Path:
        """One file per driver step.

        ``partition_id`` stays a record field.
        """

        prefix = "eval" if rollout_mode == "eval" else "rollout"
        return self.stages_dir / f"{prefix}_{rollout_id:08d}.jsonl"

    def train_scalar_path(self, rollout_id: int, dp_rank: int) -> Path:
        """One file per data-parallel rank.

        Named ``dp_`` rather than ``rank_`` because tensor- and pipeline-
        parallel peers are filtered out before writing; only the DP dimension
        produces complementary shards.
        """

        return self.train_debug_dir / f"step_{rollout_id:08d}_dp_{dp_rank:03d}.jsonl"


def agentic_obs_enabled(args: Optional[Namespace]) -> bool:
    """Report whether observability is on, without building a config.

    The producer of the ``sample_export_ids`` TransferQueue column and the
    consumer-side field whitelist must agree exactly: a column requested but
    never written makes ``async_get_meta`` wait forever. Both sides call this,
    so they cannot drift.
    """

    return bool(getattr(args, "dump_agentic_obs", None)) if args is not None else False


def train_generation_args(args: Namespace) -> Namespace:
    """Give all owners in one actor-group initialization the same observer ID.

    Copy only the Namespace; training settings and the caller's args stay
    intact. This does not communicate with workers or touch distributed
    initialization.
    """
    if not agentic_obs_enabled(args):
        return args
    import copy
    import uuid

    from relax.utils.logging_utils import get_logger

    try:
        observed = copy.copy(args)
        observed.agentic_obs_train_generation_id = uuid.uuid4().hex
        return observed
    except Exception as error:
        get_logger(__name__).warning("agentic_obs: training generation unavailable: %s", error)
        return args


def resolve_agentic_obs_config(args: Optional[Namespace]) -> Optional[AgenticObsConfig]:
    """Build the config, or return ``None`` when observability is off."""

    if not agentic_obs_enabled(args):
        return None
    root_dir = getattr(args, "dump_agentic_obs", None)
    run_id = getattr(args, "agentic_obs_run_id", None)
    if not run_id:
        # Derived in slime_validate_args, never parsed, so a missing value means
        # the caller built a Namespace by hand rather than that the operator
        # forgot a flag. Fall back rather than crash a training run.
        run_id = "run_unknown"
    return AgenticObsConfig(
        root=Path(str(root_dir)) / str(run_id),
        run_id=str(run_id),
        dump_media=bool(getattr(args, "dump_agentic_obs_media", False)),
    )


def build_manifest(args: Namespace, config: AgenticObsConfig, *, cp_size: int | None = None) -> dict[str, Any]:
    """Metadata a viewer needs before it can interpret the records.

    ``cp_size`` matters most: it is why ``TrainScalarRecord.advantage_mean``
    may be ``None``, and a viewer that cannot tell "context-parallel fail-
    closed" from "never computed" will misreport it.
    """

    from relax.utils.agentic_obs.records import SCHEMA_VERSION

    return {
        "schema_version": SCHEMA_VERSION,
        "record_format": "agentic-observability-full-v3",
        "run_id": config.run_id,
        "streams": {
            name: f"{name}/*.jsonl" for name in ("prompts", "groups", "sessions", "trajectories", "stages", "train")
        },
        "tokens": "deferred; no per-token debug stream in this contract",
        "prompt_identity": "source file and physical row before preprocessing plus content digest; version may be null",
        "custom_prompt_identity": "content digest unless Sample.agentic_prompt_source supplies source provenance",
        "inventory_sharding": "driver producer files; deduplicate prompts by (run_id, prompt_id)",
        "sampling_config": "session defaults after overrides; llm_calls carry request-specific overrides",
        "stage_boundaries": "start + outcome, paired by stage_id; event_seq is scoped to the owner (startup: stage_id)",
        "startup_stage": "stages/startup.jsonl; scope=run; completed.run_config contains the resolved rollout plan",
        "startup_steps": "rollout iterations, not optimizer updates; epoch equivalents include restored steps",
        "run_lifecycle": "stages/run.jsonl; scope=run, stage=execution; first terminal outcome ends the Run",
        "run_outcome": "occurred_at is the end time; operation is the failing stage; reason retains the error",
        "run_fallback": "missing Run outcome is unknown; the platform must fall back to authoritative Trial status",
        "session_timeline": "only start and final rollout; final contains llm_calls and llm_calls_complete",
        "selection_stages": "filter.operation distinguishes dynamic_filter, custom_advantage and disposal decisions",
        "session_counts": "requests entering the Session; attempts entering backend.generate; known generated deltas",
        "session_active_time": "managed process active-time budget, excluding generation gates",
        "session_logs": "logs/*.log; bounded final 1 MiB of combined stdout/stderr; queued reference; missing files mean unknown persistence",
        "train_scope": "megatron_policy_actor",
        "train_sharding": "join DP facts by (run_id, train_generation_id, update_id); inputs must match local stage_id",
        "optimizer_step": "successful updates within the actor-group generation; null on missing or conflicting sequence history",
        "missing_records": "unknown until all producers have been collected; dumps are best-effort",
        "loss_mask_space": "response; full mask omitted; scalars computed before loss-local filtering",
        "static_cp_scalars": "null",
        "cp_size": cp_size if cp_size is not None else getattr(args, "context_parallel_size", None),
        "dynamic_context_parallel": bool(getattr(args, "dynamic_context_parallel", False)),
        "dump_media": config.dump_media,
        "n_samples_per_prompt": getattr(args, "n_samples_per_prompt", None),
        "rollout_batch_size": getattr(args, "rollout_batch_size", None),
        "advantage_estimator": getattr(args, "advantage_estimator", None),
        "agentic_custom_advantage_path": getattr(args, "agentic_custom_advantage_path", None),
        "reward_key": getattr(args, "reward_key", None),
    }


def write_manifest(args: Namespace, config: AgenticObsConfig) -> None:
    """Publish a complete manifest atomically, including runs without
    training."""
    from relax.utils.agentic_obs.writers import atomic_write_json, snapshot_size, submit_background

    try:
        payload = build_manifest(args, config)
        submit_background(atomic_write_json, config.manifest_path, payload, size_bytes=snapshot_size(payload))
    except Exception as error:
        from relax.utils.logging_utils import get_logger

        get_logger(__name__).warning("agentic_obs: manifest snapshot failed: %s", error)


__all__ = ["AgenticObsConfig", "agentic_obs_enabled", "build_manifest", "resolve_agentic_obs_config", "write_manifest"]
