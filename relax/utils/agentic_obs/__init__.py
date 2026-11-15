# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Agentic observability: creation inventories, stages, and training inputs.

Streams joined by run, prompt, Group, Session, export, and update identity:

* ``prompts/``, ``groups/``, ``sessions/`` — original inputs and creation records.
* ``trajectories/`` — the full exported lineage, written by the owning
  SessionShard at export time, unconditionally.
* ``stages/`` — Session execution and export consumption events, from their owners.
* ``train/`` — actual update inputs and response masks, written by the actor.

Trajectories are written shard-side rather than shipped to the driver because
their payload is the full message history, including inlined base64 media, and
``TrainingFieldArtifact`` exists precisely to keep that weight off the Ray hop.

Everything is disabled unless ``--dump-agentic-obs`` names a directory.
"""

from relax.utils.agentic_obs.config import (
    AgenticObsConfig,
    agentic_obs_enabled,
    build_manifest,
    resolve_agentic_obs_config,
)
from relax.utils.agentic_obs.merge import merge_train_updates
from relax.utils.agentic_obs.records import (
    SCHEMA_VERSION,
    UNASSIGNED_SAMPLE_EXPORT_ID,
    ExportStageRecord,
    ExportTrajectoryRecord,
    GroupRecord,
    LineageNode,
    PromptRecord,
    SessionRecord,
    TrainRecord,
    TrainScalarRecord,
    pack_sample_export_id,
    strip_media,
    unpack_sample_export_id,
)
from relax.utils.agentic_obs.writers import AsyncJsonlAppender, JsonlAppender, write_media


__all__ = [
    "SCHEMA_VERSION",
    "UNASSIGNED_SAMPLE_EXPORT_ID",
    "AgenticObsConfig",
    "AsyncJsonlAppender",
    "ExportStageRecord",
    "ExportTrajectoryRecord",
    "GroupRecord",
    "JsonlAppender",
    "LineageNode",
    "PromptRecord",
    "SessionRecord",
    "TrainScalarRecord",
    "TrainRecord",
    "agentic_obs_enabled",
    "build_manifest",
    "pack_sample_export_id",
    "merge_train_updates",
    "resolve_agentic_obs_config",
    "strip_media",
    "unpack_sample_export_id",
    "write_media",
]
