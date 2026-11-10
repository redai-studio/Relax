# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay entry point.

Backend adapters are loaded only when registered.
"""

from .config import ReplayConfig, reference_environment, references_enabled, replay_fields, rollout_kinds
from .consumer import ConsumerExecutor, ConsumerRequest
from .core import ReplayKind, ReplayManager, ReplayRegion, replay_data_manager
from .ref_data import ReplayRef, register_kind


__all__ = [
    "ReplayConfig",
    "ReplayKind",
    "ConsumerRequest",
    "ConsumerExecutor",
    "register_kind",
    "ReplayRegion",
    "ReplayRef",
    "ReplayManager",
    "replay_data_manager",
    "reference_environment",
    "references_enabled",
    "replay_fields",
    "rollout_kinds",
]
