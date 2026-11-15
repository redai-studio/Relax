# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared inference contracts, independent of Ray and GPU backends."""

from relax.engine.inference.discovery import InferenceDiscoveryClient, new_manager_epoch, role_snapshot_from_dict
from relax.engine.inference.routing import resolve_model, select_target
from relax.engine.inference.types import Role, RoleSnapshot


__all__ = [
    "InferenceDiscoveryClient",
    "new_manager_epoch",
    "role_snapshot_from_dict",
    "Role",
    "RoleSnapshot",
    "resolve_model",
    "select_target",
]
