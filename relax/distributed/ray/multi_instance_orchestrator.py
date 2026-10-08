# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Launch N named managers (GenRM judges, OPD teachers, ...) sharing one
placement group.

This is the domain-agnostic core of what MOPD's
``_start_managed_multi_teacher`` did for OPD teachers: launch one manager actor
per keyed instance within a shared placement group. Manager constructor
signatures differ (``TeacherManager`` takes ``num_replicas``/
``gpus_per_replica``/``bundle_offset`` directly; ``GenRMManager`` derives them
from its args namespace), so the actual ``.remote(...)`` call is left to the
caller-supplied ``spawn_manager`` hook. Where each instance sits in the
placement group is decided by ``placement_planner.plan_placement``, which the
hook reads -- this function only owns the per-instance argument fan-out and the
offload-on-start behavior shared by every manager type.
"""

from __future__ import annotations

from typing import Any, Callable

import ray

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def start_multi_instance_managers(
    *,
    args: Any,
    instance_specs: dict[str, dict],
    build_manager_args: Callable[[Any, str, dict], Any],
    spawn_manager: Callable[[str, Any, dict], Any],
) -> dict[str, Any]:
    """Launch one manager per entry in ``instance_specs`` within whatever
    shared placement group ``spawn_manager`` closes over.

    Args:
        args: Base argument namespace.
        instance_specs: ``{key: spec}``. Each ``spec`` must contain
            ``num_gpus`` (int); any other keys are opaque to this function and
            forwarded verbatim to ``build_manager_args``/``spawn_manager``.
        build_manager_args: ``(args, key, spec) -> per_instance_args``, used
            to inject instance-specific overrides (e.g. a distinct model path)
            into a copy of ``args`` before it's passed to ``spawn_manager``.
        spawn_manager: ``(key, per_instance_args, spec) -> manager_handle``.
            Owns the manager class's actual constructor signature, its shared
            placement group, and the lookup of the instance's planned
            ``bundle_offset`` -- e.g. for ``TeacherManager`` this calls
            ``TeacherManager.options(...).remote(per_instance_args,
            num_replicas, gpus_per_replica, pg=shared_pg, shared_pg=True,
            bundle_offset=plan.claim("teacher", key).start)``.

    Returns:
        ``{key: manager_handle}`` in ``instance_specs`` iteration order.
    """
    managers: dict[str, Any] = {}
    for key, spec in instance_specs.items():
        per_instance_args = build_manager_args(args, key, spec)
        managers[key] = spawn_manager(key, per_instance_args, spec)
        logger.info(f"Launched instance '{key}': num_gpus={spec['num_gpus']}")

    if getattr(args, "offload_rollout", False):
        ray.get([m.offload.remote() for m in managers.values()])

    return managers
