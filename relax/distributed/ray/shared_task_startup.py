# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Ordered startup for inference owners sharing a task placement group."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import ray

from relax.distributed.ray.inference_manager import create_static_inference_manager, rollback_static_inference_manager
from relax.engine.inference.phase_plans import PHASE_GENERATE
from relax.engine.inference.task_placement import StaticPlacement, plan_task_placement
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class SharedStaticOwner:
    args: Any
    placement: StaticPlacement
    options: dict[str, Any]


def start_shared_task(
    args: Any,
    rollout_binding: Any,
    static: Sequence[SharedStaticOwner],
    create_rollout_owner: Callable[[], Any],
    *,
    runtime_env: Any = None,
) -> tuple[Any, tuple[Any, ...], Any]:
    """Preflight, construct and sleep each owner before constructing the next.

    Call before training Actor construction, while no training weights reside
    on the borrowed bundles. Returned owners remain sleeping until the phase
    coordinator activates generation. The caller owns all returned handles.
    """
    plan_task_placement(args, rollout_binding, [item.placement for item in static])
    if args is not None and getattr(args, "_shared_startup_pending_owners", None):
        raise RuntimeError("Shared startup has unconfirmed owners; retry cleanup before restarting")
    owners = []
    if args is not None:
        args._shared_startup_pending_owners = owners
    coordinator = None
    try:
        rollout = create_rollout_owner()
        if rollout not in owners:
            owners.append(rollout)
        shared = bool(getattr(args, "colocate", True)) and not getattr(args, "hybrid", False)
        if shared or getattr(args, "offload_rollout", False):
            ray.get(rollout.offload.remote())
        phases: dict[str, tuple[Any, ...]] = {PHASE_GENERATE: (rollout,)}
        for item in static:
            placement = item.placement
            owner = create_static_inference_manager(
                item.args,
                placement.spec.role.value,
                placement.binding,
                runtime_env=runtime_env,
                task_manager=rollout,
                **item.options,
            )
            owners.append(owner)
            # Confirm sleep even for split models whose flags keep them resident.
            if shared:
                ray.get(owner.offload.remote())
            from relax.engine.inference.types import DeploymentMode

            phase = placement.spec.deployment.phase
            if getattr(placement.spec.deployment, "mode", None) is DeploymentMode.DECOUPLED:
                phase = "resident"
            phases[phase] = (*phases.get(phase, ()), owner)
        # No owner is resident. An explicit empty startup phase makes the first
        # generation switch perform onload rather than an idempotent no-op.
        phases["startup"] = ()
        # Ray ActorHandle getattr fabricates ActorMethod objects. The rollout
        # owner is the task itself, whereas role handles carry real metadata.
        from relax.distributed.ray.inference_handle import InferenceRoleHandle

        phase_keys = {
            phase: tuple(
                (owner.role, owner.model_id) if isinstance(owner, InferenceRoleHandle) else ("rollout", "default")
                for owner in phase_owners
            )
            for phase, phase_owners in phases.items()
        }
        ray.get(
            rollout.configure_phases.remote(
                phase_keys, "startup" if shared or getattr(args, "offload_rollout", False) else PHASE_GENERATE
            )
        )
        coordinator = rollout
        if args is not None:
            del args._shared_startup_pending_owners
        return rollout, tuple(owners[1:]), coordinator
    except BaseException:
        from relax.distributed.ray.inference_handle import InferenceRoleHandle

        for owner in tuple(reversed(owners)):
            if isinstance(owner, InferenceRoleHandle):
                owners.remove(owner)
                continue
            try:
                rollback_static_inference_manager(owner)
            except BaseException:
                logger.exception("Shared startup rollback retained an unconfirmed owner")
            else:
                owners.remove(owner)
        raise


def retry_shared_startup_cleanup(args: Any) -> bool:
    """Retry partial startup owners retained on the Controller's config.

    Remove each handle only after its cleanup is confirmed. A failed retry
    leaves that owner and all remaining owners available to later shutdown.
    """
    owners = getattr(args, "_shared_startup_pending_owners", None)
    if owners is None:
        return False
    from relax.distributed.ray.inference_handle import InferenceRoleHandle

    for owner in tuple(reversed(owners)):
        if isinstance(owner, InferenceRoleHandle):
            owners.remove(owner)
            continue
        rollback_static_inference_manager(owner)
        owners.remove(owner)
    del args._shared_startup_pending_owners
    return True


def prepare_shared_task(
    args: Any, binding: Any, *, runtime_env: Any = None, bindings: dict[str, Any] | None = None
) -> Any:
    """Prepare engine owners once, before any training service is deployed."""
    from copy import copy, deepcopy
    from uuid import uuid4

    from relax.core.node_group_affinity import with_control_plane_affinity
    from relax.distributed.ray.inference_manager import InferenceManager
    from relax.engine.inference.config_adapters import static_role_spec
    from relax.engine.inference.types import Role
    from relax.utils.opd.opd_utils import _parse_teacher_routes, is_managed_opd_teacher_enabled

    if getattr(args, "_shared_startup_pending_owners", None):
        raise RuntimeError("Task startup has unconfirmed cleanup owners")
    from relax.engine.inference.phase_plans import deferred_genrm_enabled, uses_managed_genrm_hook

    if deferred_genrm_enabled(args) and not uses_managed_genrm_hook(args):
        raise ValueError(
            "Shared deferred GenRM requires a scoring-only post-process hook. Remove local memory swapping "
            "from the hook and set managed_genrm_scoring=True on the hook callable."
        )
    args._inference_startup_token = uuid4().hex
    bindings = bindings or {}
    static = []
    keys = []
    offset = 0
    for index, (key, spec) in enumerate((getattr(args, "_genrm_instances_resolved", None) or {}).items()):
        instance = copy(args)
        for field in ("model_path", "num_gpus", "num_gpus_per_engine", "engine_config", "sampling_config"):
            setattr(instance, f"genrm_{field}", spec[field])
        options = {"model_id": key, "bundle_offset": offset, "port_window_index": index}
        role_spec = static_role_spec(Role.GENRM, instance, options)
        colocated = getattr(args, "colocate", False) and not getattr(args, "hybrid", False)
        actual_offset = offset
        if colocated and role_spec.deployment.phase == PHASE_GENERATE:
            actual_offset += args.rollout_num_gpus
        static_binding = binding if colocated else bindings.get("genrm", binding)
        static.append(SharedStaticOwner(instance, StaticPlacement(role_spec, static_binding, actual_offset), options))
        keys.append(("genrm", key))
        offset += spec["num_gpus"]
    if is_managed_opd_teacher_enabled(args):
        total = args.resource["teacher"][1]
        tp = getattr(args, "teacher_num_gpus_per_engine", None)
        routes = getattr(args, "opd_teacher_routes", None)
        instances = (
            _parse_teacher_routes(routes, total, tp)
            if routes
            else {"default": {"checkpoint_path": args.teacher_hf_checkpoint, "num_gpus": total}}
        )
        offset = 0
        for key, spec in instances.items():
            instance = copy(args)
            instance.teacher_hf_checkpoint = spec["checkpoint_path"]
            size = tp or spec["num_gpus"]
            options = {
                "num_replicas": spec["num_gpus"] // size,
                "gpus_per_replica": size,
                "bundle_offset": offset,
                "model_id": key,
            }
            role_spec = static_role_spec(Role.TEACHER, instance, options)
            colocated = getattr(args, "colocate", False) and not getattr(args, "hybrid", False)
            start = args.rollout_num_gpus if colocated and role_spec.deployment.phase == PHASE_GENERATE else 0
            static_binding = binding if colocated else bindings.get("teacher")
            if static_binding is None:
                raise ValueError("Task startup requires the Teacher service placement binding")
            static.append(
                SharedStaticOwner(instance, StaticPlacement(role_spec, static_binding, start + offset), options)
            )
            keys.append(("teacher", key))
            offset += spec["num_gpus"]

    def create_rollout() -> Any:
        owner = InferenceManager.options(
            **with_control_plane_affinity(args, {"num_cpus": 1, "num_gpus": 0, "runtime_env": runtime_env})
        ).remote(startup_token=args._inference_startup_token)
        # Register before start(): its rollback can fail before the factory
        # returns a usable owner to start_shared_task.
        args._shared_startup_pending_owners.append(owner)
        rollout_args = copy(args)
        del rollout_args._shared_startup_pending_owners
        ray.get(owner.start.remote(deepcopy(rollout_args), binding, startup_token=args._inference_startup_token))
        return owner

    rollout, owners, coordinator = start_shared_task(args, binding, static, create_rollout, runtime_env=runtime_env)
    args._rollout_engine_manager = rollout
    args._inference_phase_coordinator = coordinator
    args._genrm_managers = {key: owner for (role, key), owner in zip(keys, owners) if role == "genrm"}
    teachers = {key: owner for (role, key), owner in zip(keys, owners) if role == "teacher"}
    args._teacher_inference_managers = teachers
    if teachers:
        routes = {}
        for key, owner in teachers.items():
            snapshot = ray.get(owner.snapshot.remote("teacher"))
            routes[key] = [replica.base_url + "/generate" for model in snapshot.models for replica in model.replicas]
        if getattr(args, "opd_teacher_routes", None):
            args.opd_teacher_routes_map = routes
        else:
            args._teacher_inference_manager_handle = teachers["default"]
            args.opd_teacher_url = routes["default"][0]
            args.opd_teacher_urls = routes["default"]
    return list(teachers.values()) if len(teachers) > 1 else next(iter(teachers.values()), None)
