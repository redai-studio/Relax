# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Serializable compatibility surface for the Rollout manager's consumers."""

from dataclasses import dataclass
from typing import Any


WORKLOAD_METHODS = frozenset(
    {
        "generate",
        "run_predict",
        "generate_predict",
        "eval",
        "drain_eval_judges",
        "save",
        "load",
        "get_num_rollout_per_epoch",
        "get_dynamic_global_batch_size",
        "reload_function_by_name",
        "reload_all_functions",
        "reload_module",
        "get_loaded_modules",
        "get_loaded_functions_info",
        "set_train_parallel_config",
    }
)


@dataclass(frozen=True)
class InferenceRoleHandle:
    """Serializable routing only; every operation executes in the task
    actor."""

    task: Any
    role: str
    model_id: str

    def __getattr__(self, method: str) -> Any:
        if method.startswith("_"):
            raise AttributeError(method)
        return _RoleCall(self, method)


@dataclass(frozen=True)
class _RoleCall:
    handle: InferenceRoleHandle
    method: str

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        return self.handle.task.dispatch.remote(
            self.handle.role,
            self.handle.model_id,
            self.method,
            *args,
            **kwargs,
        )


@dataclass(frozen=True)
class ManagedRolloutHandle:
    """Keep Actor/Controller interfaces while separating workload and GPU
    ownership."""

    inference_manager: Any
    worker: Any
    phase_coordinator: Any = None

    def __getattr__(self, method: str) -> Any:
        if method.startswith("_"):
            raise AttributeError(method)
        if method in WORKLOAD_METHODS:
            return getattr(self.worker, method)
        return _PoolCall(self, method)


@dataclass(frozen=True)
class _PoolCall:
    handle: ManagedRolloutHandle
    method: str

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        if self.method == "offload" and self.handle.phase_coordinator is not None:
            return self.handle.worker.offload.remote()
        if self.method == "get_status" and self.handle.phase_coordinator is not None:
            return self.handle.worker.get_status.remote()
        if self.method == "dispose":
            from relax.distributed.ray.inference_manager import dispose_managed_rollout

            return dispose_managed_rollout.remote(self.handle.inference_manager, self.handle.worker)
        return self.handle.inference_manager.pool_call.remote(self.method, *args, **kwargs)


def use_managed_rollout(args: Any) -> bool:
    """Every inference workload uses the task control plane."""
    from relax.engine.sft.runtime import is_offline_mode

    return not is_offline_mode(args) or getattr(args, "sft_predict_interval", None) is not None


def use_managed_auxiliary(args: Any) -> bool:
    return use_managed_rollout(args)


def shared_task_enabled(args: Any) -> bool:
    """Whether Controller prepares a task inference owner before deployment."""
    from relax.engine.inference.phase_plans import (
        deferred_genrm_enabled,
        deferred_opd_enabled,
        uses_managed_genrm_hook,
    )
    from relax.engine.sft.runtime import is_offline_mode
    from relax.utils.opd.opd_utils import is_managed_opd_teacher_enabled

    managed = use_managed_rollout(args) and not getattr(args, "debug_train_only", False)
    if not managed:
        return False
    shared_teacher = deferred_opd_enabled(args)
    if shared_teacher and deferred_genrm_enabled(args) and not uses_managed_genrm_hook(args):
        raise ValueError(
            "A shared OPD Teacher with deferred GenRM requires a scoring-only post-process hook "
            "with managed_genrm_scoring=True."
        )
    if (
        getattr(args, "colocate", False)
        and not getattr(args, "hybrid", False)
        and not is_offline_mode(args)
        and getattr(args, "offload_rollout", None) is False
        and not getattr(args, "debug_rollout_only", False)
        and (bool(getattr(args, "_genrm_instances_resolved", None)) or is_managed_opd_teacher_enabled(args))
    ):
        raise ValueError("Training with inference roles on training GPU bundles requires --offload-rollout.")
    return managed
