# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Serializable compatibility surface for the Rollout manager's consumers."""

from dataclasses import dataclass
from typing import Any


WORKLOAD_METHODS = frozenset(
    {
        "generate",
        "eval",
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
class ManagedRolloutHandle:
    """Keep Actor/Controller interfaces while separating workload and GPU
    ownership."""

    inference_manager: Any
    worker: Any

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
        if self.method == "dispose":
            from relax.distributed.ray.inference_manager import dispose_managed_rollout

            return dispose_managed_rollout.remote(self.handle.inference_manager, self.handle.worker)
        return self.handle.inference_manager.pool_call.remote(self.method, *args, **kwargs)


def use_managed_rollout(args: Any) -> bool:
    """Choose a complete task path once; unsupported consumers keep the old
    path.

    PR 2 owns auxiliary-role migration; PR 3 owns shared scoring. Custom/local
    hooks, Agentic sessions and native engines still depend on the original
    actor identity and are retained as complete legacy tasks in this first PR.
    """
    from relax.engine.sft.runtime import is_offline_mode

    standard = "relax.engine.rollout.sglang_rollout.generate_rollout"
    return not (
        is_offline_mode(args)
        or bool(getattr(args, "_genrm_instances_resolved", None))
        or bool(getattr(args, "use_opd", False))
        or bool(getattr(args, "use_agentic_rollout", False))
        or bool(getattr(args, "rollout_engine_class_path", None))
        or bool(getattr(args, "custom_reward_post_process_path", None))
        or bool(getattr(args, "custom_convert_samples_to_train_data_path", None))
        or bool(getattr(args, "debug_train_only", False))
        or bool(getattr(args, "rollout_external", False))
        or bool(getattr(args, "custom_generate_function_path", None))
        or bool(getattr(args, "custom_dataset_class_path", None))
        or getattr(args, "rollout_function_path", standard) != standard
        or getattr(args, "eval_function_path", standard) not in (None, standard)
        or any(
            getattr(dataset, "custom_generate_function_path", None) for dataset in getattr(args, "eval_datasets", [])
        )
    )
