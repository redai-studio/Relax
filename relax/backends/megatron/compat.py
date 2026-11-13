# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import inspect
import math
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from threading import RLock
from typing import Any


_BUGGY_NATIVE_FP32_LOOKUPS = ("self.param_to_fp32_param[param]", "self.param_to_fp32_param.get(param)")
_NATIVE_FP32_LOAD_PATCHED = "_relax_native_fp32_load_patched"
_HDO_STEP_LOAD_LOCK = RLock()


def _patch_hdo_group_steps(optimizer_cls: type) -> bool:
    """Keep group-owned Adam counters when HDO synchronizes tensor state."""
    import torch

    original = optimizer_cls._sync_sub_optimizers_state_to_hdo
    if getattr(original, "_relax_group_steps", False):
        return False

    @wraps(original)
    def sync(self: Any) -> None:
        original(self)
        owners = {param: i for i, group in enumerate(self.param_groups) for param in group["params"]}
        steps: dict[int, int] = {}
        for child in self.sub_optimizers:
            if isinstance(child, (torch.optim.Adam, torch.optim.AdamW)):
                continue
            for group in child.param_groups:
                # Torch Adam keeps counters in per-parameter state, already
                # handled by HDO; TE/Apex Adam keeps them on the group.
                if "step" not in group or not group["params"]:
                    continue
                owner = owners[self.inner_param_to_orig_param[group["params"][0]]]
                step = group["step"]
                if not isinstance(step, int) or step < 0:
                    raise RuntimeError("HDO group-owned Adam step must be a non-negative Python integer")
                if owner in steps and steps[owner] != step:
                    raise RuntimeError("HDO sub-optimizers disagree on their Adam group step")
                steps[owner] = step
        for owner, step in steps.items():
            self.param_groups[owner]["step"] = step

    sync._relax_group_steps = True
    optimizer_cls._sync_sub_optimizers_state_to_hdo = sync
    return True


def patch_hdo_group_steps() -> bool:
    """Backport HDO group-counter synchronization without tensor copies."""
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    return _patch_hdo_group_steps(HybridDeviceOptimizer)


def _patch_hybrid_optimizer_class(optimizer_cls: type) -> bool:
    """Restore all HDO inner parameters when loading older-image
    checkpoints."""
    update_method = getattr(optimizer_cls, "_update_fp32_params_by_new_state", None)
    if update_method is None or getattr(update_method, _NATIVE_FP32_LOAD_PATCHED, False):
        return False

    try:
        source = inspect.getsource(update_method)
    except (OSError, TypeError):
        return False
    if not any(lookup in source for lookup in _BUGGY_NATIVE_FP32_LOOKUPS):
        return False

    def _update_fp32_params_by_new_state(self) -> None:
        if not self.param_update_in_fp32:
            return
        for param, state in self.state.items():
            # Native FP32 parameters have no FP32 shadow, but CPU offload still
            # gives them a separate inner parameter owned by the sub-optimizer.
            self.param_to_inner_param[param].data.copy_(state["master_param"])

    setattr(_update_fp32_params_by_new_state, _NATIVE_FP32_LOAD_PATCHED, True)
    optimizer_cls._update_fp32_params_by_new_state = _update_fp32_params_by_new_state
    return True


def patch_hybrid_optimizer_native_fp32_checkpoint_load() -> bool:
    """Backport native-FP32 CPU-offload checkpoint restoration to older
    images."""
    try:
        from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
    except ImportError:
        return False
    return _patch_hybrid_optimizer_class(HybridDeviceOptimizer)


@contextmanager
def preserve_hdo_dp_reshardable_steps_on_load() -> Iterator[None]:
    """Preserve saved CPU Adam steps during HDO DCP loads.

    MCore's local parameter-state template can overwrite the saved step with
    dummy_step=1. Repair only the tested precision-aware Torch AdamW path;
    restore the original loader, including any QAT hooks, when loading ends.
    Serialize temporary class patches across threads; nested loads are
    reentrant.
    """
    import torch
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    with _HDO_STEP_LOAD_LOCK:
        original = DistributedOptimizer.load_state_dict

        @wraps(original)
        def load_state_dict(self: Any, state_dict: dict) -> Any:
            hdo = self.optimizer
            applicable = (
                isinstance(hdo, HybridDeviceOptimizer)
                and self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
                and not self.ddp_config.use_megatron_fsdp
                and hdo.param_update_in_fp32
                and 0 < hdo.offload_fraction <= 1.0
                and bool(hdo.cpu_optimizers)
                and all(isinstance(child, torch.optim.AdamW) for child in hdo.cpu_optimizers)
                and "param_state" in state_dict
                and state_dict.get("param_state_sharding_type") == "dp_reshardable"
            )
            if not applicable:
                return original(self, state_dict)
            steps = [group.get("step") for group in state_dict["optimizer"]["param_groups"]]
            if not steps or any(
                not isinstance(step, (int, float)) or not math.isfinite(step) or step < 0 or int(step) != step
                for step in steps
            ):
                raise ValueError("HDO dp_reshardable resume requires saved finite non-negative integer Adam steps")
            if len(set(steps)) != 1:
                raise ValueError("HDO dp_reshardable resume requires a single saved Adam step across groups")
            saved_step = steps[0]
            result = original(self, state_dict)
            # HDO and its CPU optimizers share each per-parameter state dictionary.
            # Verify the mapping before replacing scalars; no master/moment copy.
            for child in hdo.cpu_optimizers:
                for inner_param, state in child.state.items():
                    original_param = hdo.inner_param_to_orig_param[inner_param]
                    if state is not hdo.state[original_param]:
                        raise RuntimeError(
                            "Unsupported HDO state mapping: CPU and outer optimizer states do not alias"
                        )
            # Mixed offload also has GPU-owned states; FusedAdam uses group
            # counters. Repair only the CPU AdamW per-parameter counters.
            for child in hdo.cpu_optimizers:
                for state in child.state.values():
                    state["step"] = torch.tensor(saved_step, dtype=torch.float32, device="cpu")
            cpu_owned = {
                hdo.inner_param_to_orig_param[param]
                for child in hdo.cpu_optimizers
                for group in child.param_groups
                for param in group["params"]
            }
            gpu_owned = {
                hdo.inner_param_to_orig_param[param]
                for group in (hdo.gpu_optimizer.param_groups if hdo.gpu_optimizer is not None else [])
                for param in group["params"]
            }
            for group in hdo.param_groups:
                if group["params"] and all(param in cpu_owned and param not in gpu_owned for param in group["params"]):
                    # CPU AdamW advances per-parameter counters, not this saved
                    # group field. Leaving it stale breaks chained save checks.
                    # DistributedOptimizer.state_dict regenerates it from Adam.
                    group.pop("step", None)
            return result

        DistributedOptimizer.load_state_dict = load_state_dict
        try:
            yield
        finally:
            DistributedOptimizer.load_state_dict = original
