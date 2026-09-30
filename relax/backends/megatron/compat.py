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
    """Preserve saved Adam steps during full CPU-offloaded HDO DCP loads.

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
                and hdo.offload_fraction == 1.0
                and hdo.gpu_optimizer is None
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
            for state in hdo.state.values():
                state["step"] = torch.tensor(saved_step, dtype=torch.float32, device="cpu")
            return result

        DistributedOptimizer.load_state_dict = load_state_dict
        try:
            yield
        finally:
            DistributedOptimizer.load_state_dict = original
