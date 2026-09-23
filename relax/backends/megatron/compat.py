# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import inspect


_BUGGY_NATIVE_FP32_LOOKUPS = ("self.param_to_fp32_param[param]", "self.param_to_fp32_param.get(param)")
_NATIVE_FP32_LOAD_PATCHED = "_relax_native_fp32_inner_load_patched"


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
