# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import inspect


_BUGGY_NATIVE_FP32_LOOKUP = "self.param_to_fp32_param[param]"
_NATIVE_FP32_LOAD_PATCHED = "_relax_native_fp32_load_patched"


def _patch_hybrid_optimizer_class(optimizer_cls: type) -> bool:
    """Backport Megatron-LM 86e928a for checkpoint loads on older images."""
    update_method = getattr(optimizer_cls, "_update_fp32_params_by_new_state", None)
    if update_method is None or getattr(update_method, _NATIVE_FP32_LOAD_PATCHED, False):
        return False

    try:
        source = inspect.getsource(update_method)
    except (OSError, TypeError):
        return False
    if _BUGGY_NATIVE_FP32_LOOKUP not in source:
        return False

    def _update_fp32_params_by_new_state(self) -> None:
        if not self.param_update_in_fp32:
            return
        for param, state in self.state.items():
            # Native FP32 params are intentionally absent: they are already the
            # master params and must not be copied through a separate shadow.
            fp32_param = self.param_to_fp32_param.get(param)
            if fp32_param is not None:
                fp32_param.data.copy_(state["master_param"])

    setattr(_update_fp32_params_by_new_state, _NATIVE_FP32_LOAD_PATCHED, True)
    optimizer_cls._update_fp32_params_by_new_state = _update_fp32_params_by_new_state
    return True


def patch_hybrid_optimizer_native_fp32_checkpoint_load() -> bool:
    """Patch the native-FP32 HDO load bug in pre-86e928a Megatron images."""
    try:
        from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
    except ImportError:
        return False
    return _patch_hybrid_optimizer_class(HybridDeviceOptimizer)
