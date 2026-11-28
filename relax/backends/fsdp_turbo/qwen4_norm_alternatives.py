# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from types import MethodType


def _grouped_norm(self, hidden_states):
    from .performance_ops.grouped_rmsnorm import grouped_rmsnorm

    return grouped_rmsnorm(hidden_states, self.weight, self.group_size or self.weight.numel(), self.eps)


def install(model, options) -> None:
    counts = {"rmsnorm": 0, "grouped_rmsnorm": 0, "qhc": 0}
    for module in model.modules():
        name = type(module).__name__
        if name == "Qwen4ExpTextRMSNorm":
            key = "grouped_rmsnorm" if module.group_size is not None else "rmsnorm"
            if getattr(options, key + "_implementation") == "triton":
                module.forward = MethodType(_grouped_norm, module)
                counts[key] += 1
        elif name == "Qwen4ExpTextGatedResidual" and options.qhc_implementation == "ascendc":
            from .performance_ops.qhc_ascendc import qhc_forward

            module.forward = MethodType(qhc_forward, module)
            counts["qhc"] += 1
    for key, count in counts.items():
        selected = getattr(options, key + "_implementation")
        if selected != "native" and not count:
            raise ValueError(f"Selected {key} optimization matched no model modules")
