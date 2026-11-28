# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from types import MethodType


def _qhc(self, hyper_input):
    from .performance_ops.qhc_triton import _QHCFusedFunction

    if hyper_input.device.type not in ("npu", "privateuseone"):
        raise RuntimeError("Selected Triton QHC requires NPU tensors")
    injection = self.block_inject_weight
    result = _QHCFusedFunction.apply(
        hyper_input,
        self.hc_norm.weight,
        self.input_mix_weight_down.weight,
        self.input_mix_weight_up.weight,
        injection.weight if injection is not None else None,
        self.hc_count,
        self.hidden_size,
        self.hc_norm.eps,
    )
    if injection is None:
        return result
    mixed, weights = result
    return mixed, hyper_input, weights


def install(model, options) -> None:
    if options.qhc_implementation != "triton":
        return
    count = 0
    for module in model.modules():
        if type(module).__name__ == "Qwen4ExpTextGatedResidual":
            module.forward = MethodType(_qhc, module)
            count += 1
    if not count:
        raise ValueError("Selected qhc_implementation matched no model modules")
