# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from types import MethodType


def _gated_norm(self, hidden_states, gate):
    from .performance_ops.rmsnorm_gated_triton import _fast_forward_available, _Qwen4RMSNormGatedFunction

    if self.activation not in ("silu", "sigmoid") or not _fast_forward_available(hidden_states, gate, self.weight):
        raise RuntimeError("Selected Triton RMSNormGated does not support this dtype, layout or activation")
    return _Qwen4RMSNormGatedFunction.apply(
        hidden_states, gate, self.weight, self.variance_epsilon, self.activation, 32, 40, 32, True
    )


def install(model, options) -> None:
    if options.rmsnormgated_implementation != "triton":
        return
    count = 0
    for module in model.modules():
        if type(module).__name__ == "Qwen4ExpTextRMSNormGated":
            module.forward = MethodType(_gated_norm, module)
            count += 1
    if not count:
        raise ValueError("Selected rmsnormgated_implementation matched no model modules")
