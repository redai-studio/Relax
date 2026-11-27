# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in MegaMoE dispatcher integration."""

from types import MethodType


def configure(config, *, sft: bool) -> None:
    """Select the existing custom expert dispatcher for SFT."""
    if config.relax_performance.moe_implementation == "native":
        return
    if not sft:
        raise ValueError("MegaMoE currently supports SFT only")
    from hyper_multicore import MegaMoeExperts  # noqa: F401

    plan = config.distributed.ep_plan
    if not plan.apply_modules or not plan.use_swiglu:
        raise ValueError("MegaMoE requires routed experts and SwiGLU")
    quantization = config.quantization.quantization_plan
    if quantization is not None and (quantization.converters or quantization.quant_recipe):
        raise ValueError("MegaMoE does not support the quantization plan")
    plan.dispatcher = "custom_megamoe_forward"


def install(model, config) -> None:
    """Reuse existing expert weights without creating duplicate parameters."""
    if config.relax_performance.moe_implementation == "native":
        return
    from .performance_ops.megamoe import megamoe_forward

    count = 0
    for module in model.modules():
        if type(module).__name__ == "Qwen4ExpTextExperts":
            module.megamoe_forward = MethodType(megamoe_forward, module)
            count += 1
    if not count:
        raise ValueError("Selected MegaMoE matched no model modules")
