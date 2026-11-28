# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Explicit GDN and causal convolution backends for Qwen4-Exp."""

import importlib
import inspect
from functools import wraps
from types import MethodType


def causal_conv1d(self, x, weight, bias=None, activation=None, seq_idx=None):
    from fsdp_turbo.ops.causal_conv1d import causal_conv1d as convolution

    if seq_idx is not None:
        raise ValueError("Ulysses convolution requires unpacked sequences")
    channels = x.shape[1]
    heads = channels // self.head_v_dim if channels % self.head_v_dim == 0 else 1
    while channels // heads > 4096 and channels % (heads * 2) == 0:
        heads *= 2
    inputs = {
        "x": x.transpose(1, 2).contiguous(),
        "weight": weight.transpose(0, 1).contiguous(),
        "bias": bias,
        "activation": activation,
    }
    output, _ = convolution(**inputs, head_num=heads)
    return output.transpose(1, 2).contiguous()


def _triton_convolution(*args, head_num=1, implementation="auto", **kwargs):
    from fsdp_turbo.ops.triton.causal_conv1d import causal_conv1d_triton

    return causal_conv1d_triton(*args, **kwargs)


def configure(config, text_config) -> None:
    """Preserve the model defaults unless a backend or policy is selected."""
    options = config.relax_performance
    if options.skip_gdn_recompute is not None and options.gdn_implementation == "auto":
        raise ValueError("Explicit skip_gdn_recompute requires a specific GDN implementation")
    if options.skip_gdn_recompute is not None:
        text_config.skip_gdn_recompute = options.skip_gdn_recompute
    if options.causal_conv1d_implementation == "triton":
        # CP1 recognizes "triton" and forwards cu_seqlens through the registry.
        from fsdp_turbo.ops.registry import register_op

        register_op("causal_conv1d", "npu")(_triton_convolution)
        text_config.causal_conv1d_implementation = "triton"
    elif options.causal_conv1d_implementation == "ascendc":
        conv = importlib.import_module("fsdp_turbo.ops.npu.causal_conv1d")
        if conv._CAUSAL_CONV1D_BACKEND != "ascendc":
            raise RuntimeError("AscendC causal_conv1d is unavailable")
        text_config.causal_conv1d_implementation = "triton"
    if options.gdn_implementation != "auto":
        from fsdp_turbo.ops.registry import register_op

        if options.gdn_implementation == "ascendc":
            from fsdp_turbo.ops.npu.chunk_gdn_ascendc import chunk_gdn_ascendc as implementation
        else:
            from fsdp_turbo.ops.fla_compat import stock_chunk_gated_delta_rule

            implementation = stock_chunk_gated_delta_rule()
        supports_skip = "skip_recompute" in inspect.signature(implementation).parameters
        if text_config.skip_gdn_recompute and not supports_skip:
            raise ValueError("Selected GDN lacks skip_recompute; explicitly set skip_gdn_recompute=false")

        @wraps(implementation)
        def explicit_gdn(*args, **kwargs):
            kwargs.pop("implementation", None)
            if supports_skip and options.skip_gdn_recompute is not None:
                kwargs["skip_recompute"] = options.skip_gdn_recompute
            elif not supports_skip:
                kwargs.pop("skip_recompute", None)
            return implementation(*args, **kwargs)

        register_op("chunk_gated_delta_rule", "npu")(explicit_gdn)


def install(model, config) -> None:
    """Attach grouped-QK and CP convolution only when requested."""
    options = config.relax_performance
    for module in model.modules():
        if type(module).__name__ != "Qwen4ExpTextGatedDeltaNet":
            continue
        if options.gdn_grouped_qk:
            module.gdn_grouped_qk = True
        if config.distributed.ulysses_parallel_size > 1 and options.causal_conv1d_implementation != "auto":
            module.causal_conv1d_fn = MethodType(causal_conv1d, module)
