# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fresh-process fixed launch policy; no imports, CUDA calls or math
rewrites."""

import inspect


def install(math_module, root):
    """Call once immediately after loading immutable copied math backends."""
    from fla.modules.conv.triton.conv_policy import CONV_NB_MODE

    if CONV_NB_MODE != "none":
        raise RuntimeError("Qwen3.5 optimized recipe requires RELAX_FLA_CONV_NB_MODE=none before importing FLA")
    math_module.load_backends(root)
    # Verify the real imported FLA none-NB variant keys are T-independent.
    from . import causal_conv_backward as adapter

    checked = []
    for name in ["causal_conv1d_fwd_kernel", "causal_conv1d_bwd_kernel"]:
        obj = (
            inspect.unwrap(adapter.causal_conv1d_fwd).__globals__[name]
            if name.endswith("fwd_kernel")
            else adapter.causal_conv1d_bwd_kernel
        )
        while not hasattr(obj, "configs"):
            assert hasattr(obj, "fn"), name
            obj = obj.fn
        assert obj.keys == ["D", "W"], (name, obj.keys)
        checked.append({"name": name, "keys": list(obj.keys), "configs": len(obj.configs)})
    return {
        "fixed": ["conv QK16/V32 W2 S2"],
        "fla_autotuners": checked,
        "fla_dynamic_T_benchmark": False,
    }
