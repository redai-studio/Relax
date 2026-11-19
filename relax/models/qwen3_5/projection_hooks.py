# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Select a prebuilt exact-size UB by saved name, with native fallback and no
padding."""

import math

from relax.models.qwen3_5.userbuffers import CAPACITIES


FIELDS = (
    "ub_overlap_ag_fprop",
    "ub_overlap_ag_dgrad",
    "ub_overlap_rs_fprop",
    "ub_overlap_rs_dgrad",
    "ub_bulk_wgrad",
    "ub_bulk_dgrad",
)


def enable_projection(module, name, group):
    import torch.distributed as dist
    from transformer_engine.pytorch.module import base

    geometry = {
        "fc1": ("column", 2048, 3088),
        "qkv": ("column", 2048, 2304),
        "fc2": ("row", 1024, 2048),
        "proj": ("row", 1024, 2048),
    }
    mode, inp, out = geometry[name]
    assert base.is_ub_initialized()
    assert module.parallel_mode == mode
    assert module.tp_size == 4 and module.sequence_parallel
    assert module.tp_group is group
    assert dist.get_world_size(group) == 4
    assert (module.in_features, module.out_features) == (inp, out)
    assert tuple(module.weight.shape) == (out, inp)
    assert all(getattr(module, field) is False for field in FIELDS)
    # WeightGradStore is an installed-TE API pinned by the gate input manifest.
    assert module.wgrad_store.delay_wgrad_compute() is False
    assert getattr(module, "config", None) is None or module.config.tp_comm_overlap is False
    module.ub_name = name + "_s262144"
    if mode == "column":
        module.ub_overlap_ag_fprop = True
        module.ub_overlap_rs_dgrad = True
    else:
        module.ub_overlap_rs_fprop = True
        module.ub_overlap_ag_dgrad = True
    active = {field: getattr(module, field) for field in FIELDS}
    module._qwen_exact_ub_state = dict(entered=False, ub_forwards=0, native_forwards=0, name=name, active=active)
    return active


def run_projection(module, inputs, native_call=None):
    """Select exact buffers for one ordinary module call and restore on
    errors."""
    if native_call is None:
        native_call = module
    state = getattr(module, "_qwen_exact_ub_state", None)
    if state is None:
        return native_call(inputs)
    assert not state["entered"], "Recursive UB invocation is unsupported"
    previous = {field: getattr(module, field) for field in FIELDS}
    previous_name = module.ub_name
    rows = math.prod(inputs.shape[:-1])
    if module.parallel_mode == "column":
        rows *= 4
    assert rows > 0 and rows % 4 == 0
    use_ub = rows in CAPACITIES
    if use_ub:
        module.ub_name = f"{state['name']}_s{rows}"
    for field in FIELDS:
        setattr(module, field, state["active"][field] if use_ub else False)
    state["entered"] = True
    state["ub_forwards" if use_ub else "native_forwards"] += 1
    try:
        return native_call(inputs)
    finally:
        module.ub_name = previous_name
        for field in FIELDS:
            setattr(module, field, previous[field])
        state["entered"] = False
