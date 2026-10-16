# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Materialize Q4(FP32 optimizer master) into the BF16 model carrier.

The CPU master remains latent and unquantized. The original model Parameter,
DDP storage and main_grad are retained. DCP model weights therefore describe
the effective forward weights at the save step, rather than an old workspace.
Only full local routed-expert masters with the installed precision-aware HDO
are supported; no hidden collectives or model.float() fallback are used.
"""

from __future__ import annotations

from functools import wraps
from typing import Any


def _master(optimizer: Any, parameter: Any) -> Any:
    import torch

    group, index = optimizer.model_param_group_index_map[parameter]
    shard = optimizer.optimizer.param_groups[group]["params"][index]
    span = optimizer._get_model_param_range_map(parameter)["param"]
    master = optimizer.optimizer.param_to_fp32_param.get(shard)
    if span.start != 0 or span.end != parameter.numel():
        raise ValueError("DeepSeek-V4 MXFP4 QAT requires a complete local expert master (expert-DP=1)")
    if master is None or master.dtype != torch.float32 or master.device.type != "cpu":
        raise TypeError("DeepSeek-V4 MXFP4 QAT requires the actual CPU FP32 HybridDeviceOptimizer working parameter")
    if master.numel() != parameter.numel() or not master.is_contiguous():
        raise ValueError("DeepSeek-V4 MXFP4 QAT master layout differs from the complete model parameter")
    if shard.data_ptr() != parameter.data_ptr():
        raise ValueError("DeepSeek-V4 MXFP4 QAT requires the optimizer GPU shard to alias the model carrier")
    return master.view(parameter.shape)


def materialize(optimizer: Any, *, reason: str = "lazy") -> None:
    """Called after master updates/load, on the training CUDA stream.

    Resolve HDO mappings anew: load_state_dict rebuilds its CPU parameters.
    Only one matrix-sized FP32 temporary is transferred at a time. HDO has
    already completed CPU Adam and enqueued its copy-back dependency before
    these hooks run; no extra host-side CUDA synchronization is needed here.
    """
    import torch

    from relax.models.deepseek_v4.quantization import mxfp4_qdq

    bindings = getattr(optimizer, "_relax_dsv4_fp4_bindings", ())
    with torch.no_grad():
        for parameter, module in bindings:
            master = _master(optimizer, parameter)
            source = master.to(device=parameter.device, non_blocking=True)
            effective = mxfp4_qdq(source)
            parameter.copy_(effective)
            # Release temporaries before transferring the next expert matrix.
            del effective, source
            module._fp8_workspaces.clear()
            module._relax_mxfp4_qat.materialized = True
    optimizer._relax_dsv4_fp4_generation = getattr(optimizer, "_relax_dsv4_fp4_generation", 0) + 1
    from relax.models.deepseek_v4.update_metrics import record_master_samples

    record_master_samples(optimizer, _master, reason=reason)


def _bind(optimizer: Any) -> None:
    import torch
    import torch.distributed as dist
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    if not hasattr(optimizer, "model_param_group_index_map"):
        if any(
            hasattr(parameter, "_relax_dsv4_fp4_owner")
            for chunk in getattr(optimizer, "model_chunks", ())
            for parameter in chunk.parameters()
        ):
            raise ValueError("DeepSeek-V4 MXFP4 QAT does not support stub/FSDP optimizers for routed experts")
        return
    bindings = []
    for parameter in optimizer.model_param_group_index_map:
        owner = getattr(parameter, "_relax_dsv4_fp4_owner", None)
        if owner is not None:
            bindings.append((parameter, owner()))
    if not bindings:
        return
    hdo = optimizer.optimizer
    if (
        not isinstance(hdo, HybridDeviceOptimizer)
        or not optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        or not hdo.param_update_in_fp32
        or hdo.offload_fraction != 1.0
        or hdo.gpu_optimizer is not None
        or not all(isinstance(child, torch.optim.AdamW) for child in hdo.cpu_optimizers)
        or optimizer.ddp_config.use_megatron_fsdp
        or getattr(optimizer.config, "overlap_param_gather_with_optimizer_step", False)
        or getattr(optimizer.ddp_config, "overlap_param_gather_with_optimizer_step", False)
        or getattr(optimizer, "_state_offloader", None) is not None
    ):
        raise ValueError("DeepSeek-V4 MXFP4 QAT supports precision-aware, fully CPU-offloaded HDO AdamW only")
    if dist.get_world_size(optimizer.data_parallel_group) != 1:
        raise ValueError("DeepSeek-V4 MXFP4 QAT requires expert-DP=1; sharded expert masters are not implemented")
    for parameter, module in bindings:
        if module is None:
            raise RuntimeError("DeepSeek-V4 MXFP4 QAT expert owner no longer exists")
        _master(optimizer, parameter)
        module._relax_mxfp4_qat.optimizer = optimizer
    optimizer._relax_dsv4_fp4_bindings = bindings
    optimizer._relax_dsv4_fp4_generation = 0
    from relax.utils.logging_utils import get_logger

    get_logger(__name__).info(
        "[DSV4-MXFP4] Bound %d complete expert matrices to CPU FP32 HDO masters; expert-DP=1, first master pinned=%s",
        len(bindings),
        _master(optimizer, bindings[0][0]).is_pinned(),
    )
    # HF weights are loaded AFTER optimizer construction. Do not quantize the
    # random model here: reload_model_params must first initialize the master.


def install_master_weight_hooks() -> bool:
    """Opt-in runtime hooks; unmarked optimizers keep their original
    behavior."""
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    if getattr(DistributedOptimizer.__init__, "_relax_dsv4_fp4", False):
        return False
    original_init = DistributedOptimizer.__init__

    @wraps(original_init)
    def initialize(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        _bind(self)

    initialize._relax_dsv4_fp4 = True
    DistributedOptimizer.__init__ = initialize

    def wrap_refresh(name: str, *, require_success: bool = False) -> None:
        original = getattr(DistributedOptimizer, name)

        @wraps(original)
        def refresh(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(self, *args, **kwargs)
            bindings = getattr(self, "_relax_dsv4_fp4_bindings", ())
            if name == "load_state_dict" and bindings:
                state = args[0] if args else kwargs["state_dict"]
                if "param_state" not in state:
                    # DCP constructs optimizer templates via load_state_dict
                    # before loading real state. Never quantize that template.
                    for _, module in bindings:
                        module._relax_mxfp4_qat.materialized = False
                        module._fp8_workspaces.clear()
                    return result
                if state.get("param_state_sharding_type") != "dp_reshardable":
                    raise ValueError("DeepSeek-V4 MXFP4 QAT resume currently requires dp_reshardable optimizer state")
            if bindings and (not require_success or result):
                materialize(self, reason=name)
            return result

        setattr(DistributedOptimizer, name, refresh)

    wrap_refresh("step_with_ready_grads", require_success=True)
    wrap_refresh("load_state_dict")
    wrap_refresh("_copy_model_params_to_main_params")
    return True
