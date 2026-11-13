# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in, topology-independent rejection of nonfinite pre-update gradient
norms."""

import os


class NonFiniteGradientNormError(FloatingPointError):
    """No child optimizer has updated when this exception is raised."""


def skip_nonfinite_enabled() -> bool:
    return os.environ.get("RELAX_SKIP_NONFINITE_GRAD", "0") == "1"


def configure_nonfinite_guard(args: object) -> bool:
    """Select step-level handling before DDP constructs its gradient
    buckets."""
    if not skip_nonfinite_enabled() or getattr(args, "only_load_weight", False):
        return False
    if not getattr(args, "bf16", False) or getattr(args, "fp16", False):
        raise ValueError("Nonfinite step skipping currently requires BF16 (no dynamic loss scaler)")
    for name in (
        "fp8",
        "fp4",
        "use_megatron_fsdp",
        "reuse_grad_buf_for_mxfp8_param_ag",
        "chunked_optimizer_state_offload",
        "offload_optimizer_states",
        "overlap_param_gather_with_optimizer_step",
    ):
        if getattr(args, name, False):
            raise ValueError(f"Nonfinite step skipping has not validated {name}")
    # A per-bucket exception can strand other ranks in collectives. Let gradient
    # communication finish, then make one shared decision before any update.
    args.check_for_nan_in_loss_and_grad = False
    return True


def check_global_finite_norm(value: object, *, process_group: object, device: object) -> None:
    """Reduce only the nonfinite flag; do not change the optimizer's norm
    math."""
    import torch
    import torch.distributed as dist

    norm = torch.as_tensor(value, dtype=torch.float64, device=device)
    bad = (~torch.isfinite(norm)).any().to(dtype=torch.int32)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=process_group)
    if bad.item():
        raise NonFiniteGradientNormError(
            "Non-finite global gradient norm before optimizer update "
            "(NaN/Inf on at least one rank; all ranks must skip)"
        )


def install_nonfinite_guard(optimizer: object, model: list, *, process_group: object) -> None:
    """Guard the chained norm call after prepare_grads and before child
    updates.

    No TP/PP/CP/EP size assumptions. HybridDeviceOptimizer CPU offload executes
    its copies/Adam updates inside child.step(), after this decision. Chunked
    state offload and parameter/gradient storage reuse are rejected separately.
    """
    from types import MethodType

    import torch.distributed as dist
    from megatron.core.optimizer.optimizer import (
        ChainedOptimizer,
        _get_param_grad_norm_group,
        _is_separate_grad_norm_group,
    )

    from relax.utils.device import make_current_torch_device
    from relax.utils.logging_utils import get_logger

    if getattr(optimizer, "_relax_nonfinite_guard_installed", False):
        return
    if type(optimizer) is not ChainedOptimizer:
        raise TypeError("Nonfinite step skipping requires the verified Megatron ChainedOptimizer.step contract")
    for child in optimizer.chained_optimizers:
        if getattr(child, "grad_scaler", None) is not None:
            raise ValueError("Nonfinite step skipping cannot bypass a dynamic loss scaler")
        for name in (
            "reuse_grad_buf_for_mxfp8_param_ag",
            "chunked_optimizer_state_offload",
            "overlap_param_gather_with_optimizer_step",
        ):
            if getattr(child.config, name, False):
                raise ValueError(f"Nonfinite step skipping has not validated optimizer config {name}")
        if any(_is_separate_grad_norm_group(_get_param_grad_norm_group(p)) for p in child.get_parameters()):
            raise ValueError("Separate gradient-norm groups need an additional pre-update guard")
    if any(getattr(chunk.ddp_config, "check_for_nan_in_grad", False) for chunk in model):
        raise ValueError("Per-bucket NaN checks must be disabled before enabling synchronized step skipping")
    device = "cpu" if dist.get_backend(process_group) == "gloo" else make_current_torch_device()
    original_get_norm = optimizer.get_grad_norm

    def guarded_get_norm(self: object) -> object:
        norm = original_get_norm()
        check_global_finite_norm(norm, process_group=process_group, device=device)
        return norm

    optimizer.get_grad_norm = MethodType(guarded_get_norm, optimizer)
    optimizer._relax_nonfinite_guard_installed = True
    get_logger(__name__).warning(
        "Nonfinite optimizer-step skipping ENABLED: synchronized pre-update decision across %d ranks; "
        "bucket NaN exceptions replaced by step-level ERROR diagnostics. TP/PP/CP/EP sizes are unrestricted.",
        dist.get_world_size(group=process_group),
    )
