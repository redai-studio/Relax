# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import functools
import os
import types
from typing import Any

import torch
import torch.distributed as dist

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


ForwardArgs = tuple[Any, ...]
ForwardKwargs = dict[str, Any]

_QWEN3VL_FORWARD_ARGS = [
    "input_ids",
    "position_ids",
    "attention_mask",
    "labels",
    "loss_mask",
    "inference_params",
    "packed_seq_params",
    "extra_block_kwargs",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "image_input_mask",
    "video_input_mask",
    "cp_img_num",
    "images_padded",
    "inference_context",
    "runtime_gather_output",
    "mm_token_type_ids",
]


def _rank0_log_info(message: str) -> None:
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(message)


def _should_enable(args: argparse.Namespace) -> bool:
    return any(bool(getattr(args, name, False)) for name in ("overlap_param_gather", "overlap_grad_reduce"))


def _vision_branch_sync_disabled() -> bool:
    # Opt-out for jobs guaranteed to have no image/video input (e.g. text-only SFT of a
    # VL checkpoint). The sync issues a 1-element DP all-reduce + .item() before every
    # microbatch forward purely to detect whether ANY data-parallel rank fed a vision
    # input, so idle ranks can run a dummy vision pass and keep the vision-tower
    # collectives aligned. With no multimodal input it never fires the dummy pass, yet
    # the all-reduce + GPU->CPU sync still runs each microbatch and stalls the compute
    # stream. The gate MUST be set identically on every rank (launch-script env passed
    # through RELAX_PASSTHROUGH_ENV) — a split decision would deadlock the all-reduce.
    return os.environ.get("RELAX_DISABLE_VISION_BRANCH_SYNC", "0") == "1"


def _is_qwen3vl_with_conditional_vision(model: torch.nn.Module) -> bool:
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
    except ImportError:
        return False

    return (
        isinstance(model, Qwen3VLModel)
        and bool(getattr(model, "pre_process", False))
        and getattr(model, "vision_model", None) is not None
        and not bool(getattr(model, "use_dist_train", False))
    )


def _forward_arg(name: str, args: ForwardArgs, kwargs: ForwardKwargs) -> Any:
    if name in kwargs:
        return kwargs[name]
    try:
        index = _QWEN3VL_FORWARD_ARGS.index(name)
    except ValueError:
        return None
    return args[index] if index < len(args) else None


def _tensor_has_rows(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] > 0


def _has_local_vision_input(args: ForwardArgs, kwargs: ForwardKwargs) -> bool:
    for grid_name, values_name, mask_name in (
        ("image_grid_thw", "pixel_values", "image_input_mask"),
        ("video_grid_thw", "pixel_values_videos", "video_input_mask"),
    ):
        grid = _forward_arg(grid_name, args, kwargs)
        if _tensor_has_rows(grid):
            return True

        values = _forward_arg(values_name, args, kwargs)
        if grid is None and _tensor_has_rows(values):
            return True

        mask = _forward_arg(mask_name, args, kwargs)
        if isinstance(mask, torch.Tensor) and bool(mask.any().item()):
            return True
    return False


def _find_tensor_device(value: Any) -> torch.device | None:
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, dict):
        for item in value.values():
            device = _find_tensor_device(item)
            if device is not None:
                return device
    if isinstance(value, (list, tuple)):
        for item in value:
            device = _find_tensor_device(item)
            if device is not None:
                return device
    return None


def _first_tensor_device(args: ForwardArgs, kwargs: ForwardKwargs) -> torch.device:
    for value in list(args) + list(kwargs.values()):
        device = _find_tensor_device(value)
        if device is not None:
            return device
    return torch.device("cuda", torch.cuda.current_device())


def _dp_any(local_active: bool, group: dist.ProcessGroup, device: torch.device) -> bool:
    if group is None or group.size() == 1:
        return local_active

    active = torch.tensor([1 if local_active else 0], device=device, dtype=torch.int32)
    dist.all_reduce(active, op=dist.ReduceOp.MAX, group=group)
    return bool(active.item())


def _sum_tensors(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.sum()
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return None

    total = None
    for item in values:
        result = _sum_tensors(item)
        if result is not None:
            total = result if total is None else total + result
    return total


def _run_dummy_vision(model: torch.nn.Module, device: torch.device) -> torch.Tensor | None:
    vision_model = getattr(model, "vision_model", None)
    if vision_model is None:
        return None

    vision_config = model.vision_transformer_config
    temporal = 1
    height = vision_config.spatial_merge_size
    width = vision_config.spatial_merge_size
    feature_dim = (
        vision_config.in_channels
        * vision_config.temporal_patch_size
        * vision_config.patch_size
        * vision_config.patch_size
    )

    dtype = vision_model.patch_embed.proj.weight.dtype
    dummy_data = torch.zeros((temporal * height * width, feature_dim), device=device, dtype=dtype)
    dummy_grid_thw = torch.tensor([[temporal, height, width]], device=device, dtype=torch.long)
    return _sum_tensors(vision_model(hidden_states=dummy_data, grid_thw=dummy_grid_thw))


def _attach_zero_dependency(
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor], dependency: torch.Tensor
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output + dependency.to(dtype=output.dtype, device=output.device) * 0
    # Bridge's CP path returns (output, sliced_loss_mask). Only the output is
    # consumed by backward; the mask must keep its identity and gradient state.
    if isinstance(output, tuple) and len(output) == 2 and all(isinstance(value, torch.Tensor) for value in output):
        hidden_states, loss_mask = output
        return _attach_zero_dependency(hidden_states, dependency), loss_mask
    raise TypeError(
        "Qwen3VL conditional branch sync expected a tensor or (output, loss_mask) tensor pair, "
        f"got {type(output).__name__}"
    )


def install_conditional_branch_sync(args: argparse.Namespace, model: torch.nn.Module) -> None:
    if getattr(model, "_relax_conditional_branch_sync_installed", False):
        return
    if not _should_enable(args) or not _is_qwen3vl_with_conditional_vision(model):
        return
    if _vision_branch_sync_disabled():
        _rank0_log_info(
            "Skipped Qwen3VL conditional vision branch sync (RELAX_DISABLE_VISION_BRANCH_SYNC=1): "
            "text-only job, dropping the per-microbatch vision-presence all-reduce."
        )
        return

    original_forward = model.forward

    @functools.wraps(original_forward)
    def wrapped_forward(self, *forward_args, **forward_kwargs):
        if not self._relax_conditional_branch_sync_active:
            return original_forward(*forward_args, **forward_kwargs)
        device = _first_tensor_device(forward_args, forward_kwargs)
        local_active = _has_local_vision_input(forward_args, forward_kwargs)
        global_active = _dp_any(local_active, self.pg_collection.dp, device)
        dependency = _run_dummy_vision(self, device) if global_active and not local_active else None

        output = original_forward(*forward_args, **forward_kwargs)
        if dependency is not None:
            output = _attach_zero_dependency(output, dependency)
        return output

    model._relax_conditional_branch_sync_original_forward = original_forward
    model._relax_conditional_branch_sync_specs = ("qwen3vl_vision",)
    model._relax_conditional_branch_sync_installed = True
    model._relax_conditional_branch_sync_active = True
    model.forward = types.MethodType(wrapped_forward, model)

    _rank0_log_info("Installed Qwen3VL conditional vision branch sync.")


def disable_frozen_conditional_branch_sync(model: torch.nn.Module) -> None:
    """Finalize the adapter after all provider and name-based freezing.

    Qwen3VL's merger and DeepStack projections are children of vision_model. A
    wholly frozen branch has no DDP parameter or gradient collective that
    requires the extra cross-DP dummy forward. Keep the wrapper chain intact.
    """
    if not getattr(model, "_relax_conditional_branch_sync_installed", False):
        return
    if getattr(model, "_relax_conditional_branch_sync_specs", ()) != ("qwen3vl_vision",):
        return
    vision = getattr(model, "vision_model", None)
    if vision is not None and not any(param.requires_grad for param in vision.parameters()):
        model._relax_conditional_branch_sync_active = False
        _rank0_log_info("Disabled conditional vision branch sync for the fully frozen vision/projection subtree.")
