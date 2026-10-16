# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Routed expert MXFP4 QAT: FP32 master -> MXFP4 carrier -> BF16/FP8 GEMM.

The optimizer hooks materialize effective FP4 values in the BF16 Parameter. TE
receives that original Parameter, preserving fused main_grad/STE. Native mode
can override FP8 weights with P2 scales; stock_fp8 retains the original TE
computation and quantizers. DCP saves the same carrier.
"""

from __future__ import annotations

import copy
import os
import weakref
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable


if TYPE_CHECKING:
    import torch


@dataclass
class _QATState:
    original_workspace: Callable[..., Any]
    module_name: str
    workspace_calls: int = 0
    optimizer: Any = None
    materialized: bool = False
    compute: str = "bf16"


def _fp4_mode() -> str:
    from relax.utils.env import Envs

    mode = Envs.RELAX_DSV4_FP4_MODE
    if mode not in ("native", "stock_fp8"):
        raise ValueError("RELAX_DSV4_FP4_MODE must be native or stock_fp8")
    if mode == "stock_fp8" and os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0") != "1":
        raise ValueError("stock_fp8 MXFP4 QAT requires NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1")
    return mode


def _ensure_materialized(module: Any, inputs: tuple[Any, ...]) -> None:
    from relax.models.deepseek_v4.master_weights import materialize

    state: _QATState = module._relax_mxfp4_qat
    if state.optimizer is None:
        raise RuntimeError(f"{state.module_name}: DeepSeek-V4 MXFP4 QAT requires the actual distributed HDO optimizer")
    if not state.materialized:
        materialize(state.optimizer)


def _mxfp4_weight_workspace(
    self: Any,
    *,
    tensor: torch.Tensor | None = None,
    quantizer: Any = None,
    cache_name: str | None = None,
    update_workspace: bool = True,
    skip_update_flag: torch.Tensor | None = None,
    fsdp_group: Any = None,
    workspace_dtype: torch.dtype | None = None,
) -> Any:
    import torch
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
    from transformer_engine.pytorch.tensor.storage.float8_blockwise_tensor_storage import (
        Float8BlockwiseQTensorStorage,
    )

    from relax.models.deepseek_v4.master_weights import materialize

    state: _QATState = self._relax_mxfp4_qat
    state.workspace_calls += 1
    if state.optimizer is None:
        raise RuntimeError(
            f"{state.module_name}: DeepSeek-V4 MXFP4 QAT requires binding to the actual distributed HDO optimizer"
        )
    if not state.materialized:
        materialize(state.optimizer)
    if fsdp_group is not None:
        raise RuntimeError("MXFP4 QAT workspace sharding with FSDP has not been validated")
    if not isinstance(quantizer, Float8BlockQuantizer) or quantizer.block_scaling_dim != 2:
        raise TypeError(f"{state.module_name}: MXFP4 QAT requires the TE 2D Float8BlockQuantizer")
    if str(quantizer.dtype).split(".")[-1] != "kFloat8E4M3":
        raise TypeError(f"{state.module_name}: MXFP4 QAT requires E4M3 forward weights")
    if tensor is not None and (
        type(tensor) not in (torch.Tensor, torch.nn.Parameter)
        or tensor.dtype not in (torch.bfloat16, torch.float32)
        or tensor.ndim != 2
        or not tensor.is_contiguous()
        or not quantizer.is_quantizable(tensor)
    ):
        raise TypeError(
            f"{state.module_name}: expected a contiguous BF16/FP32 model weight with both dimensions divisible by 128"
        )

    # Recipe QParams may be class attributes. Copy the individual weight
    # quantizer; never change global input/gradient/dense/shared quantizers.
    weight_quantizer = quantizer.copy()
    weight_quantizer.force_pow_2_scales = True
    out = self._fp8_workspaces.get(cache_name) if cache_name is not None else None
    if out is not None:
        if not isinstance(out, Float8BlockwiseQTensorStorage):
            raise TypeError(f"{state.module_name}: incompatible existing FP8 weight workspace")
        if not out._quantizer.force_pow_2_scales:
            raise RuntimeError(f"{state.module_name}: found a workspace created before MXFP4 QAT installation")
        missing_usage = (weight_quantizer.rowwise_usage and out._rowwise_data is None) or (
            weight_quantizer.columnwise_usage and out._columnwise_data is None
        )
        if missing_usage or (tensor is not None and tuple(out.size()) != tuple(tensor.shape)):
            # TE's generic reset logic currently omits Float8Blockwise storage.
            # A newly required dgrad transpose must not reuse an incomplete cache.
            del self._fp8_workspaces[cache_name]
            out = None
        else:
            # Cached tensor.quantize_ uses its own copied quantizer, not the
            # quantizer argument below. Preserve that copy's storage lifetime.
            cached_quantizer = weight_quantizer.copy()
            cached_quantizer.internal = out._quantizer.internal
            out._quantizer = cached_quantizer

    # Match TE semantics exactly: a cache miss initializes even when update is
    # False; a GPU skip flag takes precedence over the Python update flag. Do
    # not inspect the GPU flag on CPU. QDQ already happened after the master
    # update; here TE receives that effective carrier and its original flags.
    if (out is None or update_workspace or skip_update_flag is not None) and tensor is None:
        raise ValueError(f"{state.module_name}: model weight is required to create/update the QAT workspace")

    return state.original_workspace(
        tensor=tensor,
        quantizer=weight_quantizer,
        cache_name=cache_name,
        update_workspace=update_workspace,
        skip_update_flag=skip_update_flag,
        fsdp_group=fsdp_group,
        workspace_dtype=workspace_dtype,
    )


def install_routed_expert_qat(module: torch.nn.Module, *, module_name: str, compute: str = "bf16") -> None:
    """Install on one explicitly selected MCore TEGroupedLinear before forward.

    The caller is responsible for selecting a routed expert. Prefer the tree
    installer below in training. Calling twice on the same module is harmless.
    ``stock_fp8`` keeps TE's original workspace and precision context; ``fp8``
    is the native mode's weight-only P2 override.
    """
    import torch
    from megatron.core.extensions.transformer_engine import TEGroupedLinear, TEQuantizationParams, TEQuantizationRecipe

    if compute not in ("bf16", "fp8", "stock_fp8"):
        raise ValueError("DeepSeek-V4 MXFP4 expert compute must be bf16, fp8 or stock_fp8")
    if not isinstance(module, TEGroupedLinear):
        raise TypeError(f"{module_name}: routed QAT currently supports MCore TEGroupedLinear only")
    if hasattr(module, "_relax_mxfp4_qat"):
        if isinstance(module._relax_mxfp4_qat, _QATState) and module._relax_mxfp4_qat.compute == compute:
            return
        raise ValueError(f"{module_name}: an incompatible QAT hook is already installed")
    if os.environ.get("OPEN_TRAINING_INT4_FAKE_QAT_FLAG", "0") != "0":
        raise RuntimeError("MXFP4 QAT cannot be combined with the existing integer INT4 fake-QAT hooks")
    config = module.config
    if getattr(config, "use_transformer_engine_op_fuser", False):
        raise ValueError(f"{module_name}: TE op fuser bypasses MXFP4 module precision hooks")
    if not getattr(config, "fp8", None) or getattr(config, "fp8_recipe", None) != "blockwise":
        raise ValueError(f"{module_name}: MXFP4 workspace QAT requires configured blockwise FP8 training")
    if getattr(module, "te_quant_params", None) is not None:
        raise ValueError(
            f"{module_name}: per-module quant_recipe overrides are not supported by DeepSeek-V4 MXFP4 QAT"
        )
    if getattr(module, "primary_weights_in_fp8", False) or getattr(config, "fp8_param", False):
        raise ValueError(f"{module_name}: QAT requires high-precision model Parameters, not primary FP8 weights")
    if getattr(config, "moe_single_grouped_weight", False):
        raise ValueError(f"{module_name}: single_grouped_weight storage has not been validated for this QAT hook")
    if module._fp8_workspaces:
        raise RuntimeError(
            f"{module_name}: install MXFP4 QAT before the first forward (workspace cache must be empty)"
        )
    module._relax_mxfp4_qat = _QATState(module.get_weight_workspace, module_name, compute=compute)
    for parameter in module.parameters(recurse=False):
        if parameter.ndim != 2 or parameter.dtype != torch.bfloat16:
            raise TypeError(
                f"{module_name}: DeepSeek-V4 MXFP4 QAT requires BF16 matrix expert Parameters without bias"
            )
        parameter._relax_dsv4_fp4_owner = weakref.ref(module)
    module.register_forward_pre_hook(_ensure_materialized)
    if compute == "bf16":
        # MCore installs this context INSIDE forward, including recomputation.
        # Empty recipes override the enclosing FP8 context for this module only.
        module.te_quant_params = TEQuantizationParams(
            training_recipe=TEQuantizationRecipe(), evaluation_recipe=TEQuantizationRecipe()
        )
    elif compute == "fp8":
        module.get_weight_workspace = MethodType(_mxfp4_weight_workspace, module)


def install_mxfp4_qat(model: torch.nn.Module, *, compute: str = "bf16") -> tuple[str, ...]:
    """Select only ``*.experts.linear_fc{1,2}``, leaving shared/dense
    unchanged."""
    selected = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if "experts" in parts and "shared_experts" not in parts and parts[-1] in ("linear_fc1", "linear_fc2"):
            install_routed_expert_qat(module, module_name=name, compute=compute)
            selected.append(name)
    if not selected:
        raise ValueError("MXFP4 QAT found no routed expert TEGroupedLinear modules on this model stage")
    return tuple(selected)


def model_provider(
    pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None
) -> torch.nn.Module:
    """Delegate construction to Relax, then install instance-scoped QAT
    hooks."""
    from megatron.training.global_vars import get_args

    from relax.backends.megatron.model_provider import get_model_provider_func
    from relax.models.deepseek_v4.master_weights import install_master_weight_hooks
    from relax.utils.env import Envs
    from relax.utils.logging_utils import get_logger

    mode = _fp4_mode()
    compute = "stock_fp8" if mode == "stock_fp8" else Envs.RELAX_DSV4_FP4_EXPERT_COMPUTE
    if mode == "native" and compute not in ("bf16", "fp8"):
        raise ValueError("RELAX_DSV4_FP4_EXPERT_COMPUTE must be bf16 or fp8")
    args = copy.copy(get_args())
    args.custom_model_provider_path = None
    if args.megatron_to_hf_mode != "bridge" or not args.bf16:
        raise ValueError("The DeepSeek-V4 MXFP4 provider currently supports Bridge BF16 model-copy training only")
    install_master_weight_hooks()
    model = get_model_provider_func(args, role="actor")(
        pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
    )
    if model.config.experimental_attention_variant != "dsv4_hybrid" or model.config.tensor_model_parallel_size != 1:
        raise ValueError("DeepSeek-V4 MXFP4 QAT supports DSv4 hybrid with TP=1 only")
    selected = install_mxfp4_qat(model, compute=compute)
    get_logger(__name__).info(
        "[DSV4-MXFP4] mode=%s; installed on %d routed expert modules; direct FP32 HDO master -> K32/E8M0 "
        "MXFP4 model carrier -> %s expert GEMM. DCP model saves effective weights; optimizer retains latent FP32. "
        "Requires expert-DP=1 and complete optimizer state for exact training resume.",
        mode,
        len(selected),
        "FP8 (stock recipe)" if compute == "stock_fp8" else compute.upper(),
    )
    return model
