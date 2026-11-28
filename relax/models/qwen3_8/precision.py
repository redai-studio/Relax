# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Selective MoE FP8 compute and parameter storage for Qwen3.8 Flash Next.

The GDN input projection contains small gate sections which make its TP shards
incompatible with block-scaled FP8. Keep attention, vision, PLE and MTP in
their original precision while quantizing the backbone's aligned expert GEMMs.
The global recipe stays enabled for expert padding and activation
recomputation.
"""

from argparse import Namespace
from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from relax.utils.logging_utils import get_logger
from relax.utils.quant_cast import get_hf_quantization_config


logger = get_logger(__name__)
_MXFP8_BLOCK_SIZE = 32


def _mxfp8_shared_linear(
    forward: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]],
    x: torch.Tensor,
    *,
    fc1: torch.nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Align the TP1 dense GEMM locally, keeping padding out of the shared gate
    and MoE output."""
    rows = x.numel() // x.size(-1)
    pad = -rows % _MXFP8_BLOCK_SIZE
    if not pad:
        return forward(x)
    padded = torch.nn.functional.pad(x.reshape(rows, x.size(-1)), (0, 0, 0, pad))
    if fc1 is not None and fc1.save_original_input:
        # This BF16 allocation is private to FC1. TE captures the save policy during
        # forward; let it retain the columnwise MXFP8 activation for weight gradients.
        fc1.save_original_input = False
        try:
            output, bias = forward(padded)
        finally:
            fc1.save_original_input = True
    else:
        output, bias = forward(padded)
    return output[:rows].reshape(*x.shape[:-1], output.size(-1)), bias


def is_qwen_fp8_source(bridge: Any) -> bool:
    config = getattr(getattr(bridge, "hf_pretrained", None), "config", None)
    return (
        getattr(config, "model_type", None) in {"qwen4_exp", "qwen4_exp_text"}
        and (get_hf_quantization_config(config) or {}).get("quant_method") == "fp8"
    )


def get_bf16_reference_loader(bridge: Any) -> Callable[[str], torch.Tensor] | None:
    """Use the HF import scales for MTP or vision weights copied during plain
    export."""
    if not is_qwen_fp8_source(bridge):
        return None
    model_bridge = bridge._model_bridge

    def load(name: str) -> torch.Tensor:
        tensor = model_bridge._load_fp8_hf_weight(name, bridge.hf_pretrained.state)
        return tensor.bfloat16() if tensor.is_floating_point() else tensor

    return load


def validate_fp8_hf_import(bridge: Any, optimizer: Any = None) -> bool:
    """Require scale-aware loading for Qwen FP8 sources before copying any
    weights."""
    if not is_qwen_fp8_source(bridge):
        return False
    if not getattr(bridge._model_bridge, "supports_fp8_hf_import", False):
        raise RuntimeError(
            "Qwen3.8 FP8 HF checkpoints require the scale-aware Megatron Bridge from the Relax image "
            "on every training node to load the original scales"
        )
    if optimizer is not None:
        for child in getattr(optimizer, "chained_optimizers", (optimizer,)):
            config = child.config
            if (
                not config.use_distributed_optimizer
                or config.main_params_dtype != torch.float32
                or config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
                or getattr(config, "optimizer_cpu_offload", False)
                or getattr(getattr(child, "ddp_config", None), "use_megatron_fsdp", False)
            ):
                raise ValueError(
                    "Qwen3.8 FP8 HF import requires Megatron-managed distributed FP32 masters. "
                    "Use the MXFP8 recipe without optimizer CPU offload, or disable "
                    "--use-precision-aware-optimizer for a pure BF16 reference run. Megatron FSDP is unsupported."
                )
    logger.info("Qwen3.8 FP8 HF import: preserve original scales and FP32 dequantized optimizer initial values")
    return True


@torch.no_grad()
def _store_mxfp8_weights(module: torch.nn.Module) -> None:
    """Convert before DDP/optimizer construction, retaining Megatron
    metadata."""
    from transformer_engine import pytorch as te
    from transformer_engine.common.recipe import Format, MXFP8BlockScaling
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    if getattr(module, "primary_weights_in_fp8", False):
        return
    if module._fp8_workspaces:
        raise RuntimeError("Configure Qwen3.8 FP8 parameter storage before the first forward")
    module.primary_weights_in_fp8 = True
    with te.fp8_model_init(enabled=True, recipe=MXFP8BlockScaling(fp8_format=Format.E4M3)):
        module.init_fp8_metadata(num_gemms=getattr(module, "num_gemms", 1))
    for name, parameter in list(module.named_parameters(recurse=False)):
        if not name.startswith("weight"):
            continue
        quantizer = module.quantizers["scaling_fwd"][module.param_init_meta[name].fp8_meta_index]
        quantizer.set_usage(rowwise=True, columnwise=True)
        quantizer.internal = False
        quantized = torch.nn.Parameter(quantizer(parameter.detach()), requires_grad=parameter.requires_grad)
        if not isinstance(quantized, MXFP8Tensor):
            raise TypeError(f"Expected MXFP8 parameter for {name}, got {type(quantized).__name__}")
        quantized.__dict__.update(parameter.__dict__)
        module.register_parameter(name, quantized)


def configure_fp8(model: torch.nn.Module, args: Namespace) -> dict[str, str]:
    """Configure TE layers before DDP; optionally replace expert weight
    storage."""
    if not getattr(args, "fp8", None):
        return {}
    if not getattr(args, "bf16", False):
        raise ValueError("Qwen3.8 FP8 requires --bf16 for its high-precision modules")
    if getattr(args, "use_transformer_engine_op_fuser", False):
        raise ValueError("Qwen3.8 module precision overrides require the TE Linear/GroupedLinear forward path")
    if getattr(args, "te_precision_config_file", None):
        raise ValueError("Qwen3.8 MoE FP8 cannot be combined with a separate TE precision config")

    recipe = getattr(args.fp8_recipe, "value", args.fp8_recipe)
    primary_fp8 = getattr(args, "fp8_param_gather", False)
    mok_muon = (
        primary_fp8
        and recipe == "mxfp8"
        and getattr(args, "moe_megakernel_backend", None) == "mok"
        and getattr(args, "optimizer", None) == "muon"
        and getattr(args, "use_layer_wise_distributed_optimizer", False)
    )
    if primary_fp8 and (
        recipe != "mxfp8"
        or not (mok_muon or getattr(args, "use_distributed_optimizer", False))
        or not getattr(args, "reuse_grad_buf_for_mxfp8_param_ag", False)
    ):
        raise ValueError(
            "Qwen3.8 FP8 parameters require MXFP8, --use-distributed-optimizer and --reuse-grad-buf-for-mxfp8-param-ag"
        )
    if mok_muon:
        # Native TE projections inherit global MXFP8 and MoK owns its expert weights.
        logger.info("Qwen3.8 MoK uses native MXFP8 weights with layer-wise distributed Muon")
        return {}
    if primary_fp8 and getattr(args, "overlap_param_gather", False):
        raise ValueError("Qwen3.8 FP8 parameter storage currently requires synchronous parameter gather")
    alignment = {"mxfp8": _MXFP8_BLOCK_SIZE, "blockwise": 128, "tensorwise": 16}.get(recipe)
    if alignment is None:
        raise ValueError(f"Qwen3.8 MoE FP8 does not support recipe {recipe!r}")

    from megatron.core.quantization.quant_config import MatchContext, QuantizationConfig
    from transformer_engine import pytorch as te

    plan: dict[str, str] = {}
    pending = []
    for name, module in model.named_modules():
        if not isinstance(module, (te.Linear, te.LayerNormLinear, te.GroupedLinear)):
            continue
        if not callable(getattr(module, "finish_init", None)):
            raise RuntimeError(f"Qwen3.8 FP8 requires MCore per-module precision support for {name}")
        parts = name.split(".")
        expert = (
            bool({"experts", "shared_experts"}.intersection(parts))
            and parts[-1] in ("linear_fc1", "linear_fc2")
            and "mtp" not in parts
            and "vision_model" not in parts
        )
        if expert:
            if primary_fp8 and getattr(module, "single_grouped_weight", False):
                raise ValueError("Qwen3.8 FP8 parameter storage requires individual expert weights")
            for param_name, param in module.named_parameters(recurse=False):
                if param_name.startswith("weight") and param.ndim >= 2:
                    if primary_fp8 and (param.ndim != 2 or hasattr(param, "main_grad")):
                        raise ValueError(f"Convert {name}.{param_name} to MXFP8 before DDP, with a 2D weight")
                    if any(size % alignment for size in param.shape[-2:]):
                        raise ValueError(
                            f"{name}.{param_name} shape {tuple(param.shape)} is not aligned to {alignment} for {recipe}"
                        )
            quant_config = None  # Inherit the global FP8 context and its recompute state.
        else:
            quant_config = QuantizationConfig(
                {
                    "transformer_engine_config_type": "TEQuantizationParams",
                    "training_recipe": {},
                },
                match_input=MatchContext(module_path=name, layer_number=None),
                config_key="qwen38_bf16",
            )
        plan[name] = recipe if expert else "bf16"
        # Overlapped shared experts invoke FC1/FC2 directly. At TP1 both preserve token rows.
        pad_shared = (
            expert
            and "shared_experts" in parts
            and recipe == "mxfp8"
            and isinstance(module, te.Linear)
            and module.tp_size == 1
        )
        pending.append((module, quant_config, expert, parts[-1] if pad_shared else None))

    # Validate the whole stage before applying any overrides.
    for module, quant_config, expert, shared_projection in pending:
        module.finish_init(quant_config)
        if primary_fp8 and expert:
            _store_mxfp8_weights(module)
        if shared_projection and getattr(module.forward, "func", None) is not _mxfp8_shared_linear:
            fc1 = module if shared_projection == "linear_fc1" else None
            module.forward = partial(_mxfp8_shared_linear, module.forward, fc1=fc1)
    logger.info(
        "Qwen3.8 precision: %d backbone expert GEMMs use %s; %d TE modules retain BF16; expert parameters use %s storage",
        sum(value == recipe for value in plan.values()),
        recipe,
        sum(value == "bf16" for value in plan.values()),
        "MXFP8" if primary_fp8 else "BF16",
    )
    logger.debug("Qwen3.8 module precision: %s", plan)
    return plan
