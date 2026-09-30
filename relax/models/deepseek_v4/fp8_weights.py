# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Optional native FP8 weight-grid alignment for DSv4's TE linear subset.

This composes with routed MXFP4 QAT without modifying its hooks. Only weight
quantizers use power-of-two scales; activation/gradient recipes stay unchanged.
The native ``wo_a`` Parameter/einsum path is deliberately reported as
uncovered. Only BF16 model copies are supported: Bridge's float32 log2 can
round an FP32 amax across a power-of-two boundary differently from TE's
reciprocal bit mask.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable


if TYPE_CHECKING:
    import torch


# Actual DSv4 Bridge mappings at TP=1: no row/column permutation. The shared
# gate/up concatenation is row-wise; its boundary must align to 128-row tiles.
_MAPPINGS = (
    ("self_attention.linear_q_down_proj", ("attn.wq_a",)),
    ("self_attention.linear_q_up_proj", ("attn.wq_b",)),
    ("self_attention.linear_kv_proj", ("attn.wkv",)),
    ("self_attention.linear_proj", ("attn.wo_b",)),
    ("mlp.shared_experts.linear_fc1", ("ffn.shared_experts.w1", "ffn.shared_experts.w3")),
    ("mlp.shared_experts.linear_fc2", ("ffn.shared_experts.w2",)),
    ("self_attention.core_attention.indexer.linear_wq_b", ("attn.indexer.wq_b",)),
)


@dataclass
class _WeightState:
    original_workspace: Callable[..., Any]
    module_name: str


def _power2_workspace(
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

    state: _WeightState = self._relax_native_fp8_weights
    if fsdp_group is not None:
        raise RuntimeError("Native FP8 weight-grid alignment does not support FSDP workspaces")
    if (
        not isinstance(quantizer, Float8BlockQuantizer)
        or quantizer.block_scaling_dim != 2
        or str(quantizer.dtype).split(".")[-1] != "kFloat8E4M3"
        or quantizer.amax_epsilon != 0.0
    ):
        raise TypeError(f"{state.module_name}: expected E4M3 128x128 block quantizer with zero amax epsilon")
    if tensor is not None and (
        type(tensor) not in (torch.Tensor, torch.nn.Parameter)
        or tensor.dtype != torch.bfloat16
        or tensor.ndim != 2
        or not tensor.is_contiguous()
        or not quantizer.is_quantizable(tensor)
    ):
        raise TypeError(f"{state.module_name}: expected contiguous BF16 model weight, dimensions divisible by 128")

    weight_quantizer = quantizer.copy()
    weight_quantizer.force_pow_2_scales = True
    out = self._fp8_workspaces.get(cache_name) if cache_name is not None else None
    if out is not None:
        if not isinstance(out, Float8BlockwiseQTensorStorage) or not out._quantizer.force_pow_2_scales:
            raise RuntimeError(f"{state.module_name}: incompatible cached weight workspace")
        missing_usage = (weight_quantizer.rowwise_usage and out._rowwise_data is None) or (
            weight_quantizer.columnwise_usage and out._columnwise_data is None
        )
        if missing_usage or (tensor is not None and tuple(out.size()) != tuple(tensor.shape)):
            del self._fp8_workspaces[cache_name]
            out = None
        else:
            # TE updates through out.quantize_ and its stored quantizer.
            cached_quantizer = weight_quantizer.copy()
            cached_quantizer.internal = out._quantizer.internal
            out._quantizer = cached_quantizer
    return state.original_workspace(
        tensor=tensor,
        quantizer=weight_quantizer,
        cache_name=cache_name,
        update_workspace=update_workspace,
        skip_update_flag=skip_update_flag,
        fsdp_group=fsdp_group,
        workspace_dtype=workspace_dtype,
    )


def _install_workspace(module: torch.nn.Module, *, module_name: str) -> None:
    from transformer_engine.pytorch import LayerNormLinear, Linear

    if not isinstance(module, (Linear, LayerNormLinear)):
        raise TypeError(f"{module_name}: native FP8 weight alignment supports TE Linear/LayerNormLinear only")
    if hasattr(module, "_relax_mxfp4_qat"):
        raise ValueError(f"{module_name}: refusing to overwrite routed MXFP4 QAT")
    if hasattr(module, "_relax_native_fp8_weights"):
        return
    if getattr(module, "primary_weights_in_fp8", False) or getattr(module, "te_quant_params", None) is not None:
        raise ValueError(f"{module_name}: primary FP8 weights and custom module recipes are unsupported")
    if module._fp8_workspaces:
        raise RuntimeError(f"{module_name}: install before the first forward, while the weight cache is empty")
    module._relax_native_fp8_weights = _WeightState(module.get_weight_workspace, module_name)
    module.get_weight_workspace = MethodType(_power2_workspace, module)


def install_native_fp8_weights(model: torch.nn.Module, *, hf_checkpoint: str | Path) -> tuple[str, ...]:
    """Validate the source schema once and install the native TE subset.

    Header reads occur only at installation; no checkpoint tensors are loaded.
    Both the original release and its expert-only FP8 conversion retain these
    native dense/shared E4M3/E8M0 tensors. A different schema is rejected.
    """
    root = Path(hf_checkpoint)
    hf_config = json.loads((root / "config.json").read_text())
    quant_config = hf_config.get("quantization_config", {})
    if (
        hf_config.get("model_type") != "deepseek_v4"
        or quant_config.get("fmt") != "e4m3"
        or quant_config.get("weight_block_size") != [128, 128]
        or quant_config.get("scale_fmt") != "ue8m0"
    ):
        raise ValueError("Native FP8 alignment requires the DSv4 E4M3/E8M0 128x128 reference schema")
    if model.config.experimental_attention_variant != "dsv4_hybrid" or model.config.tensor_model_parallel_size != 1:
        raise ValueError("Native FP8 mapping is validated only for DSv4 hybrid with TP=1")
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    headers: dict[str, dict[str, Any]] = {}

    def metadata(key: str) -> dict[str, Any]:
        shard = index[key]
        if shard not in headers:
            with (root / shard).open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                if size > 64 * 1024 * 1024:
                    raise ValueError(f"Unexpectedly large safetensors header: {shard}")
                headers[shard] = json.loads(stream.read(size))
        return headers[shard][key]

    plan = []
    uncovered = []
    for local_id, layer in enumerate(model.decoder.layers):
        global_id = layer.layer_number - 1
        uncovered.append(f"layers.{global_id}.attn.wo_a.weight (Parameter/einsum, no TE workspace)")
        for module_path, hf_paths in _MAPPINGS:
            if ".indexer." in module_path and getattr(layer.self_attention.core_attention, "indexer", None) is None:
                continue
            module = layer.get_submodule(module_path)
            shapes = []
            for hf_path in hf_paths:
                prefix = f"layers.{global_id}.{hf_path}"
                weight, scale = metadata(prefix + ".weight"), metadata(prefix + ".scale")
                shape = weight["shape"]
                if (
                    weight["dtype"] != "F8_E4M3"
                    or scale["dtype"] != "F8_E8M0"
                    or len(shape) != 2
                    or any(d % 128 for d in shape)
                    or scale["shape"] != [shape[0] // 128, shape[1] // 128]
                ):
                    raise ValueError(f"{prefix}: unsupported native FP8 dtype or block geometry")
                shapes.append(shape)
            if any(shape[1] != shapes[0][1] for shape in shapes):
                raise ValueError(f"{module_path}: incompatible gate/up concatenation")
            expected = (sum(shape[0] for shape in shapes), shapes[0][1])
            if tuple(module.weight.shape) != expected:
                raise ValueError(f"{module_path}: model shape {tuple(module.weight.shape)} != source shape {expected}")
            plan.append((f"decoder.layers.{local_id}.{module_path}", module))
    if not plan:
        raise ValueError("Native FP8 alignment found no supported TE weights on this model stage")
    for name, module in plan:
        _install_workspace(module, module_name=name)
    model._relax_native_fp8_uncovered = tuple(uncovered)
    return tuple(name for name, _ in plan)
