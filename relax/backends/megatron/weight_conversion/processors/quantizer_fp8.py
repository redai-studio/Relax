# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import re
from typing import Any, Optional, Sequence

import torch

from relax.backends.megatron.kernels.fp8_kernel import blockwise_cast_to_fp8_triton

from ...sglang import quant_weight_ue8m0, should_deepgemm_weight_requant_ue8m0, transform_scale_ue8m0


_BF16_SENDER_PRESERVE_UE8M0_GRID_ENV = "RELAX_FP8_BF16_SENDER_PRESERVE_UE8M0_GRID"


def _resolve_online_weight_scale_fmt(
    args: Any, configured_scale_fmt: Optional[str], weight_block_size: Optional[Sequence[int]]
) -> Optional[str]:
    """Resolve the online sender's weight scale grid from the trainer
    policy."""
    if configured_scale_fmt != "ue8m0":
        return configured_scale_fmt

    # BF16 training has no trainer-side FP8 weight grid. Preserve the public
    # pre-5392 behavior by default: ordinary FP32 scales on Hopper, while
    # _quantize_param can still select packed UE8M0 when the runtime requires it.
    # Reproducing the checkpoint's UE8M0 grid is an explicit DSV4-style opt-in.
    if getattr(args, "fp8", None) is None:
        preserve_ue8m0_grid = os.environ.get(_BF16_SENDER_PRESERVE_UE8M0_GRID_ENV, "0")
        if preserve_ue8m0_grid not in ("0", "1"):
            raise ValueError(f"{_BF16_SENDER_PRESERVE_UE8M0_GRID_ENV} must be 0 or 1, got {preserve_ue8m0_grid!r}")
        return configured_scale_fmt if preserve_ue8m0_grid == "1" else None

    fp8_recipe = getattr(args, "fp8_recipe", None)
    fp8_recipe = getattr(fp8_recipe, "value", fp8_recipe)
    use_fp32_scales = os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0") == "1"
    if fp8_recipe != "blockwise" or not use_fp32_scales:
        return configured_scale_fmt

    if list(weight_block_size or []) != [128, 128]:
        raise RuntimeError(
            "stock FP8 blockwise training with FP32 scales requires a 128x128 online weight block, "
            f"got {weight_block_size!r}"
        )
    if should_deepgemm_weight_requant_ue8m0 is None:
        raise RuntimeError(
            "cannot validate ordinary FP32 online weight scales because SGLang's "
            "UE8M0 runtime capability check is unavailable"
        )
    pack_ue8m0 = should_deepgemm_weight_requant_ue8m0(weight_block_size=weight_block_size)
    if pack_ue8m0:
        raise RuntimeError(
            "stock FP8 blockwise training uses ordinary FP32 weight scales, but the rollout runtime "
            "requires packed UE8M0 scales"
        )

    # ``None`` selects the existing amax/448 FP32-scale quantizer below. Keep the
    # checkpoint's configured scale format untouched for SGLang activation paths.
    return None


def quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config):
    assert quantization_config["quant_method"] == "fp8"
    fmt = quantization_config.get("fmt", "e4m3")
    assert fmt == "e4m3", f"Unsupported FP8 format: {fmt}"
    assert quantization_config["activation_scheme"] == "dynamic"
    weight_block_size = quantization_config.get("weight_block_size", None)
    scale_fmt = quantization_config.get("scale_fmt", None)
    assert scale_fmt in (None, "ue8m0"), f"Unsupported FP8 scale format: {scale_fmt}"
    scale_fmt = _resolve_online_weight_scale_fmt(args, scale_fmt, weight_block_size)

    modules_to_not_convert = quantization_config.get("modules_to_not_convert", None)
    if modules_to_not_convert:
        return _quantize_params_fp8_by_ignore_list(
            converted_named_params, set(modules_to_not_convert), weight_block_size, scale_fmt
        )

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, megatron_name)

    if not match:
        # check mtp layers
        mtp_layer_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
        match = re.match(mtp_layer_pattern, megatron_name)
        if not match:
            return converted_named_params
        layer_idx, rest = match.groups()
        rest = rest.replace("transformer_layer.", "")
    else:
        layer_idx, rest = match.groups()

    # experts
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, expert_idx = match.groups()
        if rest in [
            "linear_fc1",
            "linear_fc2",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                # skip bf16 weight_scale and input_scale
                # TODO: find a clearer way.
                if converted_name.endswith("_scale"):
                    continue
                quantize_named_params.extend(_quantize_param(converted_name, param, weight_block_size, scale_fmt))

            return quantize_named_params

    # shared expert
    shared_expert_pattern = r"mlp.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.groups()[0]
        if rest in [
            "linear_fc1.weight",
            "linear_fc2.weight",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                quantize_named_params.extend(_quantize_param(converted_name, param, weight_block_size, scale_fmt))

            return quantize_named_params

    if rest in [
        "self_attention.linear_proj.weight",
        "self_attention.linear_qkv.weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
        # mla
        "self_attention.linear_q_proj.weight",
        "self_attention.linear_q_down_proj.weight",
        "self_attention.linear_q_up_proj.weight",
        "self_attention.linear_kv_down_proj.weight",
        "self_attention.linear_kv_up_proj.weight",
        # indexer
        "self_attention.wq_b.weight",
        "self_attention.wk.weight",
        # DeepSeek-V4 (experimental_attention_variant="dsv4_hybrid"). Both names are
        # created only by transformer/experimental_attention_variant/
        # deepseek_v4_hybrid_attention.py, so they cannot collide with the V3-style
        # MLA names above (which split KV into down/up) or with GLM5's DSA indexer
        # (self_attention.wq_b, already listed).
        #   linear_kv_proj -> attn.wkv, one fused KV down-projection
        #   indexer.linear_wq_b -> attn.indexer.wq_b, the only FP8 tensor in the indexer
        # Deliberately absent, because DeepSeek-V4-Flash keeps them BF16/F32 on disk
        # and quantizing them would silently corrupt the pushed weights:
        #   linear_o_group_proj (attn.wo_a) - a bare nn.Parameter with no ".weight"
        #       suffix, so it matches nothing here; its partner linear_proj
        #       (attn.wo_b) IS fp8 and is already covered above,
        #   core_attention[.indexer].compressor.linear_wkv / linear_wgate,
        #   *_hyper_connection.mapping_proj.weight (fp32 hc_*_fn).
        "self_attention.linear_kv_proj.weight",
        "self_attention.core_attention.indexer.linear_wq_b.weight",
    ]:
        quantize_named_params = []
        for converted_name, param in converted_named_params:
            quantize_named_params.extend(_quantize_param(converted_name, param, weight_block_size, scale_fmt))

        return quantize_named_params

    # for other parameters, we just return the original converted_named_params
    return converted_named_params


def _checkpoint_module_name(hf_name):
    module = hf_name[: -len(".weight")] if hf_name.endswith(".weight") else hf_name
    match = re.match(r"(.*\.experts)\.\d+\.(gate_proj|up_proj|down_proj)$", module)
    if match:
        base, proj = match.groups()
        fused = "down_proj" if proj == "down_proj" else "gate_up_proj"
        return f"{base}.{fused}"
    return module


def _quantize_params_fp8_by_ignore_list(converted_named_params, modules_to_not_convert, weight_block_size, scale_fmt):
    quantize_named_params = []
    for name, param in converted_named_params:
        is_quantizable = (
            name.endswith(".weight")
            and param.dim() == 2
            and param.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and _checkpoint_module_name(name) not in modules_to_not_convert
        )
        if is_quantizable:
            quantize_named_params.extend(_quantize_param(name, param, weight_block_size, scale_fmt))
        else:
            quantize_named_params.append((name, param))
    return quantize_named_params


def _quantize_param(name, weight, weight_block_size, scale_fmt=None):
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"
    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    if weight_block_size is not None:
        pack_ue8m0 = should_deepgemm_weight_requant_ue8m0 and should_deepgemm_weight_requant_ue8m0(
            weight_block_size=weight_block_size
        )
        if scale_fmt == "ue8m0" or pack_ue8m0:
            if quant_weight_ue8m0 is None:
                raise RuntimeError("SGLang's UE8M0 FP8 quantizer is unavailable")
            qweight, scale = quant_weight_ue8m0(weight, weight_block_size=weight_block_size)
            if pack_ue8m0:
                scale = transform_scale_ue8m0(scale, mn=qweight.shape[-2])
        else:
            qweight, scale = blockwise_cast_to_fp8_triton(weight, weight_block_size)
        scale_name = name.replace(".weight", ".weight_scale_inv")
    else:
        # per tensor quant
        scale = weight.abs().max().clamp(min=1e-12).to(torch.float32) / FP8_MAX
        qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX).to(torch.float8_e4m3fn)
        scale = scale.view(1)
        scale_name = name.replace(".weight", ".weight_scale")
    return [(name, qweight), (scale_name, scale)]
