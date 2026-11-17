# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared FP8 tensor quantization and HuggingFace metadata helpers."""

import torch
import torch.nn.functional as F


FP8_INFO = torch.finfo(torch.float8_e4m3fn)
FP8_MAX, FP8_MIN = FP8_INFO.max, FP8_INFO.min


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def block_fp8(weight: torch.Tensor, block_size: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    block_n, block_k = block_size[0], block_size[1]

    shape_0, shape_1 = weight.shape

    n_tiles = ceildiv(shape_0, block_n)
    k_tiles = ceildiv(shape_1, block_k)

    q_weight = F.pad(
        weight,
        (0, k_tiles * block_k - shape_1, 0, n_tiles * block_n - shape_0),
        mode="constant",
        value=0.0,
    )

    qweight = q_weight.reshape(n_tiles, block_n, k_tiles, block_k)
    block_max = torch.max(torch.abs(qweight), dim=1, keepdim=True)[0]
    block_max = torch.max(block_max, dim=3, keepdim=True)[0]

    scale = block_max.to(torch.float32).clamp(min=1e-12) / FP8_MAX
    qweight = (
        (qweight / scale)
        .clamp(min=FP8_MIN, max=FP8_MAX)
        .reshape((n_tiles * block_n, k_tiles * block_k))
        .to(torch.float8_e4m3fn)
    )
    qweight = qweight[:shape_0, :shape_1].clone().detach()
    scale = scale.reshape(n_tiles, k_tiles)

    return qweight, scale


def channel_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channel_max = torch.max(weight.abs(), dim=-1, keepdim=True)[0]
    scale = channel_max.to(torch.float32).clamp(min=1e-12) / FP8_MAX
    qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX)
    qweight = qweight.to(torch.float8_e4m3fn)
    return qweight, scale


def tensor_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.abs().max().to(torch.float32).clamp(min=1e-12) / FP8_MAX
    qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX)
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.view(1)
    return qweight, scale


def quant_fp8(
    weight: torch.Tensor,
    strategy: str,
    block_size: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if strategy == "tensor":
        return tensor_fp8(weight)
    elif strategy == "channel":
        return channel_fp8(weight)
    elif strategy == "block":
        return block_fp8(weight, block_size)
    raise ValueError(f"Unsupported FP8 strategy: {strategy}")


def validate_fp8_options(strategy: str, block_size: list[int] | None) -> None:
    if strategy not in {"block", "channel", "tensor"}:
        raise ValueError(f"Unsupported FP8 strategy: {strategy}")
    if strategy == "block":
        if block_size is None or len(block_size) != 2 or any(size <= 0 for size in block_size):
            raise ValueError("block FP8 requires exactly two positive --block-size values")
    elif block_size is not None:
        raise ValueError(f"--block-size is only valid with strategy=block, got strategy={strategy}")


def _scale_name(weight_name: str, strategy: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"Expected a weight name ending in '.weight', got {weight_name}")
    suffix = ".weight_scale_inv" if strategy == "block" else ".weight_scale"
    return f"{weight_name[: -len('.weight')]}{suffix}"


def _quantize_on_device(
    weight: torch.Tensor,
    strategy: str,
    block_size: list[int] | None,
    device: str | torch.device | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        output_device = weight.device
        if device is not None:
            weight = weight.to(device)
        qweight, scale = quant_fp8(weight, strategy, block_size)
        qweight = qweight.detach()
        scale = scale.detach()
        if qweight.device != output_device:
            qweight = qweight.to(output_device)
            scale = scale.to(output_device)
    return qweight, scale


def _store_quantized_fp8(
    q_weights: dict[str, torch.Tensor],
    base_name: str,
    weight: torch.Tensor,
    strategy: str,
    block_size: list[int] | None,
    device: str | torch.device | None = None,
) -> None:
    weight_name = f"{base_name}.weight"
    qw, s = _quantize_on_device(weight, strategy, block_size, device)
    q_weights[weight_name] = qw
    q_weights[_scale_name(weight_name, strategy)] = s


_SKIPPED_WEIGHT_PATTERNS = (
    "layernorm",
    "embed",
    "router",
    "mlp.gate.",
    "norm",
    "lm_head",
    "eh_proj",
    "weights_proj",
    "conv1d",
    "A_log",
    "dt_bias",
    "in_proj_a",
    "in_proj_b",
    "shared_expert_gate",
    "visual",
)


def _is_quantizable_weight(name: str, weight: torch.Tensor) -> bool:
    return (
        name.endswith(".weight")
        and weight.dim() == 2
        and weight.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and not any(pattern in name for pattern in _SKIPPED_WEIGHT_PATTERNS)
    )


def quantize_hf_tensor(
    name: str,
    weight: torch.Tensor,
    strategy: str,
    block_size: list[int] | None = None,
    device: str | torch.device | None = None,
    *,
    expert_only: bool = False,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Quantize one HF tensor and return its output tensors plus ignored
    modules."""
    q_weights: dict[str, torch.Tensor] = {}

    # Qwen3.8 exports follow the training precision boundary, including when
    # MTP experts use the packed representation handled below.
    backbone_expert = name.startswith(("model.layers.", "model.language_model.layers.")) and any(
        part in name for part in (".mlp.experts.", ".mlp.shared_expert.")
    )
    if expert_only and not backbone_expert:
        if name.startswith("mtp."):
            ignored = ["mtp"]
        elif name.startswith("model.visual."):
            ignored = ["model.visual"]
        elif ".ple.ple_embedding.ngram_embedding.shard_" in name:
            # The inference loader selects precision on the parent table.
            ignored = [name.rsplit(".shard_", 1)[0]]
        else:
            ignored = [name[: -len(".weight")]] if name.endswith(".weight") else []
        return {name: weight}, ignored

    is_fused_experts = name.endswith(".experts.gate_up_proj") or name.endswith(".experts.down_proj")
    if weight.dim() == 3 and is_fused_experts:
        base = name.rsplit(".", 1)[0]
        for expert_idx in range(weight.shape[0]):
            expert = weight[expert_idx]
            if name.endswith(".gate_up_proj"):
                gate_w, up_w = expert.chunk(2, dim=0)
                _store_quantized_fp8(
                    q_weights,
                    f"{base}.{expert_idx}.gate_proj",
                    gate_w.contiguous(),
                    strategy,
                    block_size,
                    device,
                )
                _store_quantized_fp8(
                    q_weights,
                    f"{base}.{expert_idx}.up_proj",
                    up_w.contiguous(),
                    strategy,
                    block_size,
                    device,
                )
            else:
                _store_quantized_fp8(
                    q_weights,
                    f"{base}.{expert_idx}.down_proj",
                    expert.contiguous(),
                    strategy,
                    block_size,
                    device,
                )
        return q_weights, []

    if _is_quantizable_weight(name, weight):
        qweight, scale = _quantize_on_device(weight, strategy, block_size, device)
        return {name: qweight, _scale_name(name, strategy): scale}, []

    module_names = [name[: -len(".weight")]] if name.endswith(".weight") else []
    return {name: weight}, module_names


def build_quantization_config(
    strategy: str,
    block_size: list[int] | None,
    modules_to_not_convert: list[str],
    scale_fmt: str | None = None,
) -> dict:
    ignored_modules = sorted(set(modules_to_not_convert))
    if strategy in {"block", "tensor"}:
        quantization_config: dict = {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
        }
        if block_size:
            quantization_config["weight_block_size"] = block_size
            if scale_fmt is not None:
                quantization_config["scale_fmt"] = scale_fmt
        if ignored_modules:
            quantization_config["modules_to_not_convert"] = ignored_modules
        return quantization_config

    quant_group = {
        "group_0": {
            "input_activations": {
                "actorder": None,
                "block_structure": None,
                "dynamic": True,
                "group_size": None,
                "num_bits": 8,
                "observer": None,
                "observer_kwargs": {},
                "strategy": "token",
                "symmetric": True,
                "type": "float",
            },
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": None,
                "num_bits": 8,
                "observer": "minmax",
                "observer_kwargs": {},
                "strategy": strategy,
                "symmetric": True,
                "type": "float",
            },
        },
    }
    return {
        "config_groups": quant_group,
        "format": "float-quantized",
        "ignore": ignored_modules,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }
