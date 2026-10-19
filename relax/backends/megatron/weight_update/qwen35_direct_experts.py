# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Native BF16 expert placement for the validated Qwen3.5 TP2 Triton layout."""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch


NamedTensor = tuple[str, torch.Tensor]


_PACKED = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)")


def validate_direct_receiver(config: Mapping[str, Any]) -> None:
    required = dict(
        tp_size=2,
        ep_size=1,
        pp_size=1,
        moe_dp_size=1,
        moe_runner_backend="triton",
        disable_shared_experts_fusion=True,
        dtype="bfloat16",
    )
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Direct Qwen3.5 experts require explicit BF16 Triton, TP2/EP1/PP1 and unfused shared experts")
    if (
        config.get("enable_eplb")
        or config.get("ep_num_redundant_experts", 0)
        or config.get("init_expert_location") not in (None, "trivial")
        or config.get("ep_join_mode") is not None
        or config.get("quantization") is not None
        or config.get("enable_lora")
        or config.get("speculative_algorithm") is not None
        or config.get("enforce_shared_experts_fusion")
        or json.loads(config.get("json_model_override_args") or "{}")
    ):
        raise ValueError("Direct Qwen3.5 experts do not support quantization, remapping, LoRA or model overrides")


def _shard_expert(tensor: torch.Tensor, projection: str, tp_rank: int) -> torch.Tensor:
    if tp_rank not in (0, 1) or tensor.ndim != 3 or tensor.dtype != torch.bfloat16:
        raise ValueError("Direct expert shards require BF16 rank-3 tensors and TP rank 0/1")
    if projection == "gate_up_proj":
        if tensor.shape[1] % 4:
            raise ValueError("Gate/up rows must be divisible by four for TP2")
        intermediate = tensor.shape[1] // 2
        width = intermediate // 2
        start = tp_rank * width
        # TP slices gate and up separately, then keeps [gate_local, up_local].
        return torch.cat(
            (tensor[:, start : start + width], tensor[:, intermediate + start : intermediate + start + width]), dim=1
        )
    if projection == "down_proj":
        if tensor.shape[2] % 2:
            raise ValueError("Down columns must be divisible by two for TP2")
        width = tensor.shape[2] // 2
        return tensor[:, :, tp_rank * width : (tp_rank + 1) * width].contiguous()
    raise ValueError(f"Unsupported expert projection: {projection}")


def direct_expert_chunk(named_tensors: Sequence[NamedTensor], tp_rank: int) -> list[NamedTensor] | None:
    if len(named_tensors) != 1:
        return None
    name, tensor = named_tensors[0]
    match = _PACKED.fullmatch(name)
    if match is None:
        return None
    layer, projection = match.groups()
    expected = (256, 1024, 2048) if projection == "gate_up_proj" else (256, 2048, 512)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"Unsupported Qwen3.5 expert shape: {name}: {tuple(tensor.shape)}")
    target = "w13_weight" if projection == "gate_up_proj" else "w2_weight"
    return [(f"model.layers.{layer}.mlp.experts.{target}", _shard_expert(tensor, projection, tp_rank))]


def dispatch_typed_buckets(
    packets: Sequence[Sequence[tuple[str, str]]],
    *,
    engine: Any,
    version: str | int | None,
    serializer: Any,
    empty_flattened: Callable[[], dict[str, Any]],
    held: list[Any],
    dispatch_errors: list[str] | None = None,
) -> list[Any]:
    """Keep a single request format even with empty or differently packed
    ranks.

    Only enabled by the private direct-expert path. Format tags ride in the
    existing Gloo gather; no new collective is needed. GPU payloads are already
    held by their producers through the caller's normal ACK pipeline.
    """
    refs, empty = [], {}
    count = max((len(packet) for packet in packets), default=0)
    for index in range(count):
        formats = dict.fromkeys(packet[index][0] for packet in packets if index < len(packet))
        for load_format in formats:
            if load_format not in ("direct", "flattened_bucket"):
                raise ValueError(f"Unsupported IPC format: {load_format}")
            serialized = []
            for packet in packets:
                if index < len(packet) and packet[index][0] == load_format:
                    serialized.append(packet[index][1])
                else:
                    if load_format not in empty:
                        payload = [] if load_format == "direct" else empty_flattened()
                        held.append(payload)
                        empty[load_format] = serializer.serialize(payload, output_str=True)
                    serialized.append(empty[load_format])
            kwargs = dict(serialized_named_tensors=serialized, load_format=load_format)
            if version is not None:
                kwargs["weight_version"] = str(version)
            try:
                refs.append(engine.update_weights_from_tensor.remote(**kwargs))
            except Exception as error:
                if dispatch_errors is None:
                    raise
                dispatch_errors.append(str(error))
                return refs
    return refs
