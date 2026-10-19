# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""BF16 gated-expert layouts shared by Bridge export and SGLang FusedMoE.

Names and tensor orientation come from Bridge's canonical per-expert output.
This module only packs and shards that representation; it never converts raw
Megatron weights or keeps process groups/GPU storage across updates.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


NamedTensor = tuple[str, torch.Tensor]
_PACKED = re.compile(r"^(.*\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)$")


@dataclass(frozen=True)
class DirectExpertLayout:
    num_experts: int
    hidden_size: int
    intermediate_size: int
    tp_size: int

    @classmethod
    def from_receiver(cls, config: Mapping[str, Any]) -> "DirectExpertLayout":
        required = dict(
            ep_size=1,
            pp_size=1,
            moe_dp_size=1,
            moe_runner_backend="triton",
            disable_shared_experts_fusion=True,
            dtype="bfloat16",
        )
        if any(config.get(key) != value for key, value in required.items()):
            raise ValueError("Direct experts require BF16 Triton, EP1/PP1/MoE-DP1 and unfused shared experts")
        if (
            config.get("enable_eplb")
            or config.get("ep_num_redundant_experts", 0)
            or config.get("init_expert_location") not in (None, "trivial")
            or config.get("ep_join_mode") is not None
            or config.get("quantization") is not None
            or config.get("quantization_config")
            or config.get("enable_lora")
            or config.get("kt_weight_path")
            or config.get("speculative_algorithm") is not None
            or config.get("enforce_shared_experts_fusion")
            or json.loads(config.get("json_model_override_args") or "{}")
        ):
            raise ValueError("Direct experts do not support quantization, remapping, LoRA or model overrides")
        dimensions = tuple(
            config.get(key) for key in ("num_experts", "hidden_size", "moe_intermediate_size", "tp_size")
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("Direct expert geometry must be explicitly specified and positive")
        layout = cls(*dimensions)
        if layout.intermediate_size % layout.tp_size:
            raise ValueError("Expert intermediate size must be divisible by receiver TP size")
        return layout

    def shard(self, tensor: torch.Tensor, projection: str, tp_rank: int) -> torch.Tensor:
        if not 0 <= tp_rank < self.tp_size or tensor.dtype != torch.bfloat16:
            raise ValueError("Invalid direct expert TP rank or dtype")
        e, h, i = self.num_experts, self.hidden_size, self.intermediate_size
        expected = (e, 2 * i, h) if projection == "gate_up_proj" else (e, h, i)
        if tuple(tensor.shape) != expected or projection not in ("gate_up_proj", "down_proj"):
            raise ValueError(f"Unsupported expert projection or shape: {projection}: {tuple(tensor.shape)}")
        width = i // self.tp_size
        start = tp_rank * width
        if projection == "gate_up_proj":
            return torch.cat((tensor[:, start : start + width], tensor[:, i + start : i + start + width]), dim=1)
        return tensor[:, :, start : start + width].contiguous()


def direct_expert_chunk(
    named_tensors: Sequence[NamedTensor], layout: DirectExpertLayout, tp_rank: int
) -> list[NamedTensor] | None:
    if len(named_tensors) != 1:
        return None
    name, tensor = named_tensors[0]
    match = _PACKED.fullmatch(name)
    if match is None:
        return None
    prefix, projection = match.groups()
    # The multimodal HF namespace is stripped by the Qwen3.5 SGLang loader.
    # Standard text MoE namespaces (including DeepSeek V2/V3) are unchanged.
    prefix = prefix.replace("model.language_model.layers.", "model.layers.", 1)
    if not re.fullmatch(r"model\.layers\.\d+\.mlp\.experts", prefix):
        raise ValueError(f"Unsupported native expert namespace: {prefix}")
    target = "w13_weight" if projection == "gate_up_proj" else "w2_weight"
    return [(f"{prefix}.{target}", layout.shard(tensor, projection, tp_rank))]
