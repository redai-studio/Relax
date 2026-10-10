# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Checkpoint inspection and per-weight LoRA merge without constructing K3."""

from __future__ import annotations

import io
import json
import math
import re
import struct
from pathlib import Path
from typing import Any


def read_common_state(checkpoint: str, metadata: Any) -> dict[str, Any]:
    """Read only the small common-state storage record, never optimizer
    tensors."""
    import torch

    records = [(idx, item) for idx, item in metadata.storage_data.items() if idx.fqn.startswith("common_state/")]
    if len(records) != 1:
        raise ValueError(f"Expected one common-state record, found {len(records)}")
    _, item = records[0]
    if item.length > 64 * 1024 * 1024:
        raise ValueError("Common state exceeds 64 MiB; refusing an unexpected checkpoint layout")
    root = Path(checkpoint).resolve()
    source = (root / item.relative_path).resolve()
    if not source.is_relative_to(root):
        raise ValueError("Checkpoint storage path escapes checkpoint directory")
    with source.open("rb") as handle:
        handle.seek(item.offset)
        value = torch.load(io.BytesIO(handle.read(item.length)), weights_only=False, map_location="cpu")
    if isinstance(value, io.BytesIO):
        value.seek(0)
        value = torch.load(value, weights_only=False, map_location="cpu")
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict) or "args" not in value:
        raise ValueError("Common state does not contain training args")
    return value


def build_mappings(
    tensors: dict[str, dict[str, Any]], registry: Any, weight_map: dict[str, str], rank: int
) -> dict[str, dict[str, Any]]:
    """Resolve every adapter through K3 Bridge's own mapping registry.

    DCP full tensors already put all gate rows before all up rows. The save
    factories undo live TP [gate_shard, up_shard] interleaving. Therefore no
    training-TP permutation is applied after loading full tensors via DCP.
    """
    pairs: dict[str, dict[str, str]] = {}
    for key in tensors:
        match = re.fullmatch(r"(language_model\..+)\.adapter\.linear_(in|out)\.weight", key)
        if match is None:
            raise ValueError(f"Unsupported saved model tensor: {key}")
        pairs.setdefault(match[1], {})[match[2]] = key
    mappings: dict[str, dict[str, Any]] = {}
    for prefix, pair in sorted(pairs.items()):
        if set(pair) != {"in", "out"}:
            raise ValueError(f"Incomplete adapter pair: {prefix}")
        a_shape, b_shape = (tensors[pair[side]]["shape"] for side in ("in", "out"))
        if len(a_shape) not in (2, 3) or len(a_shape) != len(b_shape):
            raise ValueError(f"Unsupported adapter geometry: {prefix}: {a_shape}, {b_shape}")
        if a_shape[-2] != rank or b_shape[-1] != rank:
            raise ValueError(f"Adapter rank differs from saved lora_rank: {prefix}")
        expert = len(a_shape) == 3
        model_prefix = prefix.replace(".mlp.experts.experts.", ".mlp.experts.")
        if expert and (a_shape[0] != b_shape[0] or ".mlp.experts.linear_fc" not in model_prefix):
            raise ValueError(f"Unsupported grouped adapter: {prefix}")
        target = model_prefix.removeprefix("language_model.") + (".weight0" if expert else ".weight")
        mapping = registry.megatron_to_hf_lookup(target)
        if mapping is None:
            raise ValueError(f"No K3 mapping for adapter: {prefix}")
        hf = mapping.hf_param
        if isinstance(hf, dict):
            if set(hf) != {"gate", "up"} or b_shape[-2] % 2:
                raise ValueError(f"Unsupported fused adapter mapping: {prefix}: {hf}")
            outputs = [(hf["gate"], [0, b_shape[-2] // 2]), (hf["up"], [b_shape[-2] // 2, b_shape[-2]])]
        elif isinstance(hf, str):
            outputs = [(hf, None)]
        else:
            raise ValueError(f"Unsupported HF mapping type for {prefix}")
        for expert_index in range(a_shape[0] if expert else 1):
            for name, b_slice in outputs:
                if expert:
                    if ".experts.0." not in name:
                        raise ValueError(f"Expected explicit expert zero in mapping: {name}")
                    name = name.replace(".experts.0.", f".experts.{expert_index}.")
                if name not in weight_map and not (name + "_packed" in weight_map and name + "_scale" in weight_map):
                    raise ValueError(f"Adapter target absent from HF base: {name}")
                if name in mappings:
                    raise ValueError(f"Multiple adapter pairs map to {name}")
                mappings[name] = {
                    "a_key": pair["in"],
                    "b_key": pair["out"],
                    "expert_index": expert_index if expert else None,
                    "b_slice": b_slice,
                }
    if not mappings:
        raise ValueError("Checkpoint contains no supported adapters")
    return mappings


def read_checkpoint_spec(checkpoint: str, base_hf: str) -> dict[str, Any]:
    import torch
    from megatron.bridge.models.kimi.kimi_k3_bridge import KimiK3Bridge
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(checkpoint).read_metadata()
    common = read_common_state(checkpoint, metadata)
    args = common["args"]
    base = getattr(args, "hf_checkpoint", None)
    if base is None or Path(base).resolve() != Path(base_hf).resolve():
        raise ValueError(f"Base differs from training hf_checkpoint: {base!r}")
    rank, alpha = getattr(args, "lora_rank", None), getattr(args, "lora_alpha", None)
    if not isinstance(rank, int) or rank <= 0 or not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
        raise ValueError("Missing or invalid saved LoRA rank/alpha")
    if getattr(args, "normalize_moe_lora", False):
        raise ValueError("Streaming export does not yet support normalize_moe_lora")
    if common.get("content_metadata", {}).get("singleton_local_shards", False):
        raise ValueError("Streaming export requires canonical non-singleton DCP tensor layout")
    tensors = {}
    for key, value in metadata.state_dict_metadata.items():
        if not hasattr(value, "size"):
            continue
        if key.startswith("language_model."):
            tensors[key] = {"shape": list(value.size), "dtype": str(value.properties.dtype).removeprefix("torch.")}
        elif not re.match(r"(?:chained_\d+\.|optimizer\.)", key):
            raise ValueError(f"Unsupported non-language saved tensor: {key}")
    config = json.loads((Path(base_hf) / "config.json").read_text())
    if config.get("model_type") != "kimi_k3":
        raise ValueError("Base is not a Kimi-K3 model")
    bridge = object.__new__(KimiK3Bridge)
    bridge._num_hidden_layers = config["text_config"]["num_hidden_layers"]
    registry = bridge.mapping_registry()
    weight_map = json.loads((Path(base_hf) / "model.safetensors.index.json").read_text())["weight_map"]
    mappings = build_mappings(tensors, registry, weight_map, rank)
    headers: dict[str, Any] = {}
    for name, entry in mappings.items():
        packed = name + "_packed" in weight_map
        key = name + "_packed" if packed else name
        filename = weight_map[key]
        if Path(filename).name != filename:
            raise ValueError(f"HF shard path must be a flat filename: {filename}")
        if filename not in headers:
            with (Path(base_hf) / filename).open("rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                if not 0 < size < 100_000_000:
                    raise ValueError(f"Invalid HF header size: {filename}")
                headers[filename] = json.loads(handle.read(size))
        base_shape = list(headers[filename][key]["shape"])
        if packed:
            if len(base_shape) != 2:
                raise ValueError(f"Expected a packed matrix: {key}")
            base_shape[-1] *= 2
        a_shape, b_shape = (tensors[entry[key]]["shape"] for key in ("a_key", "b_key"))
        rows = b_shape[-2] if entry["b_slice"] is None else entry["b_slice"][1] - entry["b_slice"][0]
        if base_shape != [rows, a_shape[-1]]:
            raise ValueError(f"Adapter/base geometry mismatch for {name}: {base_shape} vs {[rows, a_shape[-1]]}")
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "base_hf": str(Path(base_hf).resolve()),
        "rank": rank,
        "alpha": alpha,
        "scale": alpha / rank,
        "tp": getattr(args, "tensor_model_parallel_size", None),
        "pp": getattr(args, "pipeline_model_parallel_size", None),
        "ep": getattr(args, "expert_model_parallel_size", None),
        "expert_tp": getattr(args, "expert_tensor_parallel_size", None),
        "normalize_moe_lora": False,
        "fc1_layout": "canonical gate rows followed by up rows",
        "iteration": common.get("iteration"),
        "tensors": tensors,
        "mappings": mappings,
        "adapter_tensor_bytes": sum(
            math.prod(value["shape"]) * torch.empty((), dtype=getattr(torch, value["dtype"])).element_size()
            for value in tensors.values()
        ),
    }


def merge_weight(base: Any, a: Any, b: Any, entry: dict[str, Any], alpha: float, rank: int) -> Any:
    """Select one expert/projection, merge in FP32, then restore base dtype."""
    import torch

    expert_index = entry["expert_index"]
    if expert_index is not None:
        a, b = a[expert_index], b[expert_index]
    if entry["b_slice"] is not None:
        start, stop = entry["b_slice"]
        b = b[start:stop]
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
        raise ValueError("Invalid selected LoRA matrix geometry")
    if tuple(base.shape) != (b.shape[0], a.shape[1]):
        raise ValueError(f"LoRA/base shape mismatch: {tuple(base.shape)}, {tuple(a.shape)}, {tuple(b.shape)}")
    if base.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Merge requires dequantized floating-point base")
    a = a.to(device=base.device, dtype=torch.float32)
    b = b.to(device=base.device, dtype=torch.float32)
    return (base.float() + (b @ a) * (alpha / rank)).to(base.dtype)
