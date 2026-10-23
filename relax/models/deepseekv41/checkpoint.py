# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The BF16 train/rollout checkpoint contract; frozen Engram tables stay
quantized."""

import json
import struct
from pathlib import Path
from typing import Any


_V41_MODEL_TYPES = {"deepseek_v41", "deepseek_v41_text"}
_BF16_REQUIRED = (
    "DeepSeek V4.1 requires a BF16 train/rollout checkpoint: dequantize trainable FP8/FP4 weights, "
    "remove quantization_config and obsolete weight scales, and retain frozen Engram embed weights/scales. "
    "Online synchronization into the original quantized checkpoint is not supported."
)


def _read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Invalid safetensors header in {path}.")
        size = struct.unpack("<Q", prefix)[0]
        if size > path.stat().st_size - 8:
            raise ValueError(f"Truncated safetensors header in {path}.")
        return json.loads(source.read(size))


def validate_bf16_checkpoint(checkpoint: str | None, *, require_v41: bool = False) -> bool:
    """Check config and tensor headers without loading weights, importing
    Torch, or rewriting files.

    Non-V4.1 checkpoints are untouched. Known V4.1 callers require a local
    safetensors directory, as the training Engram loader also needs local
    files. Configuration-only Bridge inspection can still use the original
    checkpoint; this check belongs at execution and online synchronization
    entry points.
    """
    root = Path(checkpoint) if checkpoint else None
    config_path = root / "config.json" if root is not None else None
    if config_path is None or not config_path.is_file():
        if require_v41:
            raise ValueError("DeepSeek V4.1 requires an existing local BF16 checkpoint directory.")
        return False
    with config_path.open() as source:
        config = json.load(source)
    text = config.get("text_config") or config
    if config.get("model_type") not in _V41_MODEL_TYPES and text.get("model_type") not in _V41_MODEL_TYPES:
        if require_v41:
            raise ValueError("DeepSeek V4.1 rollout must use a DeepSeek V4.1 BF16 checkpoint directory.")
        return False
    if config.get("quantization_config") or text.get("quantization_config"):
        raise ValueError(_BF16_REQUIRED)

    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(
            "DeepSeek V4.1 requires a local BF16 safetensors checkpoint index, including frozen Engram tables."
        )
    with index_path.open() as source:
        weight_map = json.load(source)["weight_map"]
    if not weight_map:
        raise ValueError("DeepSeek V4.1 checkpoint index contains no weights.")

    frozen_names: dict[str, tuple[str, ...]] = {}
    for layer in text.get("engram_layer_ids", ()):
        prefix = f"layers.{layer}.engram.embed"
        frozen_names[prefix + ".weight"] = ("F8_E4M3", "U8")
        frozen_names[prefix + ".scale"] = ("F8_E8M0", "U8")
    for name in frozen_names:
        if name not in weight_map:
            raise ValueError(f"DeepSeek V4.1 checkpoint is missing frozen Engram tensor {name!r}.")

    by_shard: dict[str, list[str]] = {}
    num_layers = text.get("num_hidden_layers")
    for name, shard in weight_map.items():
        parts = name.split(".")
        is_text_layer = (
            len(parts) > 2
            and parts[0] == "layers"
            and parts[1].isdigit()
            and (num_layers is None or int(parts[1]) < num_layers)
        )
        if name in frozen_names or is_text_layer or name in ("embed.weight", "head.weight", "norm.weight"):
            if name not in frozen_names and name.endswith((".scale", ".weight_scale_inv")):
                raise ValueError(f"DeepSeek V4.1 checkpoint contains obsolete weight scale {name!r}. {_BF16_REQUIRED}")
            by_shard.setdefault(shard, []).append(name)
    if not by_shard:
        raise ValueError("DeepSeek V4.1 checkpoint index contains no text model weights.")
    for shard, names in by_shard.items():
        header = _read_header(root / shard)
        for name in names:
            if name not in header:
                raise ValueError(f"DeepSeek V4.1 tensor {name!r} is missing from indexed shard {shard!r}.")
            dtype = header[name]["dtype"]
            allowed = frozen_names.get(name, ("BF16", "F32"))
            if dtype not in allowed:
                raise ValueError(f"DeepSeek V4.1 tensor {name!r} has dtype {dtype!r}. {_BF16_REQUIRED}")
    return True
