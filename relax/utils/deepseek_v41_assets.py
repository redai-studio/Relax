# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Native DeepSeek V4.1 vision checkpoint schema, independent of the model runtime."""

import hashlib
import json
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any


NATIVE_VISION_FILES = ("encoding/encoding.py", "inference/image_processor.py", "inference/vision.py")


V41_REPLICATED_LAYER_WEIGHTS = {
    "input_layernorm.weight": "attn_norm.weight",
    "pre_mlp_layernorm.weight": "ffn_norm.weight",
    "self_attention.linear_q_down_proj.weight": "attn.wq_a.weight",
    "self_attention.q_layernorm.weight": "attn.q_norm.weight",
    "self_attention.linear_kv_proj.weight": "attn.wkv.weight",
    "self_attention.kv_layernorm.weight": "attn.kv_norm.weight",
    "self_attention.core_attention.compressor.linear_wkv.weight": "attn.compressor.wkv.weight",
    "self_attention.core_attention.compressor.linear_wgate.weight": "attn.compressor.wgate.weight",
    "self_attention.core_attention.compressor.norm.weight": "attn.compressor.norm.weight",
    "self_attention.core_attention.indexer.linear_wq_b.weight": "attn.indexer.wq_b.weight",
    "self_attention.core_attention.indexer.linear_weights_proj.weight": "attn.indexer.weights_proj.weight",
    "self_attention.core_attention.indexer.linear_wk.weight": "attn.indexer.wk.weight",
    "self_attention.core_attention.indexer.k_norm.weight": "attn.indexer.k_norm.weight",
    "mlp.router.weight": "ffn.gate.weight",
    "mlp.router.expert_bias": "ffn.gate.bias",
    "v41_engram.linear_wkv.weight": "engram.wkv.weight",
    "v41_engram.q_weight": "engram.q_weight",
    "v41_engram.k_weight": "engram.k_weight",
}


def vision_tensor_shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return the native tower, aligner, image marker and language router tensor shapes."""
    vision = config["vision_config"]
    text = config["text_config"]
    hidden, intermediate = vision["hidden_size"], vision["intermediate_size"]
    language = text["hidden_size"]
    if not isinstance(config.get("image_token_id"), int) or not 0 <= config["image_token_id"] < text["vocab_size"]:
        raise ValueError("DeepSeek V4.1 requires an image token in the text vocabulary.")
    if vision["num_hidden_layers"] < 1 or hidden <= 0 or language <= 0:
        raise ValueError("Invalid DeepSeek V4.1 vision dimensions.")
    shapes = {
        "vision.patch_embed.proj.weight": (hidden, 3 * vision["patch_size"] ** 2),
        "vision.patch_embed.proj.bias": (hidden,),
        "vision.norm.weight": (hidden,),
        "aligner.w1.weight": (language, hidden * vision["downsample_ratio"] ** 2),
        "aligner.w1.bias": (language,),
        "aligner.w2.weight": (language, language),
        "aligner.w2.bias": (language,),
        "image_start": (language,),
        "image_end": (language,),
        "image_newline": (language,),
    }
    block = {
        "norm1.weight": (hidden,),
        "attn.wqkv.weight": (3 * hidden, hidden),
        "attn.wqkv.bias": (3 * hidden,),
        "attn.wo.weight": (hidden, hidden),
        "attn.wo.bias": (hidden,),
        "norm2.weight": (hidden,),
        "mlp.w1.weight": (2 * intermediate, hidden),
        "mlp.w2.weight": (hidden, intermediate),
    }
    for i in range(vision["num_hidden_layers"]):
        shapes.update({f"vision.blocks.{i}.{name}": shape for name, shape in block.items()})
    for i in range(text["num_hidden_layers"]):
        shapes[f"layers.{i}.ffn.gate.bias_vl"] = (text["n_routed_experts"],)
    return shapes


def validate_vision_files(root: Path) -> None:
    for asset in ("tokenizer.json", "tokenizer_config.json", *NATIVE_VISION_FILES):
        if not (root / asset).is_file():
            raise ValueError(f"Missing required multimodal checkpoint asset: {asset}")


def validate_vision_entry(name: str, entry: dict[str, Any], shape: tuple[int, ...]) -> None:
    if tuple(entry["shape"]) != shape or entry["dtype"] not in ("BF16", "F32"):
        raise ValueError(f"Incompatible native vision tensor {name}: {entry}")


FROZEN_BASE_RECORD = "v41-frozen-base.json"


@lru_cache(maxsize=4)
def frozen_base_identity(root: Path) -> dict[str, Any]:
    """Fingerprint immutable base metadata and shard stats once per process, without reading TBs."""
    validate_vision_files(root)
    index = root / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    stats = []
    for name in shards:
        stat = (root / name).stat()
        stats.append((name, stat.st_size, stat.st_mtime_ns))
    files = ["config.json", index.name, "tokenizer.json", "tokenizer_config.json", *NATIVE_VISION_FILES]
    for name in ("chat_template.jinja", "conversion.json"):
        if (root / name).is_file():
            files.append(name)
    return {
        "files_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
        "shard_stats_sha256": hashlib.sha256(json.dumps(stats).encode()).hexdigest(),
    }


def check_frozen_base(directory: Path, base: Path, *, record: bool = False) -> None:
    """Bind a DCP iteration to the exact immutable composite base used to train it."""
    path = directory / FROZEN_BASE_RECORD
    expected = frozen_base_identity(base.resolve())
    if path.is_file():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"Frozen vision/Engram base differs from the training record in {path}")
        return
    if not record:
        raise ValueError(f"Missing training base identity in {path}")
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / f".{FROZEN_BASE_RECORD}.{uuid.uuid4().hex}.tmp"
    try:
        with pending.open("x") as stream:
            json.dump(expected, stream)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
