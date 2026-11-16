# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Expose GLM LoRA-only DCP tensors through the HF adapter reader interface.

Only checkpoint metadata, tensor layouts and bounded adapter reads live here.
The merge command owns base dequantization, arithmetic and shard publication.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from scripts.tools.convert_checkpoint_to_hf_cpu import check_chunks, resolve_checkpoint


_A = ".adapter.linear_in.weight"
_B = ".adapter.linear_out.weight"
_ATTENTION = {
    "linear_q_down_proj": "q_a_proj",
    "linear_q_up_proj": "q_b_proj",
    "linear_kv_down_proj": "kv_a_proj_with_mqa",
    "linear_kv_up_proj": "kv_b_proj",
    "linear_proj": "o_proj",
}


def base_identity(base: Path, weight_map: dict[str, str]) -> dict[str, str]:
    """Match version-1 lightweight checkpoint identity, including shard
    stats."""
    stats = []
    for name in sorted(set(weight_map.values())):
        stat = (base / name).stat()
        stats.append((name, stat.st_size, stat.st_mtime_ns))
    return {
        "config_sha256": hashlib.sha256((base / "config.json").read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256((base / "model.safetensors.index.json").read_bytes()).hexdigest(),
        "shard_stat_sha256": hashlib.sha256(
            json.dumps(stats, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest(),
        "shard_count": str(len(stats)),
    }


def adapter_targets(
    shapes: dict[str, tuple[int, ...]], schema: dict[str, dict[str, Any]], rank: int
) -> dict[str, tuple[str, int | None, int | None]]:
    """Map HF keys to (DCP module, expert index, gate/up half).

    DCP factories already gather the expert axis and restore FC1's global gate-
    then-up row order; applying a second TP interleave would corrupt it.
    """
    prefixes = {key[: -len(_A)] for key in shapes if key.endswith(_A)}
    expected = {prefix + suffix for prefix in prefixes for suffix in (_A, _B)}
    if not prefixes or set(shapes) != expected:
        raise ValueError("Incomplete or unsupported DCP adapter pairs")
    targets = {}
    for prefix in sorted(prefixes):
        match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", prefix)
        if match is None:
            raise ValueError(f"Unsupported adapter module: {prefix}")
        layer, module = match.groups()
        a, b = shapes[prefix + _A], shapes[prefix + _B]
        if len(a) not in (2, 3) or len(a) != len(b) or a[-2] != rank or b[-1] != rank:
            raise ValueError(f"Invalid adapter shape: {prefix}: {a}, {b}")
        if len(a) == 3 and a[0] != b[0]:
            raise ValueError(f"Mismatched expert axes: {prefix}")
        head = f"model.layers.{layer}."
        expert_indices: list[int | None] = [None]
        split = False
        if module.startswith("self_attention.") and module.split(".")[-1] in _ATTENTION:
            names = ["self_attn." + _ATTENTION[module.split(".")[-1]]]
        else:
            match = re.fullmatch(r"mlp\.(?:(shared_experts|experts\.experts)\.)?linear_fc([12])", module)
            if match is None:
                raise ValueError(f"Unsupported adapter module: {prefix}")
            kind, fc = match.groups()
            split = fc == "1"
            tail = ["gate_proj", "up_proj"] if split else ["down_proj"]
            if kind == "experts.experts":
                if len(a) != 3:
                    raise ValueError(f"Expected globally expanded expert axis: {prefix}")
                expert_indices = list(range(a[0]))
                names = ["mlp.experts.{expert}." + name for name in tail]
            else:
                names = ["mlp." + ("shared_experts." if kind else "") + name for name in tail]
        if expert_indices == [None] and len(a) != 2:
            raise ValueError(f"Unexpected expert axis: {prefix}")
        if split and b[-2] % 2:
            raise ValueError(f"Odd gate/up dimension: {prefix}")
        for expert in expert_indices:
            for half, name in enumerate(names):
                key = head + name.format(expert=expert) + ".weight"
                expected_shape = (b[-2] // (2 if split else 1), a[-1])
                if key not in schema or tuple(schema[key]["shape"]) != expected_shape:
                    raise ValueError(f"Adapter/base shape mismatch: {prefix} -> {key}, expected {expected_shape}")
                if key in targets:
                    raise ValueError(f"Duplicate adapter target: {key}")
                targets[key] = (prefix, expert, half if split else None)
    return targets


class DcpAdapter:
    """Read validated GLM DCP adapters with HF-style A/B keys, without a
    model."""

    def __init__(self, checkpoint: Path, base: Path, weight_map: dict, schema: dict) -> None:
        import torch
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.metadata import TensorStorageMetadata

        checkpoint = resolve_checkpoint(checkpoint)
        config = json.loads((base / "config.json").read_text())
        if config.get("model_type") != "glm_moe_dsa":
            raise ValueError("This converter supports only glm_moe_dsa checkpoints")
        reader = dcp.FileSystemReader(checkpoint)
        metadata = reader.read_metadata()
        if (checkpoint / "common.pt").is_file():
            common = torch.load(checkpoint / "common.pt", map_location="cpu", weights_only=False)
        else:
            # Current MCore stores its one common object inside DCP, wrapped in a
            # single-element list. Keep this CPU tool independent of CUDA/MCore.
            common_key = "common_state/shard_0_1"
            state = {common_key: io.BytesIO()}
            dcp.load(state_dict=state, storage_reader=reader, no_dist=True)
            common = state[common_key]
            if isinstance(common, io.BytesIO):
                common.seek(0)
                common = torch.load(common, map_location="cpu", weights_only=False)
            if not isinstance(common, list) or len(common) != 1:
                raise ValueError("Invalid embedded common checkpoint state")
            common = common[0]
        spec = getattr(common.get("args"), "relax_lora_checkpoint_metadata", None)
        if not spec or spec.get("format_version") != 1 or spec.get("lora_dropout") != 0:
            raise ValueError("Expected a version-1 LoRA-only checkpoint with zero adapter dropout")
        if not isinstance(spec.get("lora_rank"), int) or spec["lora_rank"] <= 0:
            raise ValueError("Expected a positive integer LoRA rank")
        if not math.isfinite(spec.get("lora_alpha", float("nan"))) or spec["lora_alpha"] <= 0:
            raise ValueError("Expected a positive finite LoRA alpha")
        identity = base_identity(base, weight_map)
        if spec.get("base_hf_checkpoint") != identity:
            raise ValueError("Base checkpoint identity differs from the training checkpoint")
        tensor_metadata = {
            k: v for k, v in metadata.state_dict_metadata.items() if isinstance(v, TensorStorageMetadata)
        }
        shapes = {k: tuple(v.size) for k, v in tensor_metadata.items() if ".adapter." in k}
        if any(k.startswith(("decoder.", "embedding.", "output_layer.")) and k not in shapes for k in tensor_metadata):
            raise ValueError("Expected adapter-only model tensors")
        for key in shapes:
            check_chunks(key, tensor_metadata[key])
        targets = adapter_targets(shapes, schema, spec["lora_rank"])
        self.checkpoint = checkpoint
        self.base = base
        self.weight_map = weight_map
        self.identity = identity
        self.fingerprint = hashlib.sha256((checkpoint / ".metadata").read_bytes()).hexdigest()
        self.scaling = spec["lora_alpha"] / spec["lora_rank"]
        self.targets = targets
        self.report = {
            "checkpoint": str(checkpoint),
            "base_identity": identity,
            "dcp_metadata_sha256": self.fingerprint,
            "adapter_tensors": len(shapes),
        }

        @lru_cache(maxsize=4)
        def load_pair(prefix: str) -> tuple[Any, Any]:
            state = {
                key: torch.empty(shapes[key], dtype=metadata.state_dict_metadata[key].properties.dtype)
                for key in (prefix + _A, prefix + _B)
            }
            dcp.load(state_dict=state, storage_reader=reader, no_dist=True)
            if not all(bool(torch.isfinite(value).all()) for value in state.values()):
                raise ValueError(f"Nonfinite adapter: {prefix}")
            return state[prefix + _A], state[prefix + _B]

        self._load_pair = load_pair

    def keys(self) -> list[str]:
        return [
            key.removesuffix(".weight") + suffix
            for key in self.targets
            for suffix in (".lora_A.weight", ".lora_B.weight")
        ]

    def get_tensor(self, key: str) -> Any:
        prefix, side = key.rsplit(".lora_", 1)
        module, expert, half = self.targets[prefix + ".weight"]
        a, b = self._load_pair(module)
        if expert is not None:
            a, b = a[expert], b[expert]
        if half is not None:
            b = b.chunk(2, dim=0)[half]
        if side not in ("A.weight", "B.weight"):
            raise KeyError(key)
        return a if side == "A.weight" else b

    def validate_unchanged(self) -> None:
        if hashlib.sha256((self.checkpoint / ".metadata").read_bytes()).hexdigest() != self.fingerprint:
            raise ValueError("Checkpoint changed during conversion")
        if base_identity(self.base, self.weight_map) != self.identity:
            raise ValueError("Base changed during conversion")

    def close(self) -> None:
        self._load_pair.cache_clear()
