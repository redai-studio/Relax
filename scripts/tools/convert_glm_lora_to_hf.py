# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Merge a Relax GLM LoRA-only DCP into standalone BF16 HF shards on CPU.

Memory is bounded by one HF shard and four adapter pairs. The original FP8 base
is dequantized and rounded to BF16 before applying the adapter, matching
GLM5Bridge training initialization. Input checkpoints and base shards are never
modified. Output is published only after all adapters and HF shards validate.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from relax.utils.logging_utils import get_logger
from scripts.tools.convert_checkpoint_to_hf_cpu import check_chunks, read_hf_schema, resolve_checkpoint


logger = get_logger(__name__)
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


def dequantize_base(weight: Any, scale: Any | None) -> Any:
    import torch

    if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        if scale is not None:
            raise ValueError("Scale companion attached to a non-FP8 weight")
        return weight
    if weight.ndim != 2 or scale is None:
        raise ValueError("FP8 matrix requires its block scale companion")
    rows, cols = weight.shape
    nr, nc = (rows + 127) // 128, (cols + 127) // 128
    if tuple(scale.shape) != (nr, nc):
        raise ValueError(f"Invalid FP8 scale grid: {tuple(scale.shape)}, expected {(nr, nc)}")
    values = torch.nn.functional.pad(weight.float(), (0, nc * 128 - cols, 0, nr * 128 - rows))
    values = values.view(nr, 128, nc, 128)
    values.mul_(scale.float()[:, None, :, None])
    return values.view(nr * 128, nc * 128)[:rows, :cols].to(torch.bfloat16).contiguous()


def merge_adapter(base: Any, a: Any, b: Any, scaling: float, expert: int | None, half: int | None) -> Any:
    import torch

    if expert is not None:
        a, b = a[expert], b[expert]
    if half is not None:
        b = b.chunk(2, dim=0)[half]
    delta = b.float() @ a.float()
    if delta.shape != base.shape:
        raise ValueError("LoRA delta shape differs from base")
    return (base.to(torch.bfloat16).float() + scaling * delta).to(torch.bfloat16)


def convert(checkpoint: Path, base: Path, output: Path, *, dry_run: bool = False) -> dict[str, Any]:
    import torch
    import torch.distributed.checkpoint as dcp
    from safetensors import safe_open
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    checkpoint = resolve_checkpoint(checkpoint)
    weight_map, schema = read_hf_schema(base)
    config = json.loads((base / "config.json").read_text())
    if config.get("model_type") != "glm_moe_dsa":
        raise ValueError("This converter supports only glm_moe_dsa checkpoints")
    quant = config.get("quantization_config")
    if quant and (quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [128, 128]):
        raise ValueError("Only BF16 or block-128 FP8 bases are supported")
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
    tensor_metadata = {k: v for k, v in metadata.state_dict_metadata.items() if isinstance(v, TensorStorageMetadata)}
    shapes = {k: tuple(v.size) for k, v in tensor_metadata.items() if ".adapter." in k}
    if any(k.startswith(("decoder.", "embedding.", "output_layer.")) and k not in shapes for k in tensor_metadata):
        raise ValueError("Expected adapter-only model tensors")
    for key in shapes:
        check_chunks(key, tensor_metadata[key])
    targets = adapter_targets(shapes, schema, spec["lora_rank"])
    scales = {key for key in schema if key.endswith("_scale_inv")}
    for key, entry in schema.items():
        if entry["dtype"].startswith("F8") and not quant:
            raise ValueError("FP8 weights require explicit block quantization metadata")
        if entry["dtype"].startswith("F8") and key + "_scale_inv" not in scales:
            raise ValueError(f"Missing FP8 scale: {key}")
    if any(key[: -len("_scale_inv")] not in schema for key in scales):
        raise ValueError("Orphan FP8 scale companion")
    fingerprint = hashlib.sha256((checkpoint / ".metadata").read_bytes()).hexdigest()
    report = {
        "checkpoint": str(checkpoint),
        "base": str(base),
        "base_identity": identity,
        "dcp_metadata_sha256": fingerprint,
        "adapter_tensors": len(shapes),
        "merged_matrices": len(targets),
        "output_dtype": "bfloat16",
        "dry_run": dry_run,
    }
    if dry_run:
        return report
    temporary = output.with_name(output.name + ".partial")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {temporary}")
    temporary.mkdir(parents=True)

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

    seen = set()
    output_map = {}
    total_bytes = 0
    shards = sorted(set(weight_map.values()))
    for index, name in enumerate(shards):
        tensors = {}
        with safe_open(base / name, framework="pt", device="cpu") as source:
            for key in sorted(source.keys()):
                if key in scales:
                    continue
                weight = source.get_tensor(key)
                scale_key = key + "_scale_inv"
                scale = None
                if scale_key in scales:
                    if weight_map[scale_key] == name:
                        scale = source.get_tensor(scale_key)
                    else:
                        with safe_open(base / weight_map[scale_key], framework="pt", device="cpu") as other:
                            scale = other.get_tensor(scale_key)
                weight = dequantize_base(weight, scale)
                if key in targets:
                    prefix, expert, half = targets[key]
                    a, b = load_pair(prefix)
                    weight = merge_adapter(weight, a, b, spec["lora_alpha"] / spec["lora_rank"], expert, half)
                    seen.add(key)
                if not bool(torch.isfinite(weight).all()):
                    raise ValueError(f"Nonfinite output: {key}")
                tensors[key] = weight.contiguous()
                output_map[key] = name
                total_bytes += weight.numel() * weight.element_size()
            save_file(tensors, temporary / name, metadata={"format": "pt"})
        del tensors
        logger.info("Converted shard %d/%d: %s", index + 1, len(shards), name)
    if seen != set(targets):
        raise ValueError(f"Unmerged adapter targets: {sorted(set(targets) - seen)[:5]}")
    if hashlib.sha256((checkpoint / ".metadata").read_bytes()).hexdigest() != fingerprint:
        raise ValueError("Checkpoint changed during conversion")
    if base_identity(base, weight_map) != identity:
        raise ValueError("Base changed during conversion")
    for path in base.iterdir():
        if (
            path.is_file()
            and path.suffix not in (".safetensors", ".bin", ".pt", ".pth")
            and path.name not in ("config.json", "model.safetensors.index.json")
        ):
            shutil.copy2(path, temporary / path.name)
    config.pop("quantization_config", None)
    config["dtype"] = "bfloat16"
    if "torch_dtype" in config:
        config["torch_dtype"] = "bfloat16"
    (temporary / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (temporary / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": output_map}, indent=2) + "\n"
    )
    physical_map, physical_schema = read_hf_schema(temporary)
    if physical_map != output_map or any(v["dtype"].startswith("F8") for v in physical_schema.values()):
        raise ValueError("Final HF shard/index validation failed")
    report.update(total_bytes=total_bytes, shards=len(shards), output_tensors=len(output_map), all_finite=True)
    (temporary / "relax_conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    temporary.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-hf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    import torch

    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    logger.info(
        "Conversion result: %s", convert(args.checkpoint, args.base_hf_dir, args.output_dir, dry_run=args.dry_run)
    )


if __name__ == "__main__":
    main()
