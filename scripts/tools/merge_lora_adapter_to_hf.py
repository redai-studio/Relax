# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Merge a trained LoRA adapter back into its base weights, offline.

Relax exports the trained policy LoRA adapter as a *standard HF-PEFT* directory
(``adapter_config.json`` + ``adapter_model.safetensors``) under
``<checkpoint_dir>/lora_adapter/`` (see
``relax/backends/megatron/checkpoint.py::_save_lora_to_checkpoint``). The adapter
keys are HF-style and are named after the *base tensor* they adapt.

This tool folds that adapter into the base HuggingFace model and writes a
standalone merged HF checkpoint -- no Ray / Megatron / GPU cluster required. It is
the offline counterpart of the inline merge done during RL weight sync
(``HfWeightIteratorBridge._merge_base_with_adapter``); both fold
``base + (alpha/r)*(B @ A)`` into the base weight, so the result matches what the
rollout engine saw in ``--lora-merge-mode`` training.

Usage:
    python scripts/tools/merge_lora_adapter_to_hf.py \\
        --base-hf-dir  /path/to/Qwen3.6-35B-A3B \\
        --adapter-dir  /path/to/save/iter_0000100/lora_adapter \\
        --output-dir   /path/to/Qwen3.6-35B-A3B-merged

For GLM LoRA-only DCP checkpoints saved with --save-lora-only:
    python scripts/tools/merge_lora_adapter_to_hf.py \\
        --base-hf-dir /path/to/GLM-5.3 \\
        --checkpoint /path/to/save/iter_0000049 \\
        --output-dir /path/to/GLM-5.3-merged

The two inputs share tensor-level merge and shard I/O. No model is constructed,
so unrelated architecture/configuration fields and tensors survive the export.
Grouped PEFT experts support batched A/B tensors; GLM DCP expert and gate/up
layouts are mapped to HF keys by lora_dcp.DcpAdapter.

Unquantized PEFT exports preserve the base dtypes and config. Block-128 FP8
bases are dequantized and rounded to BF16 before applying LoRA, and the output
config drops quantization metadata. Other quantization schemes are rejected.
DCP input is currently restricted to GLM version-1 LoRA-only checkpoints with
zero dropout and the identical training base. DCP outputs use a .partial
directory and are published only after identity and shard validation; --force
is supported only for PEFT input.

Memory is bounded by one base shard plus up to four DCP adapter pairs and
per-matrix arithmetic buffers. --device controls the merge matmuls, while base
shards stay on CPU. --dry-run validates metadata/mappings without merging or
publishing files; it does not inspect all tensor values.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_LORA_A_SUFFIXES = (".lora_A.weight", ".lora_A.default.weight")
_LORA_B_SUFFIXES = (".lora_B.weight", ".lora_B.default.weight")
_PEFT_KEY_PREFIX = "base_model.model."
# Weight containers are rebuilt from the base shards; everything else is copied verbatim.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth", ".pt")
_INDEX_FILE = "model.safetensors.index.json"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Merge HF-PEFT or GLM LoRA-only DCP adapters into an HF model, shard by shard."
    )
    parser.add_argument(
        "--base-hf-dir",
        type=str,
        required=True,
        help="Base HuggingFace model directory (the --hf-checkpoint / --ref-load used for training).",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--adapter-dir",
        type=str,
        help="Exported LoRA adapter directory (contains adapter_config.json + adapter_model.safetensors, "
        "e.g. <save>/iter_XXXXXXX/lora_adapter).",
    )
    inputs.add_argument(
        "--checkpoint",
        type=Path,
        help="GLM LoRA-only DCP iteration directory or checkpoint root; mutually exclusive with --adapter-dir.",
    )
    parser.add_argument("--cpu-threads", type=int, default=16, help="CPU tensor-operation threads (default: 16).")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write the merged, standalone HuggingFace model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used for the merge matmuls, e.g. 'cpu' or 'cuda' (default: cpu). The model itself is "
        "never materialized on it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate every adapter/base tensor pairing, report what would be merged, and exit "
        "without writing anything.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists and is non-empty.",
    )
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.checkpoint and args.force:
        parser.error("--force is not supported for DCP exports; choose a new output directory")
    return args


def _validate_adapter_dir(adapter_dir: Path) -> None:
    config_file = adapter_dir / "adapter_config.json"
    if not config_file.is_file():
        raise FileNotFoundError(
            f"{config_file} not found. --adapter-dir must point at the exported HF-PEFT adapter "
            "directory (the one containing adapter_config.json + adapter_model.safetensors)."
        )
    if not (adapter_dir / "adapter_model.safetensors").is_file() and not (adapter_dir / "adapter_model.bin").is_file():
        raise FileNotFoundError(f"No adapter weights (adapter_model.safetensors/.bin) found in {adapter_dir}.")


def _read_adapter_scaling(adapter_dir: Path) -> float:
    """Return the LoRA scaling factor and log the adapter's config + Relax
    sidecar."""
    with open(adapter_dir / "adapter_config.json") as f:
        cfg = json.load(f)

    rank = cfg.get("r")
    alpha = cfg.get("lora_alpha")
    if not rank or alpha is None:
        raise ValueError(f"adapter_config.json in {adapter_dir} is missing 'r' / 'lora_alpha'; cannot merge.")

    if cfg.get("use_rslora"):
        scaling = alpha / (rank**0.5)
        rule = "alpha/sqrt(r) [rslora]"
    else:
        scaling = alpha / rank
        rule = "alpha/r"
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("Expected a positive finite LoRA scaling factor")
    logger.info(
        f"[merge-lora] adapter config: r={rank} alpha={alpha} dropout={cfg.get('lora_dropout')} "
        f"scaling={scaling:g} ({rule}) targets={cfg.get('target_modules', [])}"
    )

    meta_file = adapter_dir / "relax_lora_meta.json"
    if meta_file.is_file():
        with open(meta_file) as f:
            meta = json.load(f)
        mode = "adapter" if meta.get("lora_adapter_mode") else "merge" if meta.get("lora_merge_mode") else "standard"
        logger.info(
            f"[merge-lora] relax sidecar: trained mode={mode}, "
            f"target_modules(megatron)={meta.get('lora_target_modules')}"
        )
    return scaling


def _strip_suffix(key: str, suffixes) -> str:
    """Return the base-tensor prefix if ``key`` ends with one of ``suffixes``,
    else ''."""
    for suffix in suffixes:
        if key.endswith(suffix):
            prefix = key[: -len(suffix)]
            return prefix[len(_PEFT_KEY_PREFIX) :] if prefix.startswith(_PEFT_KEY_PREFIX) else prefix
    return ""


def _pair_adapter_keys(adapter_keys) -> dict:
    """Group ``lora_A`` / ``lora_B`` tensors by the base tensor prefix they
    adapt.

    Returns ``{base_prefix: {"A": key, "B": key}}``. Raises on any half-pair,
    since a lone A or B means the export is corrupt and merging it would be
    wrong.
    """
    pairs: dict = {}
    unrecognized = []
    for key in adapter_keys:
        prefix = _strip_suffix(key, _LORA_A_SUFFIXES)
        if prefix:
            pairs.setdefault(prefix, {})["A"] = key
            continue
        prefix = _strip_suffix(key, _LORA_B_SUFFIXES)
        if prefix:
            pairs.setdefault(prefix, {})["B"] = key
            continue
        unrecognized.append(key)

    if unrecognized:
        raise ValueError(
            f"{len(unrecognized)} adapter tensor(s) are neither lora_A nor lora_B and would be dropped, "
            f"e.g. {unrecognized[:5]}. Refusing to merge a partially-understood adapter."
        )
    incomplete = sorted(p for p, slot in pairs.items() if "A" not in slot or "B" not in slot)
    if incomplete:
        raise ValueError(f"{len(incomplete)} LoRA pair(s) are missing their A or B half, e.g. {incomplete[:5]}.")
    return pairs


def _resolve_base_keys(pairs: dict, base_keys: set) -> dict:
    """Map each adapter prefix to the base tensor key it adapts.

    Handles both ``nn.Linear`` weights (``<prefix>.weight``) and grouped MoE
    expert parameters (``<prefix>`` itself, a 3-D ``nn.Parameter``). Every pair
    must resolve.
    """
    resolved, unmatched = {}, []
    for prefix in pairs:
        if f"{prefix}.weight" in base_keys:
            resolved[prefix] = f"{prefix}.weight"
        elif prefix in base_keys:
            resolved[prefix] = prefix
        else:
            unmatched.append(prefix)

    if unmatched:
        raise ValueError(
            f"{len(unmatched)} LoRA pair(s) have no matching tensor in the base model, e.g. {sorted(unmatched)[:5]}. "
            "This usually means --base-hf-dir is not the base the adapter was trained against."
        )
    return resolved


def _merge_shard(shard_tensors, targets, adapter, pairs, scaling, device):
    """Fold the adapter into ``shard_tensors`` in place.

    Returns the merged key list.
    """
    import torch

    merged = []
    for prefix, base_key in targets.items():
        base = shard_tensors[base_key]
        lora_a = adapter.get_tensor(pairs[prefix]["A"]).to(device=device, dtype=torch.float32)
        lora_b = adapter.get_tensor(pairs[prefix]["B"]).to(device=device, dtype=torch.float32)

        if not torch.isfinite(lora_a).all() or not torch.isfinite(lora_b).all():
            raise ValueError(f"Nonfinite adapter: {prefix}")

        # `@` covers both cases: 2-D (out,r)@(r,in) for nn.Linear weights, and batched
        # (E,out,r)@(E,r,in) for grouped MoE experts stored as one 3-D parameter.
        delta = scaling * (lora_b @ lora_a)
        if delta.shape != base.shape:
            raise ValueError(
                f"LoRA delta shape {tuple(delta.shape)} does not match base tensor {base_key} "
                f"{tuple(base.shape)} (lora_A={tuple(lora_a.shape)}, lora_B={tuple(lora_b.shape)})."
            )
        shard_tensors[base_key] = (base.to(device=device, dtype=torch.float32) + delta).to(
            dtype=base.dtype, device="cpu"
        )
        merged.append(base_key)
    return merged


def _copy_aux_files(base_dir: Path, output_dir: Path) -> None:
    """Copy config.json, tokenizer, preprocessor, index -- everything but the
    weights.

    Copying ``config.json`` verbatim is what keeps ``architectures`` /
    ``model_type`` / ``vision_config`` intact for Megatron-Bridge and SGLang.
    """
    for item in sorted(base_dir.iterdir()):
        if not item.is_file() or item.suffix in _WEIGHT_SUFFIXES:
            continue
        shutil.copy2(item, output_dir / item.name)
        logger.info(f"[merge-lora]   copied {item.name}")


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


def merge(
    base_dir: Path,
    output_dir: Path,
    *,
    adapter_dir: Path | None = None,
    checkpoint: Path | None = None,
    device: str = "cpu",
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Merge either adapter format using the same tensor and shard pipeline.

    DCP exports are published atomically after base/checkpoint identity checks.
    PEFT exports retain the existing --force behavior and base tensor dtypes.
    Block-128 FP8 bases are dequantized to BF16 before adding LoRA deltas.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from scripts.tools.convert_checkpoint_to_hf_cpu import read_hf_schema

    if (adapter_dir is None) == (checkpoint is None):
        raise ValueError("Specify exactly one of adapter_dir or checkpoint")
    base_dir, output_dir = Path(base_dir), Path(output_dir)
    if output_dir.resolve() == base_dir.resolve():
        raise ValueError("Output directory must differ from the base directory")
    if checkpoint is not None and force:
        raise ValueError("--force is not supported for DCP exports")
    weight_map, schema = read_hf_schema(base_dir)
    shard_names = sorted(set(weight_map.values()))
    config = json.loads((base_dir / "config.json").read_text())
    quant = config.get("quantization_config")
    if quant and (quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [128, 128]):
        raise ValueError("Only unquantized or block-128 FP8 bases are supported")
    scales = {key for key in schema if key.endswith("_scale_inv")}
    for key, entry in schema.items():
        if entry["dtype"].startswith("F8"):
            if not quant:
                raise ValueError("FP8 weights require explicit block quantization metadata")
            if key + "_scale_inv" not in scales:
                raise ValueError(f"Missing FP8 scale: {key}")
        if key in scales:
            base_key = key.removesuffix("_scale_inv")
            if base_key not in schema or not schema[base_key]["dtype"].startswith("F8"):
                raise ValueError(f"Orphan FP8 scale companion: {key}")

    with ExitStack() as stack:
        if checkpoint is not None:
            from scripts.tools.lora_dcp import DcpAdapter

            adapter = DcpAdapter(Path(checkpoint), base_dir, weight_map, schema)
            stack.callback(adapter.close)
            scaling = adapter.scaling
        else:
            adapter_dir = Path(adapter_dir)
            if output_dir.resolve() == adapter_dir.resolve():
                raise ValueError("Output directory must differ from the adapter directory")
            _validate_adapter_dir(adapter_dir)
            scaling = _read_adapter_scaling(adapter_dir)
            adapter_file = adapter_dir / "adapter_model.safetensors"
            if not adapter_file.is_file():
                raise FileNotFoundError(f"{adapter_file} not found; convert adapter_model.bin to safetensors first")
            adapter = stack.enter_context(safe_open(str(adapter_file), framework="pt"))

        pairs = _pair_adapter_keys(adapter.keys())
        if not pairs:
            raise ValueError("No LoRA adapter pairs found")
        targets_by_key = _resolve_base_keys(pairs, set(weight_map))
        per_shard: dict = {}
        for prefix, base_key in targets_by_key.items():
            per_shard.setdefault(weight_map[base_key], {})[prefix] = base_key
        report = {"base": str(base_dir), "merged_matrices": len(pairs), "dry_run": dry_run}
        if checkpoint is not None:
            report.update(adapter.report)
            report["output_dtype"] = "bfloat16"
        if dry_run:
            return report

        destination = output_dir.with_name(output_dir.name + ".partial") if checkpoint is not None else output_dir
        if checkpoint is not None:
            if output_dir.exists() or destination.exists():
                raise FileExistsError(f"Refusing to overwrite {output_dir} or {destination}")
        elif output_dir.exists() and any(output_dir.iterdir()) and not force:
            raise ValueError(f"Output directory {output_dir} already exists. Use --force to overwrite.")
        destination.mkdir(parents=True, exist_ok=True)
        total_merged, total_bytes = 0, 0
        output_map = {}
        for i, shard in enumerate(shard_names, start=1):
            targets = per_shard.get(shard, {})
            target_keys = set(targets.values())
            src, dst = base_dir / shard, destination / shard
            if not targets and not quant and checkpoint is None:
                shutil.copy2(src, dst)
                output_map.update({key: shard for key, name in weight_map.items() if name == shard})
                continue
            with safe_open(str(src), framework="pt") as source:
                metadata = source.metadata() or {"format": "pt"}
                tensors = {}
                for key in source.keys():
                    if key in scales:
                        continue
                    scale = None
                    scale_key = key + "_scale_inv"
                    if scale_key in scales:
                        if weight_map[scale_key] == shard:
                            scale = source.get_tensor(scale_key)
                        else:
                            with safe_open(str(base_dir / weight_map[scale_key]), framework="pt") as other:
                                scale = other.get_tensor(scale_key)
                    weight = dequantize_base(source.get_tensor(key), scale)
                    if checkpoint is not None and key in target_keys:
                        weight = weight.to(torch.bfloat16)
                    tensors[key] = weight
                merged = _merge_shard(tensors, targets, adapter, pairs, scaling, device)
                for key, value in tensors.items():
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError(f"Nonfinite output: {key}")
                    tensors[key] = value.contiguous()
                    output_map[key] = shard
                    total_bytes += value.numel() * value.element_size()
                if tensors:
                    save_file(tensors, str(dst), metadata=metadata)
            total_merged += len(merged)
            del tensors
            logger.info(f"[merge-lora] [{i}/{len(shard_names)}] {shard}: merged {len(merged)} tensor(s)")
        if total_merged != len(pairs):
            raise RuntimeError(f"Merged {total_merged} tensors but the adapter has {len(pairs)} pairs")
        if checkpoint is not None:
            adapter.validate_unchanged()
        _copy_aux_files(base_dir, destination)
        if quant or checkpoint is not None:
            config.pop("quantization_config", None)
            config["dtype"] = "bfloat16"
            if "torch_dtype" in config:
                config["torch_dtype"] = "bfloat16"
            (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            (destination / _INDEX_FILE).write_text(
                json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": output_map}, indent=2) + "\n"
            )
        physical_map, physical_schema = read_hf_schema(destination)
        if physical_map != output_map or any(v["dtype"].startswith("F8") for v in physical_schema.values()):
            raise ValueError("Final HF shard/index validation failed")
        report.update(merged_matrices=total_merged, output_tensors=len(output_map))
        if checkpoint is not None:
            report.update(total_bytes=total_bytes, shards=len(set(output_map.values())), all_finite=True)
            (destination / "relax_conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
            destination.rename(output_dir)
        return report


def main() -> None:
    args = _parse_args()
    import torch

    torch.set_num_threads(args.cpu_threads)
    report = merge(
        Path(args.base_hf_dir),
        Path(args.output_dir),
        adapter_dir=Path(args.adapter_dir) if args.adapter_dir else None,
        checkpoint=args.checkpoint,
        device=args.device,
        dry_run=args.dry_run,
        force=args.force,
    )
    logger.info(f"[merge-lora] Done: {report}")


if __name__ == "__main__":
    main()
