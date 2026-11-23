# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Merge a trained LoRA adapter back into its base weights, offline.

Relax exports the trained policy LoRA adapter as a *standard HF-PEFT* directory
(``adapter_config.json`` + ``adapter_model.safetensors``) under
``<checkpoint_dir>/lora_adapter/`` (see
``relax/backends/megatron/checkpoint.py::_save_lora_to_checkpoint``). The adapter
keys are HF-style and are named after the *base tensor* they adapt.

``--adapter-dir`` also accepts a native Megatron torch_dist LoRA checkpoint
(an iteration directory or checkpoint root). This path delegates to the CPU
checkpoint converter and supports Qwen3.5 MoE / Qwen3.8 full and LoRA-only
checkpoints. It requires saved training args, an unquantized HF reference, and
a fresh output directory; ``--device`` must be ``cpu``.

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

Why this is a tensor-level merge and not ``peft.merge_and_unload()``:

    * ``AutoModelForCausalLM`` silently resolves a multimodal checkpoint to its
      TEXT-ONLY class. For ``model_type: qwen3_5_moe`` transformers maps to
      ``Qwen3_5MoeForCausalLM`` ("# VLM compatibility" in ``modeling_auto.py``),
      whose ``_keys_to_ignore_on_load_unexpected`` drops ``^model.visual.*`` and
      ``^mtp.*`` without a warning. Re-saving then rewrites ``config.json`` to the
      text sub-config (``architectures: Qwen3_5MoeForCausalLM``,
      ``model_type: qwen3_5_moe_text``), losing ``vision_config`` and the image /
      video token ids. Neither Megatron-Bridge nor SGLang recognise the resulting
      architecture, so the merged model cannot be loaded back for training.
    * Grouped MoE experts are stored as 3-D ``nn.Parameter`` tensors
      (``Qwen3_5MoeExperts.gate_up_proj`` / ``.down_proj``), not ``nn.Linear``
      submodules. PEFT matches *modules* by name, so it cannot see them and would
      silently ignore every routed-expert LoRA tensor -- the bulk of the trained
      capacity on an A3B MoE.

    Merging at the tensor level avoids both: unrelated tensors (vision tower, MTP
    head) pass through byte-for-byte, ``config.json`` and every auxiliary file are
    copied verbatim from the base, and an adapter tensor that fails to pair with a
    base tensor is a hard error rather than a silent drop.

Notes:
    * ``--base-hf-dir`` must be the SAME base model used for training (the
      ``--hf-checkpoint`` / ``--ref-load`` dir); LoRA deltas are only meaningful
      against the weights they were trained on.
    * Memory is bounded by the largest single shard (a few GB), not by the model
      size -- shards are streamed one at a time.
    * ``--device cuda`` only moves the merge matmuls to the GPU; the model is never
      materialized there.
    * MoE ``router`` LoRA (base tensor ``mlp.gate.weight``) is merged like any other
      2-D weight; a warning is printed since router LoRA is disabled in merge-mode
      training and is rarely intended for a merged export.
"""

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any


_RELAX_ROOT = str(Path(__file__).resolve().parents[2])
if _RELAX_ROOT in sys.path:
    sys.path.remove(_RELAX_ROOT)
sys.path.insert(0, _RELAX_ROOT)

_LORA_A_SUFFIXES = (".lora_A.weight", ".lora_A.default.weight")
_LORA_B_SUFFIXES = (".lora_B.weight", ".lora_B.default.weight")
_PEFT_KEY_PREFIX = "base_model.model."
# Weight containers are rebuilt from the base shards; everything else is copied verbatim.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth", ".pt")
_INDEX_FILE = "model.safetensors.index.json"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Merge HF-PEFT adapters or native torch_dist LoRA checkpoints into HF weights, offline."
    )
    parser.add_argument(
        "--base-hf-dir",
        type=str,
        required=True,
        help="Base HuggingFace model directory (the --hf-checkpoint / --ref-load used for training).",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        required=True,
        help="HF-PEFT adapter directory, or native torch_dist LoRA iteration/checkpoint root.",
    )
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
        help="Overwrite an existing HF-PEFT merge output. Native DCP exports require a fresh directory.",
    )
    parser.add_argument(
        "--legacy-vision-qkv-layout",
        choices=("hf", "megatron"),
        help="B row order of an unmarked Relax visual QKV adapter: hf for standard PEFT, "
        "megatron for historical head-interleaved Relax exports.",
    )
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--memory-budget-gb", type=float, default=64, help="Native DCP tensor-buffer budget in GiB")
    parser.add_argument("--max-shard-size-gb", type=float, default=2, help="Native DCP output shard target in GiB")
    parser.add_argument(
        "--allow-base-fallback",
        choices=("vision", "ple", "mtp"),
        action="append",
        default=[],
        help="Native full DCP only: copy missing frozen tensors in this group from the exact training base",
    )
    return parser.parse_args()


def _validate_adapter_dir(adapter_dir: Path) -> None:
    config_file = adapter_dir / "adapter_config.json"
    if not config_file.is_file():
        raise FileNotFoundError(
            f"{config_file} not found. --adapter-dir must point at the exported HF-PEFT adapter "
            "directory (the one containing adapter_config.json + adapter_model.safetensors)."
        )
    if not (adapter_dir / "adapter_model.safetensors").is_file() and not (adapter_dir / "adapter_model.bin").is_file():
        raise FileNotFoundError(f"No adapter weights (adapter_model.safetensors/.bin) found in {adapter_dir}.")


def _read_adapter_config(adapter_dir: Path) -> tuple[dict[str, Any], bool, str | None]:
    """Read the PEFT config and recognize the Relax expert-rank convention."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    rank = cfg.get("r")
    alpha = cfg.get("lora_alpha")
    if not rank or alpha is None:
        raise ValueError(f"adapter_config.json in {adapter_dir} is missing 'r' / 'lora_alpha'; cannot merge.")

    scaling = alpha / (math.sqrt(rank) if cfg.get("use_rslora") else rank)
    rule = "alpha/sqrt(r) [rslora]" if cfg.get("use_rslora") else "alpha/r"
    print(
        f"[merge-lora] adapter config: r={rank} alpha={alpha} dropout={cfg.get('lora_dropout')} "
        f"scaling={scaling:g} ({rule}) targets={cfg.get('target_modules', [])}"
    )

    meta_file = adapter_dir / "relax_lora_meta.json"
    relax_export = False
    vision_layout = None
    if meta_file.is_file():
        with open(meta_file) as f:
            meta = json.load(f)
        relax_export = meta.get("lora_rank") == rank and meta.get("lora_alpha") == alpha
        vision_layout = meta.get("vision_qkv_b_layout")
        mode = "adapter" if meta.get("lora_adapter_mode") else "merge" if meta.get("lora_merge_mode") else "standard"
        print(
            f"[merge-lora] relax sidecar: trained mode={mode}, "
            f"target_modules(megatron)={meta.get('lora_target_modules')}"
        )
    return cfg, relax_export, vision_layout


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
            if "A" in pairs.get(prefix, {}):
                raise ValueError(f"Duplicate LoRA A factor for {prefix}")
            pairs.setdefault(prefix, {})["A"] = key
            continue
        prefix = _strip_suffix(key, _LORA_B_SUFFIXES)
        if prefix:
            if "B" in pairs.get(prefix, {}):
                raise ValueError(f"Duplicate LoRA B factor for {prefix}")
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
    if not pairs:
        raise ValueError("No LoRA factor pairs found in adapter")
    return pairs


def _read_base_index(base_dir: Path):
    """Return ``(weight_map, shard_filenames)`` for the base checkpoint."""
    index_file = base_dir / _INDEX_FILE
    if index_file.is_file():
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
        return weight_map, sorted(set(weight_map.values()))

    single = base_dir / "model.safetensors"
    if single.is_file():
        from safetensors import safe_open

        with safe_open(str(single), framework="pt") as f:
            weight_map = {key: single.name for key in f.keys()}
        return weight_map, [single.name]

    raise FileNotFoundError(
        f"No {_INDEX_FILE} or model.safetensors in {base_dir}. Only safetensors checkpoints are supported; "
        "convert a .bin checkpoint first."
    )


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


def _module_scaling(config: dict[str, Any], prefix: str, actual_rank: int, *, relax_export: bool = False) -> float:
    """Apply PEFT per-module scaling, including Relax's normalized expert
    ranks."""
    if config.get("use_dora") or config.get("fan_in_fan_out"):
        raise ValueError("DoRA and fan_in_fan_out adapters are not supported by this tensor merge")

    def pattern_value(field: str, default: Any) -> tuple[Any, bool]:
        for pattern, value in config.get(field, {}).items():
            if re.match(rf"(.*\.)?({pattern})$", prefix):
                return value, True
        return default, False

    rank, explicit_rank = pattern_value("rank_pattern", config.get("r"))
    alpha, _ = pattern_value("alpha_pattern", config.get("lora_alpha"))
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError(f"Invalid LoRA rank for {prefix}: {rank}")
    if actual_rank != rank:
        if not (relax_export and not explicit_rank and ".experts." in prefix):
            raise ValueError(f"LoRA rank mismatch for {prefix}: config={rank}, tensors={actual_rank}")
        # Native Relax export writes global r in adapter_config.json, while
        # normalized routed-expert tensors retain their effective smaller rank.
        rank = actual_rank
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha):
        raise ValueError(f"Invalid LoRA alpha for {prefix}: {alpha}")
    return alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)


def _check_delta_shape(key: str, base: tuple[int, ...], delta: tuple[int, ...]) -> None:
    if base == delta:
        return
    if len(base) == 3 and ".experts." in key and delta in (base[1:], (1, *base[1:])):
        return
    raise ValueError(f"LoRA delta shape {delta} does not match base tensor {key} {base}")


def _vision_qkv_reorders(
    targets_by_key: dict[str, str],
    base_config: dict[str, Any],
    layout: str | None,
    legacy_layout: str | None,
    *,
    relax_export: bool,
) -> dict[str, tuple[int, int]]:
    """Identify packed visual B factors that need conversion from Megatron head
    order."""

    from relax.utils.megatron_peft_utils import is_visual_qkv_weight

    keys = [key for key in targets_by_key.values() if is_visual_qkv_weight(key)]
    if not keys:
        return {}
    if relax_export and layout is None and legacy_layout is None:
        raise ValueError(
            "Unmarked Relax vision QKV adapter: select --legacy-vision-qkv-layout megatron "
            "for the historical Relax export, or hf for a verified HF-ordered adapter."
        )
    if layout is not None and legacy_layout is not None and layout != legacy_layout:
        raise ValueError("Requested legacy vision QKV layout conflicts with the adapter metadata")
    selected = legacy_layout if legacy_layout is not None else layout
    if selected is None or selected == "hf":
        return {}
    if selected != "megatron":
        raise ValueError(f"Unsupported vision QKV B layout: {selected!r}")
    vision = base_config.get("vision_config", {})
    heads, width = vision.get("num_heads"), vision.get("hidden_size")
    if not isinstance(heads, int) or heads <= 0 or not isinstance(width, int) or width <= 0 or width % heads:
        raise ValueError("Legacy vision QKV conversion requires the original vision hidden_size and num_heads")
    return dict.fromkeys(keys, (heads, width // heads))


def _merge_shard(shard_tensors, targets, adapter, pairs, scaling, device, *, vision_reorders=None):
    """Fold the adapter into ``shard_tensors`` in place.

    Returns the merged key list.
    """
    import torch

    merged = []
    for prefix, base_key in targets.items():
        base = shard_tensors[base_key]
        lora_a = adapter.get_tensor(pairs[prefix]["A"]).to(device=device, dtype=torch.float32)
        lora_b = adapter.get_tensor(pairs[prefix]["B"]).to(device=device, dtype=torch.float32)

        if vision_reorders and base_key in vision_reorders:
            heads, head_dim = vision_reorders[base_key]
            lora_b = lora_b.reshape(heads, 3, head_dim, -1).transpose(0, 1).reshape_as(lora_b)

        # `@` covers both cases: 2-D (out,r)@(r,in) for nn.Linear weights, and batched
        # (E,out,r)@(E,r,in) for grouped MoE experts stored as one 3-D parameter.
        factor = scaling[prefix] if isinstance(scaling, dict) else scaling
        delta = factor * (lora_b @ lora_a)
        _check_delta_shape(base_key, tuple(base.shape), tuple(delta.shape))
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
        print(f"[merge-lora]   copied {item.name}")


def main() -> None:
    args = _parse_args()

    base_dir = Path(args.base_hf_dir)
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)

    if not base_dir.is_dir():
        raise FileNotFoundError(f"--base-hf-dir {base_dir} does not exist.")
    if output_dir.resolve() == base_dir.resolve() or output_dir.resolve() == adapter_dir.resolve():
        raise ValueError("--output-dir must differ from the base model and adapter directories")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    if (adapter_dir / ".metadata").is_file() or (adapter_dir / "latest_checkpointed_iteration.txt").is_file():
        if args.device != "cpu":
            raise ValueError("Native torch_dist LoRA conversion requires --device cpu")
        import torch

        from scripts.tools.convert_checkpoint_to_hf_cpu import export_checkpoint

        torch.set_num_threads(args.cpu_threads)
        export_checkpoint(
            adapter_dir,
            base_dir,
            output_dir,
            shard_bytes=int(args.max_shard_size_gb * 1024**3),
            memory_budget=int(args.memory_budget_gb * 1024**3),
            allow_base_fallback=tuple(args.allow_base_fallback),
            dry_run=args.dry_run,
            require_lora=True,
        )
        return
    _validate_adapter_dir(adapter_dir)

    if not args.dry_run and output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise ValueError(f"Output directory {output_dir} already exists and is non-empty. Use --force to overwrite.")

    config, relax_export, vision_layout = _read_adapter_config(adapter_dir)
    base_config = json.loads((base_dir / "config.json").read_text())
    if base_config.get("quantization_config") or base_config.get("text_config", {}).get("quantization_config"):
        raise ValueError("Merge requires an unquantized HF base model")

    # Heavy deps imported lazily so `--help` / arg validation work without a full ML stack.
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_num_threads(args.cpu_threads)

    weight_map, shard_names = _read_base_index(base_dir)
    print(f"[merge-lora] base model: {len(weight_map)} tensors across {len(shard_names)} shard(s) in {base_dir}")

    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(
            f"{adapter_file} not found. Only safetensors adapters are supported; convert adapter_model.bin first."
        )

    with safe_open(str(adapter_file), framework="pt") as adapter:
        pairs = _pair_adapter_keys(adapter.keys())
        targets_by_key = _resolve_base_keys(pairs, set(weight_map))
        vision_reorders = _vision_qkv_reorders(
            targets_by_key, base_config, vision_layout, args.legacy_vision_qkv_layout, relax_export=relax_export
        )
        scaling = {}
        delta_shapes = {}
        for prefix, pair in pairs.items():
            a, b = (tuple(adapter.get_slice(pair[side]).get_shape()) for side in ("A", "B"))
            if len(a) not in (2, 3) or len(b) not in (2, 3) or a[-2] != b[-1] or a[-2] <= 0:
                raise ValueError(f"Invalid LoRA factor shapes for {prefix}: {a}, {b}")
            if targets_by_key[prefix] in vision_reorders:
                heads, dim = vision_reorders[targets_by_key[prefix]]
                if len(b) != 2 or b[0] != 3 * heads * dim:
                    raise ValueError(f"Invalid legacy vision QKV B shape for {prefix}: {b}")
            scaling[prefix] = _module_scaling(config, prefix, a[-2], relax_export=relax_export)
            delta_shapes[prefix] = (*torch.broadcast_shapes(a[:-2], b[:-2]), b[-2], a[-1])
            if any(adapter.get_slice(key).get_dtype() not in ("BF16", "F16", "F32", "F64") for key in pair.values()):
                raise ValueError(f"LoRA factors must be floating point: {prefix}")
        print(f"[merge-lora] adapter: {len(pairs)} LoRA pair(s), all resolved against the base model")

        router = sorted(p for p in targets_by_key if p.endswith(".mlp.gate"))
        if router:
            print(
                f"[merge-lora] WARNING: {len(router)} MoE router tensor(s) carry LoRA and will be merged "
                f"(e.g. {router[0]}). Router LoRA is disabled in merge-mode training -- double-check "
                "this is intended."
            )

        # Group the targets by the shard holding their base tensor, so each shard is
        # read, merged and written exactly once.
        per_shard: dict = {}
        for prefix, base_key in targets_by_key.items():
            per_shard.setdefault(weight_map[base_key], {})[prefix] = base_key

        # Validate every target before creating output, including --dry-run.
        for shard, targets in per_shard.items():
            with safe_open(base_dir / shard, framework="pt", device="cpu") as handle:
                for prefix, key in targets.items():
                    base = handle.get_slice(key)
                    shape, delta = tuple(base.get_shape()), delta_shapes[prefix]
                    _check_delta_shape(key, shape, delta)
                    if base.get_dtype() not in ("BF16", "F16", "F32", "F64"):
                        raise ValueError(f"LoRA merge requires a floating-point base tensor: {key}")

        if args.dry_run:
            for shard in shard_names:
                print(f"[merge-lora]   {shard}: {len(per_shard.get(shard, {}))} tensor(s) to merge")
            print(f"[merge-lora] dry run OK -- {len(pairs)} pair(s) would be merged into {len(per_shard)} shard(s).")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        total_merged = 0
        for i, shard in enumerate(shard_names, start=1):
            targets = per_shard.get(shard, {})
            src, dst = base_dir / shard, output_dir / shard
            if not targets:
                # Nothing to merge here (vision tower, MTP head, ...) -- keep the bytes as-is.
                shutil.copy2(src, dst)
                print(f"[merge-lora] [{i}/{len(shard_names)}] {shard}: no LoRA, copied verbatim")
                continue

            with safe_open(str(src), framework="pt") as f:
                metadata = f.metadata() or {"format": "pt"}
                shard_tensors = {key: f.get_tensor(key) for key in f.keys()}
            merged = _merge_shard(
                shard_tensors, targets, adapter, pairs, scaling, args.device, vision_reorders=vision_reorders
            )
            save_file(shard_tensors, str(dst), metadata=metadata)
            total_merged += len(merged)
            print(f"[merge-lora] [{i}/{len(shard_names)}] {shard}: merged {len(merged)} tensor(s)")
            del shard_tensors

    if total_merged != len(pairs):
        raise RuntimeError(f"Merged {total_merged} tensors but the adapter has {len(pairs)} pairs -- refusing to lie.")

    print("[merge-lora] copying config / tokenizer / processor files from the base model ...")
    _copy_aux_files(base_dir, output_dir)

    print(f"[merge-lora] Done! Merged {total_merged} tensor(s) into {output_dir}")


if __name__ == "__main__":
    main()
