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
      submodules. Older Relax exports address these parameters directly. Bridge
      exports can instead pack their adapters under nested PEFT ParamWrapper
      keys and record the original names in ``target_parameters``.

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
import shutil
import sys
from pathlib import Path


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
        description="Merge a trained LoRA adapter (HF-PEFT dir) back into its base HF model, offline."
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
        help="Exported LoRA adapter directory (contains adapter_config.json + adapter_model.safetensors, "
        "e.g. <save>/iter_XXXXXXX/lora_adapter).",
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
        help="Overwrite the output directory if it already exists and is non-empty.",
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


def _adapter_scaling(config: dict, target: str = "") -> float:
    # Bridge records rank overrides under the original HF parameter names.
    rank = config.get("rank_pattern", {}).get(target, config["r"])
    alpha = config.get("alpha_pattern", {}).get(target, config["lora_alpha"])
    return alpha / (rank**0.5 if config.get("use_rslora") else rank)


def _read_adapter_config(adapter_dir: Path) -> dict:
    """Read the PEFT config and log its default scaling and Relax sidecar."""
    with open(adapter_dir / "adapter_config.json") as f:
        cfg = json.load(f)

    rank = cfg.get("r")
    alpha = cfg.get("lora_alpha")
    if not rank or alpha is None:
        raise ValueError(f"adapter_config.json in {adapter_dir} is missing 'r' / 'lora_alpha'; cannot merge.")

    scaling = _adapter_scaling(cfg)
    rule = "alpha/sqrt(r) [rslora]" if cfg.get("use_rslora") else "alpha/r"
    print(
        f"[merge-lora] adapter config: r={rank} alpha={alpha} dropout={cfg.get('lora_dropout')} "
        f"scaling={scaling:g} ({rule}) targets={cfg.get('target_modules', [])}"
    )

    meta_file = adapter_dir / "relax_lora_meta.json"
    if meta_file.is_file():
        with open(meta_file) as f:
            meta = json.load(f)
        mode = "adapter" if meta.get("lora_adapter_mode") else "merge" if meta.get("lora_merge_mode") else "standard"
        print(
            f"[merge-lora] relax sidecar: trained mode={mode}, "
            f"target_modules(megatron)={meta.get('lora_target_modules')}"
        )
    return cfg


def _strip_suffix(key: str, suffixes) -> str:
    """Return the base-tensor prefix if ``key`` ends with one of ``suffixes``,
    else ''."""
    for suffix in suffixes:
        if key.endswith(suffix):
            prefix = key[: -len(suffix)]
            return prefix[len(_PEFT_KEY_PREFIX) :] if prefix.startswith(_PEFT_KEY_PREFIX) else prefix
    return ""


def _pair_adapter_keys(adapter_keys, target_parameters=()) -> dict:
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
    # PEFT nests one ParamWrapper per target on a parent module, in config order.
    next_wrapper = {}
    for target in target_parameters:
        parent = target.rpartition(".")[0]
        wrapper = next_wrapper.get(parent, parent)
        if wrapper not in pairs:
            raise ValueError(f"Missing PEFT adapter pair {wrapper!r} for target parameter {target!r}.")
        pairs[wrapper]["parameter"] = target
        next_wrapper[parent] = f"{wrapper}.base_layer"
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
    for prefix, pair in pairs.items():
        name = pair.get("parameter", prefix)
        if "parameter" not in pair and f"{name}.weight" in base_keys:
            resolved[prefix] = f"{name}.weight"
        elif name in base_keys:
            resolved[prefix] = name
        else:
            unmatched.append(name)

    if unmatched:
        raise ValueError(
            f"{len(unmatched)} LoRA pair(s) have no matching tensor in the base model, e.g. {sorted(unmatched)[:5]}. "
            "This usually means --base-hf-dir is not the base the adapter was trained against."
        )
    return resolved


def _merge_shard(shard_tensors, targets, adapter, pairs, config, device):
    """Fold the adapter into ``shard_tensors`` in place.

    Returns the merged key list.
    """
    import torch

    merged = []
    for prefix, base_key in targets.items():
        base = shard_tensors[base_key]
        lora_a = adapter.get_tensor(pairs[prefix]["A"]).to(device=device, dtype=torch.float32)
        lora_b = adapter.get_tensor(pairs[prefix]["B"]).to(device=device, dtype=torch.float32)

        if "parameter" in pairs[prefix]:
            # Bridge stacks transposed expert B matrices by rows in PEFT A, and
            # transposed expert A matrices by columns in PEFT B.
            num_experts = base.size(0)
            lora_a, lora_b = lora_b.T.unflatten(0, (num_experts, -1)), lora_a.unflatten(0, (num_experts, -1)).mT

        # `@` covers both cases: 2-D (out,r)@(r,in) for nn.Linear weights, and batched
        # (E,out,r)@(E,r,in) for grouped MoE experts stored as one 3-D parameter.
        scaling = _adapter_scaling(config, pairs[prefix].get("parameter", prefix))
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
        print(f"[merge-lora]   copied {item.name}")


def main() -> None:
    args = _parse_args()

    base_dir = Path(args.base_hf_dir)
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)

    if not base_dir.is_dir():
        raise FileNotFoundError(f"--base-hf-dir {base_dir} does not exist.")
    _validate_adapter_dir(adapter_dir)

    if not args.dry_run and output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise ValueError(f"Output directory {output_dir} already exists and is non-empty. Use --force to overwrite.")

    config = _read_adapter_config(adapter_dir)

    # Heavy deps imported lazily so `--help` / arg validation work without a full ML stack.
    from safetensors import safe_open
    from safetensors.torch import save_file

    weight_map, shard_names = _read_base_index(base_dir)
    print(f"[merge-lora] base model: {len(weight_map)} tensors across {len(shard_names)} shard(s) in {base_dir}")

    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(
            f"{adapter_file} not found. Only safetensors adapters are supported; convert adapter_model.bin first."
        )

    with safe_open(str(adapter_file), framework="pt") as adapter:
        pairs = _pair_adapter_keys(adapter.keys(), config.get("target_parameters") or ())
        targets_by_key = _resolve_base_keys(pairs, set(weight_map))
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
            merged = _merge_shard(shard_tensors, targets, adapter, pairs, config, args.device)
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
