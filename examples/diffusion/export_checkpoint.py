#!/usr/bin/env python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Export a committed DCP checkpoint to a loadable diffusers tree (design §12).

Loads the adapter's ``weight_name_map`` and calls the FSDP checkpoint exporter,
then reloads the safetensors and checks the tensor set is non-empty and finite.

Pass ``--base-model-path`` (the training ``--model-path``) to also copy the
frozen pipeline components — VAE, text encoder, scheduler, tokenizer,
``model_index.json`` and the transformer ``config.json`` — so ``--output-dir``
is a directory ``DiffusionPipeline.from_pretrained`` can load directly. Without
it only the trained transformer weights are written.

For a LoRA run the checkpoint holds only the adapter, and there are two useful
export forms:

``--form merged`` (default)
    Fold the adapter into the base and write a full diffusers tree.
    ``--base-model-path`` is REQUIRED — without the base there is nothing to fold
    into.
``--form adapter``
    Write a portable HF-PEFT directory (``adapter_config.json`` +
    ``adapter_model.safetensors``), a few hundred MB instead of tens of GB.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from relax.backends.fsdp import checkpoint as ckpt
from relax.utils.utils import load_function


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save-dir", required=True, help="${SAVE_DIR} root containing <task>/iter_*.")
    p.add_argument("--task", required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--model-adapter-path",
        default=None,
        help="Adapter class dotpath. Required for --form merged; unused for --form adapter.",
    )
    p.add_argument(
        "--form",
        default="merged",
        choices=["merged", "adapter"],
        help=(
            "merged: a full diffusers tree (LoRA folded into the base if the checkpoint is "
            "adapter-only). adapter: an HF-PEFT adapter directory; LoRA checkpoints only."
        ),
    )
    p.add_argument(
        "--base-model-path",
        default=None,
        help=(
            "Base model directory (the training --model-path). When given, the frozen pipeline "
            "components are copied alongside the trained transformer so the export is a directly "
            "loadable diffusers pipeline. Required for --form merged on a LoRA checkpoint."
        ),
    )
    p.add_argument(
        "--transformer-subfolder",
        default="transformer",
        help="Subfolder the trained transformer is written to (matches the base model layout).",
    )
    p.add_argument("--verify", action="store_true", help="Reload exported safetensors and sanity-check.")
    args = p.parse_args()

    contract = ckpt.read_adapter_contract(args.save_dir, args.task, args.iteration)
    is_lora = contract.get("save_mode") == "adapter"

    if args.form == "adapter":
        out = ckpt.export_peft_adapter(args.save_dir, args.task, args.iteration, args.output_dir)
        print(f"Exported PEFT adapter to {out}")
        if args.verify:
            _verify_adapter(out, contract)
        return

    if not args.model_adapter_path:
        raise SystemExit("--model-adapter-path is required for --form merged.")
    adapter = load_function(args.model_adapter_path)()
    out = ckpt.export_hf(
        args.save_dir,
        args.task,
        args.iteration,
        args.output_dir,
        name_map=adapter.weight_name_map,
        base_model_path=args.base_model_path,
        subfolder=args.transformer_subfolder,
    )
    print(f"Exported to {out}" + (" (LoRA folded into base)" if is_lora else ""))

    if args.verify:
        _verify_merged(out, args)


def _verify_adapter(out: str, contract: dict) -> None:
    """Check the PEFT directory is complete and its config matches the run."""
    import torch
    from safetensors import safe_open

    config_path = os.path.join(out, "adapter_config.json")
    weights_path = os.path.join(out, "adapter_model.safetensors")
    for path in (config_path, weights_path):
        if not os.path.exists(path):
            raise SystemExit(f"Export verification failed: missing {path}.")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    lora_meta = contract.get("lora") or {}
    for cfg_key, meta_key in (("r", "rank"), ("lora_alpha", "alpha")):
        if meta_key in lora_meta and config.get(cfg_key) != lora_meta[meta_key]:
            raise SystemExit(
                f"Export verification failed: adapter_config.json {cfg_key}={config.get(cfg_key)} "
                f"!= checkpoint {meta_key}={lora_meta[meta_key]}."
            )
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if not keys:
            raise SystemExit(f"Export verification failed: {weights_path} has no tensors.")
        for k in keys:
            if not torch.isfinite(f.get_tensor(k)).all():
                raise SystemExit(f"Export verification failed: non-finite tensor {k}.")
    print(f"Verified {len(keys)} adapter tensors, r={config.get('r')} alpha={config.get('lora_alpha')}.")


def _verify_merged(out: str, args: argparse.Namespace) -> None:
    import torch
    from safetensors import safe_open

    # The transformer lands in a subfolder only when the frozen components
    # were copied next to it; otherwise it is written at the export root.
    weight_dir = os.path.join(out, args.transformer_subfolder) if args.base_model_path else out
    shards = sorted(glob.glob(os.path.join(weight_dir, "diffusion_pytorch_model*.safetensors")))
    if not shards:
        raise SystemExit(f"Export verification failed: no safetensors shards under {weight_dir}.")

    total = 0
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if not keys:
                raise SystemExit(f"Export verification failed: {shard} has no tensors.")
            # A merged export must contain no adapter tensors: their presence means
            # the fold was skipped and the "transformer" is really a bare adapter,
            # which is non-empty and finite and would otherwise pass silently.
            leaked = [k for k in keys if ".lora_A." in k or ".lora_B." in k]
            if leaked:
                raise SystemExit(
                    f"Export verification failed: {shard} still contains LoRA tensors "
                    f"(e.g. {leaked[0]}); the adapter was not folded into the base."
                )
            for k in keys:
                if not torch.isfinite(f.get_tensor(k)).all():
                    raise SystemExit(f"Export verification failed: non-finite tensor {k} in {shard}.")
            total += len(keys)
    print(f"Verified {total} tensors across {len(shards)} shard(s) in {weight_dir}.")


if __name__ == "__main__":
    main()
