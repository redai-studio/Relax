# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""save-hf post-hook: fill a text-only export's vision weights from the base.

A VL base trained text-only builds no vision tower, so ``--save-hf`` writes a
checkpoint whose ``config.json`` still declares ``vision_config`` while the
shards hold language weights only (gemma-4-26B: 657 of the reference's 1013
tensors), and ``from_pretrained`` fails on the 356 absent keys. The tower never
trained, so copying it verbatim out of the base is the correct weight:

    --save-hf-post-hook-path scripts.tools.vision_supplement_hook.supplement_vision

Runs synchronously -- ~1 GiB, a few seconds for gemma-4-26B -- because the
export is incomplete until it returns.
"""

from __future__ import annotations

import json
import os
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_INDEX = "model.safetensors.index.json"
_SHARD = "model-vision-00001-of-00001.safetensors"


def supplement_vision(args: Any, hf_path: str, rollout_id: int, *, dtype: str, is_lora: bool) -> None:
    """Copy the reference's vision tensors into ``hf_path`` and extend its
    index.

    A no-op once they are present. Skipped for LoRA adapter directories, and
    for FP8 exports whose ``modules_to_not_convert`` would not cover a bf16
    tower. Assumes a sharded export; the caller logs and swallows anything
    raised here.
    """
    import safetensors
    import safetensors.torch

    if is_lora or dtype == "fp8":
        logger.info(f"[vision-supplement] skipped: is_lora={is_lora} dtype={dtype}")
        return

    reference = args.hf_checkpoint
    with open(os.path.join(reference, _INDEX)) as f:
        ref_map = json.load(f)["weight_map"]
    with open(os.path.join(hf_path, _INDEX)) as f:
        index = json.load(f)

    missing = sorted(k for k in ref_map if "vision" in k.lower() and k not in index["weight_map"])
    if not missing:
        logger.info("[vision-supplement] no vision weights missing; nothing to do")
        return

    tensors = {}
    for shard in sorted({ref_map[k] for k in missing}):
        with safetensors.safe_open(os.path.join(reference, shard), framework="pt", device="cpu") as fh:
            tensors.update({k: fh.get_tensor(k) for k in missing if ref_map[k] == shard})
    safetensors.torch.save_file(tensors, os.path.join(hf_path, _SHARD), metadata={"format": "pt"})

    index["weight_map"].update({k: _SHARD for k in missing})
    meta = index.setdefault("metadata", {})
    meta["total_size"] = meta.get("total_size", 0) + sum(t.numel() * t.element_size() for t in tensors.values())
    with open(os.path.join(hf_path, _INDEX), "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"[vision-supplement] rollout={rollout_id}: added {len(missing)} tensor(s) from {reference}")
