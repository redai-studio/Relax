# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Identity of the frozen Hugging Face weights used by native LoRA
checkpoints."""

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from relax.utils.model_source import is_model_source_alias, normalize_model_path


def hf_checkpoint_identity(path: str | Path) -> dict[str, str | None]:
    root = Path(path)

    def digest(filename: str) -> str | None:
        candidate = root / filename
        if not candidate.is_file():
            return None
        return hashlib.sha256(candidate.read_bytes()).hexdigest()

    index_path = next(
        (
            candidate
            for candidate in (
                root / "model.safetensors.index.json",
                root / "pytorch_model.bin.index.json",
            )
            if candidate.is_file()
        ),
        None,
    )
    identity = {
        "config_sha256": digest("config.json"),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest() if index_path else None,
    }
    shard_names: list[str] = []
    if index_path is not None:
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index.get("weight_map", {}).values()))
    elif root.is_dir():
        shard_names = sorted(
            candidate.name for pattern in ("*.safetensors", "pytorch_model*.bin") for candidate in root.glob(pattern)
        )

    shard_stats = []
    for name in shard_names:
        shard = root / name
        if not shard.is_file():
            raise FileNotFoundError(f"HF base checkpoint shard is missing: {shard}")
        stat = shard.stat()
        shard_stats.append((name, stat.st_size, stat.st_mtime_ns))
    if shard_stats:
        encoded = json.dumps(shard_stats, separators=(",", ":"), ensure_ascii=True).encode()
        identity["shard_stat_sha256"] = hashlib.sha256(encoded).hexdigest()
        identity["shard_count"] = str(len(shard_stats))
    if not shard_stats:
        identity["path"] = str(root.resolve())
    return identity


def verified_lora_base_identity(saved: dict[str, Any], marker: dict[str, Any]) -> dict[str, str | None]:
    """Read new provenance and accept v1 only when its saved source is
    unambiguous."""
    if not isinstance(marker, dict) or marker.get("format_version") not in (1, 2):
        raise ValueError("Unsupported LoRA-only checkpoint metadata version")
    base = marker.get("base_hf_checkpoint")
    fields = {"config_sha256", "index_sha256", "shard_stat_sha256", "shard_count"}
    if not isinstance(base, dict) or not fields.issubset(base) or not base.get("shard_stat_sha256"):
        raise ValueError("LoRA-only checkpoint is missing its base HF identity")
    if marker["format_version"] == 1:
        hf = saved.get("hf_checkpoint")
        args = Namespace(**saved)

        def identifies_hf(path: str | None) -> bool:
            return (
                not path or normalize_model_path(path) == normalize_model_path(hf) or is_model_source_alias(args, path)
            )

        if (
            not hf
            or "load" not in saved
            or "ref_load" not in saved
            or saved.get("ckpt_step")
            or not identifies_hf(saved.get("load"))
            or not identifies_hf(saved.get("ref_load"))
            or (saved.get("save") and identifies_hf(saved["save"]))
        ):
            raise ValueError("Cannot establish the frozen HF base of this legacy v1 LoRA-only checkpoint")
    return base
