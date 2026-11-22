# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Identity of the frozen Hugging Face weights used by native LoRA
checkpoints."""

import hashlib
import json
import re
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from relax.utils.model_source import is_model_source_alias, normalize_model_path


def hf_checkpoint_identity(path: str | Path, *, version: int = 1) -> dict[str, str | None]:
    """Fingerprint the index and ordered shards.

    V1 preserves its original stat-based encoding.
    """
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

    if not shard_names:
        identity["path"] = str(root.resolve())
        return identity

    def fingerprint(name: str) -> tuple:
        shard = root / name
        if not shard.is_file():
            raise FileNotFoundError(f"HF base checkpoint shard is missing: {shard}")
        if version == 1:
            stat = shard.stat()
            return name, stat.st_size, stat.st_mtime_ns
        digest = hashlib.sha256()
        with shard.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024**2), b""):
                digest.update(block)
        return name, digest.hexdigest()

    if version == 1:
        entries = [fingerprint(name) for name in shard_names]
    else:
        with ThreadPoolExecutor(max_workers=min(4, len(shard_names))) as pool:
            entries = list(pool.map(fingerprint, shard_names))
    encoded = json.dumps(entries, separators=(",", ":"), ensure_ascii=True).encode()
    field = "shard_stat_sha256" if version == 1 else "shard_content_sha256"
    identity[field] = hashlib.sha256(encoded).hexdigest()
    identity["shard_count"] = str(len(entries))
    return identity


def has_mutable_router_bias(args: dict[str, Any]) -> bool:
    return bool(args.get("moe_router_enable_expert_bias") and args.get("moe_router_bias_update_rate", 1e-3) != 0)


def verified_lora_base_identity(
    saved: dict[str, Any], marker: dict[str, Any], reference: Path | None = None
) -> dict[str, str | None]:
    """Read new provenance and accept v1 only when its saved source is
    unambiguous."""
    if not isinstance(marker, dict) or marker.get("format_version") not in (1, 2):
        raise ValueError("Unsupported LoRA-only checkpoint metadata version")
    base = marker.get("base_hf_checkpoint")
    if has_mutable_router_bias(saved):
        raise ValueError("LoRA-only checkpoint omits the updated MoE expert bias; use a full DCP")
    field = "shard_stat_sha256" if marker["format_version"] == 1 else "shard_content_sha256"
    fields = {"config_sha256", "index_sha256", field, "shard_count"}
    if not isinstance(base, dict) or not fields.issubset(base) or not base.get(field):
        raise ValueError("LoRA-only checkpoint is missing its base HF identity")
    if marker["format_version"] == 1:
        hf = saved.get("hf_checkpoint")
        args = Namespace(**saved)

        def identifies_hf(path: str | None) -> bool:
            return (
                not path or normalize_model_path(path) == normalize_model_path(hf) or is_model_source_alias(args, path)
            )

        def native_evidence(directory: Path) -> bool:
            if (directory / "latest_checkpointed_iteration.txt").is_file():
                return True
            return bool(re.fullmatch(r"iter_\d{7}", directory.name)) and (
                (directory / ".metadata").is_file()
                or (directory / "metadata.json").is_file()
                or any(directory.glob("mp_rank_*/model_optim_rng.pt"))
            )

        if (
            not hf
            or "load" not in saved
            or "ref_load" not in saved
            or saved.get("ckpt_step")
            or not identifies_hf(saved.get("load"))
            or not identifies_hf(saved.get("ref_load"))
            or (saved.get("save") and identifies_hf(saved["save"]))
            or (
                saved.get("load")
                and not is_model_source_alias(args, saved["load"])
                and not (Path(hf) / "config.json").is_file()
            )
            or native_evidence(Path(hf))
            or (reference is not None and native_evidence(reference))
        ):
            raise ValueError("Cannot establish the frozen HF base of this legacy v1 LoRA-only checkpoint")
    return base
