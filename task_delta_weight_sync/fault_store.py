#!/usr/bin/env python3
"""Damage an isolated delivery copy of an already sealed frozen target.

The ordinary HF export and owner canonical bytes remain unchanged. Put the new
config beside the original config so replay keeps the same HF oracle. Run only
after the producer and other writers have stopped.
"""

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from relax.distributed.weight_sync.config import WeightSyncConfig
from relax.distributed.weight_sync.format import digest
from relax.distributed.weight_sync.store import SharedFSStore, atomic_write


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--delivery-root", type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--fault", choices=["missing-base", "corrupt-object"], required=True)
    args = parser.parse_args()
    config = WeightSyncConfig.load(args.config)
    output = args.output_config.resolve()
    delivery = args.delivery_root.resolve()
    source = SharedFSStore(config.store_root, config.disk_budget_bytes, canonical_root=config.canonical_root)
    if output.parent != args.config.resolve().parent or output.exists():
        raise ValueError("Use a fresh config beside the original frozen sequence config")
    if delivery.exists() or delivery.is_relative_to(source.root) or source.root.is_relative_to(delivery):
        raise ValueError("The delivery copy must be new and disjoint from the original store")
    if delivery.is_relative_to(source.canonical_root) or source.canonical_root.is_relative_to(delivery):
        raise ValueError("The delivery copy must be disjoint from owner canonical material")
    inventory = source.inventory()
    if not any(
        seal["stream"] == config.stream and args.version in seal["versions"] for seal in inventory["seals"].values()
    ):
        raise ValueError("Seal the target and stop its producer before injecting a fault")
    record = source.record(config.stream, args.version)
    primary = source.manifest(record["artifact_id"])
    if primary["kind"] != "DELTA":
        raise ValueError("The target must actually use a DELTA artifact")
    full_id = record["artifacts"]["FULL"]
    if args.fault == "missing-base":
        base_id = source.record(config.stream, primary["base_version"])["artifact_id"]
        relative = Path("artifacts") / base_id
    else:
        blocks = [block for block in primary["blocks"] if block["object"] is not None]
        # Prefer a DELTA-specific object; a RAW object may also belong to FULL,
        # in which case owner canonical recovery must repair it as well.
        blocks.sort(key=lambda block: block["encoding"] in {"RAW", "RAW_ZLIB"})
        relative = Path("objects") / blocks[0]["object"]
    original = (source.root / relative).read_bytes()
    if digest(original) != relative.name:
        raise ValueError("The original delivery material is already corrupt")
    index = (source.root / "index.json").read_bytes()
    if any(path.is_symlink() for path in source.root.rglob("*")):
        raise ValueError("Refuse a source delivery tree containing symlinks")
    shutil.copytree(source.root, delivery)
    if (source.root / "index.json").read_bytes() != index:
        raise RuntimeError("Producer state changed while copying; abandon this copy")
    target = delivery / relative
    before, after = (source.root / relative).stat(), target.stat()
    if (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino):
        raise RuntimeError("Fault copy unexpectedly aliases the original file")
    atomic_write(delivery / "original-object.backup", original)
    if args.fault == "missing-base":
        target.unlink()
    else:
        atomic_write(target, b"task-injected-truncated-object")
    if (source.root / relative).read_bytes() != original:
        raise RuntimeError("Original delivery material changed")
    settings = asdict(config) | {"store_root": str(delivery), "full_only": False}
    atomic_write(output, json.dumps(settings, indent=2).encode())
    result = {
        "fault": args.fault,
        "stream": config.stream,
        "version": args.version,
        "root": record["root"],
        "primary_delta_id": record["artifact_id"],
        "expected_full_id": full_id,
        "damaged_relative_path": str(relative),
        "original_sha256": digest(original),
        "original_index_sha256": digest(index),
        "damaged_sha256": digest(target.read_bytes()) if target.exists() else None,
        "config": str(output),
        "delivery": str(delivery),
        "canonical_read_source": config.canonical_root,
    }
    atomic_write(delivery / "fault.json", json.dumps(result, indent=2).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
