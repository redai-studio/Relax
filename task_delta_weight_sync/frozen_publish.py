#!/usr/bin/env python3
"""Publish an independent FULL control from the same frozen ordinary HF
sequence.

This measures offline HF read/capture and encoding/persistence, not a second
training export. Consume primary FULL artifacts without forcing ensure_full.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from relax.distributed.weight_sync.__main__ import checkpoint_tensors
from relax.distributed.weight_sync.config import WeightSyncConfig
from relax.distributed.weight_sync.provenance import runtime_identity
from relax.distributed.weight_sync.publisher import Publisher
from relax.distributed.weight_sync.store import SharedFSStore, atomic_write


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--versions", type=int, nargs="+", required=True)
    args = parser.parse_args()
    config = WeightSyncConfig.load(args.config)
    root, output = args.output_root.resolve(), args.output_config.resolve()
    if root.exists() or any(
        root.is_relative_to(Path(path).resolve()) for path in [config.store_root, config.canonical_root]
    ):
        raise ValueError("Use a new FULL output root outside the original stores")
    if output.parent != args.config.resolve().parent or output.exists():
        raise ValueError("Use a fresh config beside the original frozen exports")
    if args.versions != list(range(len(args.versions))):
        raise ValueError("Replay the contiguous frozen sequence starting at version zero")
    full = WeightSyncConfig.from_dict(
        asdict(config)
        | {"store_root": str(root / "store"), "canonical_root": str(root / "canonical"), "full_only": True}
    )
    root.mkdir(parents=True)
    atomic_write(output, json.dumps(asdict(full), indent=2).encode())
    original = SharedFSStore(config.store_root, canonical_root=config.canonical_root)
    store = SharedFSStore(full.store_root, full.disk_budget_bytes, canonical_root=full.canonical_root)
    started = time.monotonic()
    result = {"status": "PUBLISHING", "versions": [], "runtime": runtime_identity()}
    for version in args.versions:
        expected = original.record(config.stream, version)
        schema = original.manifest(expected["artifact_id"])["catalog"]["schema_context"]
        export = args.config.resolve().parent / "exports" / f"update-{version:06d}"
        publisher = Publisher(store, full.stream, schema, **full.publisher_kwargs())
        stage = time.monotonic()
        try:
            published = publisher.finalize_pending(
                publisher.submit(expected["update_id"], checkpoint_tensors(export)), full.sync_timeout_s
            )
        finally:
            publisher.close()
        if (published.stream, published.version, published.root) != (full.stream, version, expected["root"]):
            raise RuntimeError("FULL control differs from the frozen training target")
        if store.manifest(published.artifact_id)["kind"] != "FULL":
            raise RuntimeError("The control did not publish a primary FULL artifact")
        result["versions"].append(
            asdict(published)
            | {"ordinary_hf": str(export), "metrics": publisher.last_metrics, "wall_s": time.monotonic() - stage}
        )
        atomic_write(root / "results.json", json.dumps(result, indent=2).encode())
    result["status"] = "SEALING"
    atomic_write(root / "results.json", json.dumps(result, indent=2).encode())
    stage = time.monotonic()
    store.seal("frozen-full-control", full.stream, args.versions)
    result.update(
        status="SEALED",
        seal_s=time.monotonic() - stage,
        elapsed_s=time.monotonic() - started,
        sealed_at=time.time(),
        runtime=runtime_identity(),
    )
    atomic_write(root / "results.json", json.dumps(result, indent=2).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
