#!/usr/bin/env python3
"""Seal frozen versions before an independent offline replay.

Run this CPU-only producer as a bounded Job and require its terminal success
before starting consumers. The serving object server is a separate process.
"""

import argparse
import json
import os
import time
from pathlib import Path

from relax.distributed.weight_sync.config import WeightSyncConfig
from relax.distributed.weight_sync.provenance import runtime_identity
from relax.distributed.weight_sync.store import SharedFSStore, atomic_write


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--versions", type=int, nargs="+", required=True)
    parser.add_argument("--seal-id", required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    config = WeightSyncConfig.load(args.config)
    store = SharedFSStore(config.store_root, config.disk_budget_bytes, canonical_root=config.canonical_root)
    started = time.monotonic()
    result = {"pid": os.getpid(), "runtime": runtime_identity(), "versions": [], "status": "PREPARING"}
    atomic_write(args.result, json.dumps(result, indent=2).encode())
    # The public operation pins, materializes FULL variants and verifies their
    # complete contents. Bound the producer Job externally; do not repeat this
    # GB-scale validation solely to obtain per-version stage timings.
    store.seal(args.seal_id, config.stream, args.versions)
    for version in args.versions:
        artifact = store.record(config.stream, version)["artifacts"]["FULL"]
        manifest = store.manifest(artifact)
        result["versions"].append(
            {"version": version, "artifact_id": artifact, "root": manifest["root"], "kind": manifest["kind"]}
        )
    result.update(status="SEALED", seal_id=args.seal_id, elapsed_s=time.monotonic() - started, sealed_at=time.time())
    atomic_write(args.result, json.dumps(result, indent=2).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
