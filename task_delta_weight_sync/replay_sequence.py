#!/usr/bin/env python3
"""Replay frozen real HF targets through an independent SharedFS or TCP
cache."""

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from relax.distributed.weight_sync.config import WeightSyncConfig
from relax.distributed.weight_sync.consumer import Consumer
from relax.distributed.weight_sync.store import SharedFSStore, atomic_write
from relax.distributed.weight_sync.tcp import TCPTransport
from relax.distributed.weight_sync.tensor_diff import compare


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", help="tcp://HOST:PORT; omitted means SharedFS")
    parser.add_argument("--versions", type=int, nargs="+", required=True)
    parser.add_argument("--full-only", action="store_true")
    args = parser.parse_args()
    from urllib.parse import urlparse

    config = WeightSyncConfig.load(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.endpoint:
        endpoint = urlparse(args.endpoint)
        if endpoint.scheme != "tcp" or not endpoint.hostname or endpoint.port is None:
            raise ValueError("Expected tcp://HOST:PORT")
        transport = TCPTransport(endpoint.hostname, endpoint.port, timeout_s=config.sync_timeout_s)
    else:
        transport = SharedFSStore(config.store_root, config.disk_budget_bytes, canonical_root=config.canonical_root)
    cache = SharedFSStore(args.output / "cache", config.disk_budget_bytes)
    consumer = Consumer(
        transport,
        cache,
        args.output / "checkpoints",
        disk_budget_bytes=config.disk_budget_bytes,
        timeout_s=config.sync_timeout_s,
    )
    rows = []
    for version in args.versions:
        started = time.monotonic()
        # This reference is a normal independently exported HF checkpoint.
        export = args.config.parent / "exports" / f"update-{version:06d}"
        reference = args.output / "full" / f"{version:06d}"
        if not reference.exists():
            reference.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(export, reference)
        before = dict(getattr(transport, "metrics", {}))
        prepared = consumer.prepare(config.stream, version, full=args.full_only)
        check = compare(reference, Path(prepared.path))
        check_path = args.output / f"{version:06d}-diff.json"
        atomic_write(check_path, json.dumps(check, indent=2).encode())
        row = asdict(prepared) | {
            "transport": "TCP" if args.endpoint else "SharedFS",
            "consumer_metrics": dict(consumer.last_metrics),
            "full_only": args.full_only,
            "elapsed_s": time.monotonic() - started,
            "diff": str(check_path),
            "wire_delta": {k: v - before.get(k, 0) for k, v in getattr(transport, "metrics", {}).items()},
            "equal": check["equal"],
            "reference": str(reference),
        }
        rows.append(row)
        atomic_write(args.output / "results.json", json.dumps(rows, indent=2).encode())
        consumer.release(prepared)
        if not check["equal"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
