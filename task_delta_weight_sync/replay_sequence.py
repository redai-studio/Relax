#!/usr/bin/env python3
"""Replay frozen real HF targets through an independent SharedFS or TCP
cache."""

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

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
    return replay(args.config, args.output, args.versions, args.full_only, args.endpoint)


def replay(
    config_path: Path,
    output: Path,
    versions: list[int],
    full_only: bool = False,
    tcp_endpoint: str | None = None,
    installed: Callable[[dict], None] | None = None,
) -> int:
    from urllib.parse import urlparse

    config = WeightSyncConfig.load(config_path)
    output.mkdir(parents=True, exist_ok=True)
    if tcp_endpoint:
        endpoint = urlparse(tcp_endpoint)
        if endpoint.scheme != "tcp" or not endpoint.hostname or endpoint.port is None:
            raise ValueError("Expected tcp://HOST:PORT")
        transport = TCPTransport(endpoint.hostname, endpoint.port, timeout_s=config.sync_timeout_s)
    else:
        transport = SharedFSStore(config.store_root, config.disk_budget_bytes, canonical_root=config.canonical_root)
    cache = SharedFSStore(output / "cache", config.disk_budget_bytes)
    consumer = Consumer(
        transport,
        cache,
        output / "checkpoints",
        disk_budget_bytes=config.disk_budget_bytes,
        timeout_s=config.sync_timeout_s,
    )
    rows = []
    for version in versions:
        started = time.monotonic()
        # This reference is a normal independently exported HF checkpoint.
        export = config_path.parent / "exports" / f"update-{version:06d}"
        reference = output / "full" / f"{version:06d}"
        reference_started = time.monotonic()
        if not reference.exists():
            reference.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(export, reference)
        reference_copy_s = time.monotonic() - reference_started
        before = dict(getattr(transport, "metrics", {}))
        prepared = consumer.prepare(config.stream, version, full=full_only)
        manifest = cache.manifest(prepared.artifact_id)
        check_started = time.monotonic()
        check = compare(reference, Path(prepared.path))
        hf_check_s = time.monotonic() - check_started
        check_path = output / f"{version:06d}-diff.json"
        atomic_write(check_path, json.dumps(check, indent=2).encode())
        consumer.release(prepared)
        row = asdict(prepared) | {
            "transport": "TCP" if tcp_endpoint else "SharedFS",
            "consumer_metrics": dict(consumer.last_metrics),
            "full_only": full_only,
            "artifact_kind": manifest["kind"],
            "reference_copy_s": reference_copy_s,
            "hf_check_s": hf_check_s,
            "elapsed_s": time.monotonic() - started,
            "diff": str(check_path),
            "wire_delta": {k: v - before.get(k, 0) for k, v in getattr(transport, "metrics", {}).items()},
            "equal": check["equal"],
            "reference": str(reference),
        }
        if not check["equal"]:
            rows.append(row)
            atomic_write(output / "results.json", json.dumps(rows, indent=2).encode())
            return 1
        if installed:
            installed(row)
        row["elapsed_s"] = time.monotonic() - started
        rows.append(row)
        atomic_write(output / "results.json", json.dumps(rows, indent=2).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
