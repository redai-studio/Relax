# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Command line entrypoint for offline publication and replay."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .checkpoint import snapshot_checkpoint, write_checkpoint
from .store import VersionStore, canonical
from .transport import ObjectServer, SharedFSTransport, TCPTransport, fetch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--disk-budget", required=True, type=int, help="Store byte limit")
    parser.add_argument("--max-weight-bytes", required=True, type=int)
    parser.add_argument("--canonical-root", help="Independent volume for retained logical weights")
    parser.add_argument("--canonical-budget", type=int, help="Byte limit for this store's canonical namespace")
    sub = parser.add_subparsers(dest="command", required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("--checkpoint", required=True)
    publish.add_argument("--stream", required=True)
    publish.add_argument("--version", type=int, required=True)
    publish.add_argument("--anchor-interval", type=int, default=20)
    publish.add_argument("--full-only", action="store_true")
    publish.add_argument("--compression", action="store_true", help="Compare raw and XOR zlib encodings")
    serve = sub.add_parser("serve")
    serve.add_argument("--full-timeout", type=float, default=600, help="Owner FULL repair operation budget in seconds")
    serve.add_argument("--control-timeout", type=float, default=120, help="Store control operation budget in seconds")
    serve.add_argument("--bind", default="127.0.0.1")
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--metrics-output", help="Save server socket and repair-write counters on orderly shutdown")
    get = sub.add_parser("fetch")
    location = get.add_mutually_exclusive_group(required=True)
    location.add_argument("--source-store")
    location.add_argument("--host")
    get.add_argument("--port", type=int)
    get.add_argument("--control-timeout", type=float, default=120, help="TCP control operation budget in seconds")
    get.add_argument("--stream", required=True)
    get.add_argument("--version", type=int, required=True)
    get.add_argument("--checkpoint-output", required=True)
    get.add_argument("--config", help="Optional exact config override; otherwise use the published configuration")
    get.add_argument("--full", action="store_true", help="Fetch a same-target FULL repair package")
    status = sub.add_parser("status")
    status.add_argument("--stream", required=True)
    status.add_argument("--version", required=True, type=int)
    full = sub.add_parser("full")
    full.add_argument("--stream", required=True)
    full.add_argument("--version", required=True, type=int)
    collect = sub.add_parser("collect")
    collect.add_argument("--retain-anchors", type=int, default=2)
    register = sub.add_parser("register")
    register.add_argument("--consumer", required=True)
    register.add_argument("--stream", required=True)
    register.add_argument("--start-version", type=int, default=0)
    register.add_argument("--mode", choices=("all_versions", "latest"), default="all_versions")
    acknowledge = sub.add_parser("acknowledge")
    acknowledge.add_argument("--consumer", required=True)
    acknowledge.add_argument("--receipt", required=True)
    unregister = sub.add_parser("unregister")
    unregister.add_argument("--consumer", required=True)
    versions = sub.add_parser("list")
    versions.add_argument("--stream", required=True)
    versions.add_argument("--after", type=int, default=-1)
    versions.add_argument("--limit", type=int, default=100)
    pin = sub.add_parser("pin")
    pin.add_argument("--stream", required=True)
    pin.add_argument("--version", required=True, type=int)
    pin.add_argument("--owner", required=True)
    unpin = sub.add_parser("unpin")
    unpin.add_argument("--owner", required=True)
    seal = sub.add_parser("seal", help="Verify and retain a version set for offline replay")
    seal.add_argument("--name", required=True)
    seal.add_argument("--stream", required=True)
    seal.add_argument("--versions", type=int, nargs="+", required=True)
    release = sub.add_parser("release-seal", help="Release a sealed set's retention")
    release.add_argument("--name", required=True)
    consume = sub.add_parser("consume")
    consume.add_argument(
        "--runtime-config", required=True, help="Explicit engine, transport and bounded version range"
    )
    args = parser.parse_args()
    store = VersionStore(
        args.store,
        disk_budget=args.disk_budget,
        max_weight_bytes=args.max_weight_bytes,
        canonical_root=args.canonical_root,
        canonical_budget=args.canonical_budget,
    )
    if args.command == "publish":
        with tempfile.TemporaryDirectory(prefix="export-", dir=store.root) as directory:
            tensors = snapshot_checkpoint(args.checkpoint, directory, max_weight_bytes=args.max_weight_bytes)
            manifest = store.publish(
                args.stream,
                args.version,
                tensors,
                anchor_interval=args.anchor_interval,
                full_only=args.full_only,
                compression=args.compression,
                model_config=json.loads((Path(args.checkpoint) / "config.json").read_bytes()),
                converter_revision="hf-safetensors-v1",
            )
        sys.stdout.buffer.write(canonical(manifest) + b"\n")
    elif args.command == "serve":
        from .diagnostics import save_diagnostic
        from .metrics import SyncMetrics

        metrics = SyncMetrics("object-service") if args.metrics_output else None
        server = ObjectServer(
            (args.bind, args.port),
            store,
            metrics=metrics,
            full_timeout=args.full_timeout,
            control_timeout=args.control_timeout,
        )
        try:
            with server:
                server.serve_forever()
        finally:
            if metrics is not None:
                try:
                    save_diagnostic(
                        Path(args.metrics_output),
                        canonical({"io_metrics": metrics.report(), "socket_ledger": server.ledger.report()}),
                    )
                except OSError as error:
                    print(f"Object service stopped; metric output could not be saved: {error}", file=sys.stderr)
    elif args.command == "status":
        sys.stdout.buffer.write(canonical(store.manifest(args.stream, args.version)) + b"\n")
    elif args.command == "full":
        sys.stdout.buffer.write(canonical(store.ensure_full(args.stream, args.version)) + b"\n")
    elif args.command == "collect":
        sys.stdout.buffer.write(canonical(store.collect(retain_anchors=args.retain_anchors)) + b"\n")
    elif args.command == "register":
        sys.stdout.buffer.write(
            canonical(store.register(args.consumer, args.stream, args.start_version, mode=args.mode)) + b"\n"
        )
    elif args.command == "acknowledge":
        result = store.acknowledge(args.consumer, json.loads(Path(args.receipt).read_bytes()))
        sys.stdout.buffer.write(canonical(result) + b"\n")
    elif args.command == "unregister":
        store.unregister(args.consumer)
    elif args.command == "list":
        sys.stdout.buffer.write(canonical(store.versions(args.stream, after=args.after, limit=args.limit)) + b"\n")
    elif args.command == "pin":
        store.pin(args.stream, args.version, args.owner)
    elif args.command == "unpin":
        store.unpin(args.owner)
    elif args.command == "seal":
        from .seal import seal_versions

        sys.stdout.buffer.write(canonical(seal_versions(store, args.stream, args.versions, args.name)) + b"\n")
    elif args.command == "release-seal":
        from .seal import release_seal

        release_seal(store, args.name)
    elif args.command == "consume":
        from .cli_consume import consume_versions

        consume_versions(store, args.runtime_config)
    elif args.command == "fetch":
        if args.source_store:
            source = VersionStore(
                args.source_store, disk_budget=args.disk_budget, max_weight_bytes=args.max_weight_bytes
            )
            transport = SharedFSTransport(source)
        else:
            if not args.port:
                parser.error("--host requires --port")
            transport = TCPTransport((args.host, args.port), control_timeout=args.control_timeout)
        manifest = fetch(transport, store, args.stream, args.version, full=args.full)
        path = write_checkpoint(
            store, args.stream, args.version, Path(args.checkpoint_output), config_source=args.config
        )
        sys.stdout.buffer.write(
            canonical(
                {
                    "checkpoint": str(path),
                    "root": manifest["root"],
                    "received_bytes": transport.bytes_received,
                    "installed": False,
                }
            )
            + b"\n"
        )


if __name__ == "__main__":
    main()
