# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Independent CPU publication/consumption tools; no Ray process is
required."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from relax.utils.logging_utils import get_logger

from .config import WeightSyncConfig
from .consumer import Consumer
from .format import WeightSyncError, digest, parse_json
from .publisher import Publisher
from .store import SharedFSStore, atomic_write
from .tcp import ObjectServer, TCPTransport


logger = get_logger(__name__)


def checkpoint_tensors(path: Path):
    from safetensors import safe_open

    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = parse_json(index_path.read_bytes())["weight_map"]
        filenames = sorted(set(weight_map.values()))
    else:
        filenames = [file.name for file in sorted(path.glob("*.safetensors"))]
        weight_map = None
    if not filenames:
        raise WeightSyncError("No safetensors checkpoint found")
    seen = set()
    for filename in filenames:
        if Path(filename).name != filename:
            raise WeightSyncError("Invalid checkpoint shard name")
        with safe_open(path / filename, framework="pt", device="cpu") as source:
            for name in source.keys():
                if weight_map is not None and weight_map.get(name) != filename:
                    raise WeightSyncError("Checkpoint index and shard disagree")
                seen.add(name)
                yield name, source.get_tensor(name)
    if weight_map is not None and seen != set(weight_map):
        raise WeightSyncError("Checkpoint index names missing from safetensors shards")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="POSIX content store directory")
    parser.add_argument("--canonical-root", help="Publisher-local persistent recovery directory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish", help="Publish a frozen HF safetensors checkpoint")
    publish.add_argument("--config", required=True)
    publish.add_argument("--checkpoint", type=Path, required=True)
    publish.add_argument("--update-id", required=True)
    publish.add_argument("--result", type=Path, required=True)
    prepare = subparsers.add_parser("prepare", help="Fetch and reconstruct without a training process")
    prepare.add_argument("--stream", required=True)
    prepare.add_argument("--version", type=int, required=True)
    prepare.add_argument("--cache", required=True)
    prepare.add_argument("--checkpoints", required=True)
    prepare.add_argument("--endpoint", help="Optional tcp://host:port; otherwise SharedFS")
    prepare.add_argument("--full", action="store_true")
    prepare.add_argument("--timeout", type=float, default=600)
    prepare.add_argument("--disk-budget-bytes", type=int, required=True)
    prepare.add_argument("--result", type=Path, required=True)
    query = subparsers.add_parser("query")
    query.add_argument("--result", type=Path, required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--stream", required=True)
    seal.add_argument("--versions", type=int, nargs="+", required=True)
    seal.add_argument("--seal-id", required=True)
    release = subparsers.add_parser("release")
    release.add_argument("--reference", required=True)
    subparsers.add_parser("gc")
    serve = subparsers.add_parser("serve", help="Serve content on a trusted network or authenticated tunnel")
    serve.add_argument("--host", required=True)
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    store = SharedFSStore(args.store, canonical_root=args.canonical_root)
    if args.command == "publish":
        config = WeightSyncConfig.load(args.config)
        if config.mode != "delta" or Path(config.store_root).resolve() != store.root:
            raise WeightSyncError("Publish requires delta config bound to --store")
        store = SharedFSStore(config.store_root, config.disk_budget_bytes, canonical_root=config.canonical_root)
        model_config = parse_json((args.checkpoint / "config.json").read_bytes())
        schema = {"model_config": model_config, "exporter": "hf-safetensors-v1"}
        pub = Publisher(store, config.stream, schema, **config.publisher_kwargs())
        try:
            result = pub.finalize_pending(
                pub.submit(args.update_id, checkpoint_tensors(args.checkpoint)), config.sync_timeout_s
            )
            output = asdict(result) | {"metrics": pub.last_metrics}
            atomic_write(args.result, json.dumps(output, ensure_ascii=False, indent=2).encode())
            logger.info("Published %s/%s root=%s", result.stream, result.version, result.root)
        finally:
            pub.close()
    elif args.command == "prepare":
        transport = store
        if args.endpoint:
            endpoint = urlparse(args.endpoint)
            if endpoint.scheme != "tcp" or not endpoint.hostname or endpoint.port is None:
                raise WeightSyncError("Expected tcp://host:port endpoint")
            transport = TCPTransport(endpoint.hostname, endpoint.port, timeout_s=args.timeout)
        consumer = Consumer(
            transport,
            SharedFSStore(args.cache, args.disk_budget_bytes),
            args.checkpoints,
            disk_budget_bytes=args.disk_budget_bytes,
            timeout_s=args.timeout,
        )
        result = consumer.prepare(args.stream, args.version, full=args.full)
        atomic_write(args.result, json.dumps(asdict(result) | {"metrics": consumer.last_metrics}, indent=2).encode())
        logger.info("Prepared %s/%s at %s", result.stream, result.version, result.path)
    elif args.command == "query":
        atomic_write(args.result, json.dumps(store.inventory(), ensure_ascii=False, indent=2).encode())
    elif args.command == "seal":
        store.seal(args.seal_id, args.stream, args.versions)
        logger.info("Sealed %s, identity=%s", args.seal_id, digest(args.seal_id.encode()))
    elif args.command == "release":
        store.release(args.reference)
    elif args.command == "gc":
        logger.info("Collected %s", store.collect())
    elif args.command == "serve":
        with ObjectServer((args.host, args.port), store, workers=args.workers) as server:
            logger.info("Serving weight objects on %s:%s", *server.server_address)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                return


if __name__ == "__main__":
    main()
