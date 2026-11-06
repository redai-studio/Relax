#!/usr/bin/env python3
"""CPU-only filesystem profiling of an existing frozen weight sequence.

Inputs remain unchanged. Independent stores/cache prefixes are seeded with
hardlinks to immutable input objects; all mutable metadata is freshly written.
File API wall time includes filesystem/cache/remote-storage waits, not solely
physical disk service time. Socket waits include the peer's processing too.
"""

import argparse
import copy
import cProfile
import json
import os
import pstats
import threading
from dataclasses import asdict
from pathlib import Path
from time import monotonic, process_time
from typing import Any, Callable

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from relax.distributed.weight_sync.config import WeightSyncConfig
from relax.distributed.weight_sync.consumer import Consumer
from relax.distributed.weight_sync.format import canonical_json, validate_manifest
from relax.distributed.weight_sync.provenance import runtime_identity
from relax.distributed.weight_sync.publisher import Publisher
from relax.distributed.weight_sync.store import SharedFSStore, atomic_write
from relax.distributed.weight_sync.tcp import ObjectServer, TCPTransport
from relax.distributed.weight_sync.tensor_diff import compare


def category(filename: str, name: str) -> str:
    if filename != "~":
        return "other_python_and_compute"
    if "fsync" in name or "fdatasync" in name:
        return "file_sync"
    if "_io." in name or "_io " in name:
        if "read" in name:
            return "file_read"
        if any(word in name for word in ("write", "flush", "BufferedWriter")):
            return "file_write_flush_close"
        return "file_open_close_metadata"
    if "posix.read" in name and "readlink" not in name:
        return "file_read"
    if "posix.write" in name:
        return "file_write_flush_close"
    filesystem_calls = (
        ".stat",
        ".lstat",
        ".fstat",
        ".open",
        ".close",
        ".rename",
        ".replace",
        ".unlink",
        ".mkdir",
        ".scandir",
        ".listdir",
        ".readlink",
        ".chmod",
        ".access",
        ".utime",
        ".getcwd",
        ".rmdir",
    )
    if (
        ("posix." in name and any(call in name for call in filesystem_calls))
        or "DirEntry" in name
        or "ScandirIterator" in name
    ):
        return "file_open_close_metadata"
    if "flock" in name:
        return "file_lock"
    if "socket" in name or "select." in name:
        return "socket_calls_including_peer_wait"
    if "hashlib" in name or "HASH" in name:
        return "hash"
    if "zlib" in name:
        return "compression_decompression"
    return "other_python_and_compute"


def profiled(function: Callable[..., Any], destination: Path, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
    profile = cProfile.Profile()
    started, cpu_started = monotonic(), process_time()
    value = profile.runcall(function, *args, **kwargs)
    wall, cpu = monotonic() - started, process_time() - cpu_started
    stats = pstats.Stats(profile)
    buckets, calls = {}, []
    for (filename, line, name), (_, count, own, cumulative, _) in stats.stats.items():
        key = category(filename, name)
        buckets[key] = buckets.get(key, 0.0) + own
        calls.append(
            {
                "file": filename,
                "line": line,
                "function": name,
                "calls": count,
                "own_s": own,
                "cumulative_s": cumulative,
                "category": key,
            }
        )
    report = {
        "wall_s": wall,
        "process_cpu_s": cpu,
        "profile_total_s": stats.total_tt,
        "exclusive_categories_s": buckets,
        "functions": sorted(calls, key=lambda row: row["own_s"], reverse=True),
    }
    report_started = monotonic()
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile.dump_stats(str(destination.with_suffix(".pstats")))
    atomic_write(destination.with_suffix(".json"), json.dumps(report, indent=2).encode())
    summary = {key: data for key, data in report.items() if key != "functions"}
    summary["report_write_s_outside_profile"] = monotonic() - report_started
    return value, summary


def seed(original: WeightSyncConfig, destination: Path, through: int, *, canonical: bool) -> SharedFSStore:
    if destination.exists():
        raise FileExistsError(destination)
    source = SharedFSStore(original.store_root, canonical_root=original.canonical_root)
    target = SharedFSStore(destination / "store", original.disk_budget_bytes, canonical_root=destination / "canonical")
    state = {"format": 1, "streams": {original.stream: {}}, "references": {}, "consumers": {}, "seals": {}}
    objects, raw = set(), set()
    for version in range(through + 1):
        record = copy.deepcopy(source.record(original.stream, version))
        record["artifacts"] = {"FULL" if version == 0 else "DELTA": record["artifact_id"]}
        manifest = source.manifest(record["artifact_id"])
        state["streams"][original.stream][str(version)] = record
        os.link(source.root / "artifacts" / record["artifact_id"], target.root / "artifacts" / record["artifact_id"])
        objects.update(block["object"] for block in manifest["blocks"] if block["object"] is not None)
        if canonical:
            raw.update(block["digest"] for block in validate_manifest(manifest))
            for folder, filename in [("catalogs", manifest["root"]), ("manifests", record["artifact_id"])]:
                os.link(source.canonical_root / folder / filename, target.canonical_root / folder / filename)
    for object_id in objects:
        os.link(source.root / "objects" / object_id, target.root / "objects" / object_id)
    for block_id in raw:
        os.link(source.canonical_root / "blocks" / block_id, target.canonical_root / "blocks" / block_id)
    atomic_write(target.root / "index.json", canonical_json(state))
    return target


@ray.remote(num_cpus=1)
class ProfileSource:
    def __init__(self, config_path: str, frozen_path: str, output: str) -> None:
        self.original = WeightSyncConfig.load(config_path)
        self.frozen_path, self.output = frozen_path, Path(output)
        self.server = self.thread = None

    def publish_one(self) -> dict:
        from gpu_replay import publish_frozen_target

        store = seed(self.original, self.output / "publisher", 0, canonical=True)
        config = WeightSyncConfig.from_dict(
            asdict(self.original) | {"store_root": str(store.root), "canonical_root": str(store.canonical_root)}
        )
        original_method, measurements = Publisher._publish_snapshot, {}

        def measured(publisher: Publisher, *args: Any) -> Any:
            value, stats = profiled(original_method, self.output / "source-publish", publisher, *args)
            measurements.update(stats)
            return value

        Publisher._publish_snapshot = measured
        try:
            result = publish_frozen_target(store, config, self.frozen_path, 1)
        finally:
            Publisher._publish_snapshot = original_method
        return {
            "published": result,
            "profile": measurements,
            "scope": "Publisher._publish_snapshot worker only; HF reading and capture excluded; published.source_wall_s includes report_write_s_outside_profile",
            "runtime": runtime_identity(),
        }

    def start_server(self, version: int) -> dict:
        self.stop_server()
        if version == 1:
            store = SharedFSStore(self.output / "publisher/store", canonical_root=self.output / "publisher/canonical")
        else:
            store = seed(self.original, self.output / f"source-v{version}", version, canonical=False)
        self.server = ObjectServer((ray.util.get_node_ip_address(), 0), store)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return {
            "host": self.server.server_address[0],
            "port": self.server.server_address[1],
            "runtime": runtime_identity(),
        }

    def stop_server(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=5)
            self.server = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-node-ip", required=True)
    parser.add_argument("--consumer-node-ip", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = WeightSyncConfig.load(args.config)
    if config.full_only or args.source_node_ip == args.consumer_node_ip:
        raise ValueError("Use a DELTA store and two distinct nodes")
    ray.init(address="auto", log_to_driver=True)
    if ray.util.get_node_ip_address() != args.consumer_node_ip:
        raise ValueError("Submit the driver on the consumer node")
    nodes = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes() if node["Alive"]}
    source = ProfileSource.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[args.source_node_ip], soft=False)
    ).remote(str(args.config), str(args.frozen_config), str(args.output))
    result = {"status": "PROFILING_SOURCE", "runtime": runtime_identity(), "receivers": []}

    def save() -> None:
        atomic_write(args.output / "results.json", json.dumps(result, indent=2).encode())

    save()
    try:
        result["source"] = ray.get(source.publish_one.remote(), timeout=config.sync_timeout_s + 60)
        for version in [1, 10]:
            result["status"] = f"PROFILING_RECEIVER_{version}"
            save()
            cache = seed(config, args.output / f"receiver-v{version}/cache", version - 1, canonical=False)
            endpoint = ray.get(source.start_server.remote(version), timeout=180)
            transport = TCPTransport(endpoint["host"], endpoint["port"], timeout_s=config.sync_timeout_s)
            consumer = Consumer(
                transport,
                cache,
                args.output / f"receiver-v{version}/checkpoints",
                disk_budget_bytes=config.disk_budget_bytes,
                timeout_s=config.sync_timeout_s,
            )
            prepared, stats = profiled(consumer.prepare, args.output / f"receiver-v{version}", config.stream, version)
            expected = SharedFSStore(config.store_root, canonical_root=config.canonical_root).record(
                config.stream, version
            )
            if prepared.root != expected["root"] or consumer.last_metrics.get("fallback_reason"):
                raise RuntimeError("Wrong target or unexpected fallback")
            check = compare(args.frozen_config.parent / "exports" / f"update-{version:06d}", Path(prepared.path))
            atomic_write(args.output / f"hf-v{version}.json", json.dumps(check, indent=2).encode())
            if not check["equal"]:
                raise RuntimeError("HF raw-bit mismatch")
            consumer.release(prepared)
            result["receivers"].append(
                {
                    "version": version,
                    "profile": stats,
                    "consumer": consumer.last_metrics,
                    "wire": transport.metrics,
                    "endpoint": endpoint,
                    "hf_equal": True,
                    "root": prepared.root,
                }
            )
        result["status"] = "COMPLETE"
        save()
    except BaseException:
        result["status"] = "FAILED"
        save()
        raise
    finally:
        try:
            ray.get(source.stop_server.remote(), timeout=30)
        except BaseException:
            result["status"] = "FAILED_CLEANUP"
            save()
            raise
        finally:
            try:
                ray.kill(source, no_restart=True)
            finally:
                ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
