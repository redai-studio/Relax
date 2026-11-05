#!/usr/bin/env python3
"""Time FULL/DELTA delivery of identical frozen HF targets without audit I/O.

Both arms use the real TCP consumer and ordinary SGLang disk loader. File
checks follow both timed arms; initial/final GPU dumps are compared separately.
This measures a replay segment, not the audited product or training E2E.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from urllib.parse import urlparse

import ray
from gpu_replay import CheckedEngine, SourceServer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from relax.distributed.weight_sync.config import WeightSyncConfig
from relax.distributed.weight_sync.consumer import Consumer
from relax.distributed.weight_sync.provenance import runtime_identity
from relax.distributed.weight_sync.store import SharedFSStore, atomic_write
from relax.distributed.weight_sync.tcp import TCPTransport
from relax.distributed.weight_sync.tensor_diff import compare


def load_engine(actor, target: str, version: int, timeout: float) -> dict:
    started = monotonic()
    response = actor.engine._make_request(
        "update_weights_from_disk",
        {
            "model_path": target,
            "load_format": "auto",
            "weight_version": str(version),
            "keep_pause": True,
            "flush_cache": True,
            "recapture_cuda_graph": False,
        },
        timeout=timeout,
    )
    if not isinstance(response, dict) or response.get("success") is not True:
        raise RuntimeError(f"Ordinary disk loader failed: {response}")
    return {"load_s": monotonic() - started, "response": response}


def snapshot_engine(actor, destination: str) -> str:
    import torch
    from sglang.srt.utils import MultiprocessingSerializer

    from relax.backends.sglang.disk_weight_dump import DUMP_LOADER

    path = Path(destination) / f"engine-{actor.engine.rank}"
    path.mkdir(parents=True, exist_ok=False)
    serialized = MultiprocessingSerializer.serialize([(str(path), torch.empty(0))], output_str=True)
    result = actor.engine._make_request(
        "update_weights_from_tensor",
        {
            "serialized_named_tensors": [serialized],
            "load_format": DUMP_LOADER,
            "flush_cache": False,
            "selector": "target",
        },
        timeout=180,
    )
    if not isinstance(result, dict) or result.get("success") is not True:
        raise RuntimeError(f"Read-only GPU dump failed: {result}")
    return str(path / "rank-0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument("--delta-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-node-ip", required=True)
    parser.add_argument("--consumer-node-ip", required=True)
    parser.add_argument("--updates", type=int, default=10)
    args = parser.parse_args()
    if args.source_node_ip == args.consumer_node_ip or not 10 <= args.updates <= 20:
        raise ValueError("Use two different nodes and 10–20 updates")
    args.output.mkdir(parents=True, exist_ok=False)
    ray.init(address="auto", log_to_driver=True)
    nodes = {n["NodeManagerAddress"]: n["NodeID"] for n in ray.nodes() if n["Alive"]}
    if ray.util.get_node_ip_address() != args.consumer_node_ip:
        raise ValueError("Submit on the consumer node")
    config_paths = {"FULL": args.full_config, "DELTA": args.delta_config}
    configs = {name: WeightSyncConfig.load(path) for name, path in config_paths.items()}
    if not configs["FULL"].full_only or configs["DELTA"].full_only:
        raise ValueError("Use independent primary FULL and DELTA publishers")
    sources, engines, consumers, transports = {}, [], {}, {}
    result = {"status": "STARTING", "runtime": runtime_identity(), "rows": []}
    cleanup = {"errors": []}

    def save() -> None:
        atomic_write(args.output / "results.json", json.dumps(result, indent=2).encode())

    save()
    try:
        for arm, path in config_paths.items():
            sources[arm] = SourceServer.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[args.source_node_ip], soft=False)
            ).remote(str(path))
            identity = ray.get(sources[arm].identity.remote(), timeout=60)
            atomic_write(args.output / f"source-{arm}.json", json.dumps(identity, indent=2).encode())
            endpoint = urlparse(identity["endpoint"])
            config = configs[arm]
            transports[arm] = TCPTransport(endpoint.hostname, endpoint.port, timeout_s=config.sync_timeout_s)
            cache = SharedFSStore(args.output / arm / "cache", config.disk_budget_bytes)
            consumers[arm] = Consumer(
                transports[arm],
                cache,
                args.output / arm / "checkpoints",
                disk_budget_bytes=config.disk_budget_bytes,
                timeout_s=config.sync_timeout_s,
            )
        engines = [
            CheckedEngine.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[args.consumer_node_ip], soft=False)
            ).remote(rank, args.model, str(args.output))
            for rank in range(4)
        ]
        ray.get([e.pause.remote() for e in engines], timeout=600)
        ray.get([e.resume.remote() for e in engines], timeout=60)
        result["status"] = "RUNNING"
        save()
        for version in range(args.updates + 1):
            order = ["FULL", "DELTA"] if version % 2 == 0 else ["DELTA", "FULL"]
            rows = []
            for position, arm in enumerate(order):
                consumer, transport, config = consumers[arm], transports[arm], configs[arm]
                before = dict(transport.metrics)
                started = monotonic()
                prepared = consumer.prepare(config.stream, version)
                kind = consumer.cache.manifest(prepared.artifact_id)["kind"]
                expected_kind = "FULL" if arm == "FULL" or version % config.anchor_interval == 0 else "DELTA"
                fallback = consumer.last_metrics.get("fallback_reason")
                if kind != expected_kind or fallback:
                    raise RuntimeError(f"Unexpected {arm} delivery: {kind}, {fallback}")
                prepared_at = monotonic()
                ray.get([e.pause.remote() for e in engines], timeout=60)
                paused_at = monotonic()
                loaded = ray.get(
                    [e.__ray_call__.remote(load_engine, prepared.path, version, 300) for e in engines], timeout=360
                )
                loaded_at = monotonic()
                ray.get([e.resume.remote() for e in engines], timeout=60)
                ray.get([e.probe.remote() for e in engines], timeout=60)
                finished = monotonic()
                release_started = monotonic()
                consumer.release(prepared)
                release_s = monotonic() - release_started
                row = asdict(prepared) | {
                    "arm": arm,
                    "order_position": position,
                    "version": version,
                    "artifact_kind": kind,
                    "source_full_only": config.full_only,
                    "request_full": False,
                    "prepare_s": prepared_at - started,
                    "pause_calls_s": paused_at - prepared_at,
                    "group_load_s": loaded_at - paused_at,
                    "resume_probe_s": finished - loaded_at,
                    "generation_pause_s": finished - prepared_at,
                    "sync_segment_s": finished - started,
                    "release_s": release_s,
                    "members": loaded,
                    "consumer_metrics": dict(consumer.last_metrics),
                    "wire_delta": {k: v - before.get(k, 0) for k, v in transport.metrics.items()},
                }
                rows.append(row)
                result["rows"].append(row)
                save()
            if rows[0]["root"] != rows[1]["root"]:
                raise RuntimeError("Arms used different frozen targets")
            for row in rows:
                audit_at = monotonic()
                reference = config_paths[row["arm"]].parent / "exports" / f"update-{version:06d}"
                check = compare(reference, Path(row["path"]))
                diff = args.output / f"hf-v{version}-{row['arm']}.json"
                atomic_write(diff, json.dumps(check, indent=2).encode())
                row.update(hf_equal=check["equal"], hf_diff=str(diff), hf_audit_s=monotonic() - audit_at)
                if not check["equal"]:
                    raise RuntimeError("HF bitwise comparison failed")
            if version in {0, args.updates}:
                # Separate audit pass after both timed arms; its reloads/dumps
                # never intervene between the paired timing measurements.
                audit_started = monotonic()
                dumps = {}
                ray.get([e.pause.remote() for e in engines], timeout=60)
                for arm in ["FULL", "DELTA"]:
                    row = next(row for row in rows if row["arm"] == arm)
                    ray.get(
                        [e.__ray_call__.remote(load_engine, row["path"], version, 300) for e in engines], timeout=360
                    )
                    dumps[arm] = ray.get(
                        [
                            e.__ray_call__.remote(snapshot_engine, str(args.output / "gpu" / f"v{version}-{arm}"))
                            for e in engines
                        ],
                        timeout=240,
                    )
                for rank, (full, delta) in enumerate(zip(dumps["FULL"], dumps["DELTA"], strict=True)):
                    check = compare(Path(full), Path(delta))
                    if json.loads((Path(full) / "coverage.json").read_text()) != json.loads(
                        (Path(delta) / "coverage.json").read_text()
                    ):
                        raise RuntimeError("GPU persistent coverage or aliases differ")
                    atomic_write(
                        args.output / f"gpu-v{version}-engine-{rank}.json", json.dumps(check, indent=2).encode()
                    )
                    if not check["equal"]:
                        raise RuntimeError("GPU bitwise comparison failed")
                ray.get([e.resume.remote() for e in engines], timeout=60)
                ray.get([e.probe.remote() for e in engines], timeout=60)
                for row in rows:
                    row["separate_gpu_audit_s"] = monotonic() - audit_started
            save()
        result["status"] = "COMPLETE"
        save()
        return 0
    except Exception:
        result["status"] = "FAILED"
        save()
        for engine in engines:
            try:
                ray.get(engine.pause.remote(), timeout=10)
            except Exception:
                try:
                    ray.kill(engine, no_restart=True)
                except Exception as error:
                    cleanup["errors"].append(str(error))
        raise
    finally:
        original_error = sys.exc_info()[0]
        for arm, source in sources.items():
            try:
                final = ray.get(source.identity.remote(), timeout=30)
                atomic_write(args.output / f"source-{arm}-final.json", json.dumps(final, indent=2).encode())
            except Exception as error:
                cleanup["errors"].append(f"source identity {arm}: {error}")
        for actor in [*engines, *sources.values()]:
            try:
                ray.get(actor.shutdown.remote(), timeout=30)
            except Exception as error:
                cleanup["errors"].append(str(error))
            finally:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as error:
                    cleanup["errors"].append(str(error))
        atomic_write(args.output / "cleanup.json", json.dumps(cleanup, indent=2).encode())
        ray.shutdown()
        if cleanup["errors"] and original_error is None:
            result["status"] = "FAILED_CLEANUP"
            save()
            raise RuntimeError(f"Benchmark cleanup failed: {cleanup['errors']}")


if __name__ == "__main__":
    raise SystemExit(main())
