#!/usr/bin/env python3
"""Load frozen replay checkpoints with the real SGLang disk loader and GPU
audit.

Run as a bounded Ray Job. GPU engines are pinned to the requested consumer
node; the optional object server is pinned to a different source node. The CPU
replay driver uses TCPTransport exclusively for protocol
records/manifests/objects.
"""

import argparse
import json
import os
import socket
import sys
import threading
from argparse import Namespace
from pathlib import Path
from time import monotonic

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from relax.distributed.weight_sync.config import WeightSyncConfig
from relax.distributed.weight_sync.store import SharedFSStore, atomic_write
from relax.distributed.weight_sync.tcp import ObjectServer


@ray.remote(num_cpus=1, num_gpus=1)
class CheckedEngine:
    """Reuse the product engine, parent-death guard, loader and audit hook."""

    def __init__(self, rank: int, model: str, output: str) -> None:
        from relax.backends.sglang.disk_weight_dump import DUMP_LOADER
        from relax.backends.sglang.sglang_engine import SGLangEngine
        from relax.distributed.weight_sync.provenance import runtime_identity

        root = Path(output) / f"engine-{rank}"
        root.mkdir(parents=True, exist_ok=False)
        os.chdir(root)
        self.root = root
        self.engine = SGLangEngine(
            Namespace(rollout_external=False, fully_async=False, rollout_num_gpus_per_engine=1),
            rank,
            base_gpu_id=0,
            num_gpus_per_engine=1,
        )
        host = ray.util.get_node_ip_address()
        with socket.socket() as listener:
            listener.bind((host, 0))
            port = listener.getsockname()[1]
        self.engine.server_host, self.engine.server_port = host, port
        self.engine.node_rank = 0
        self.engine.router_ip, self.engine.router_port = None, None
        self.engine._skip_router_registration = True
        server_args = {
            "model_path": model,
            "host": host,
            "port": port,
            "tp_size": 1,
            "pp_size": 1,
            "dp_size": 1,
            "dtype": "auto",
            "base_gpu_id": 0,
            "mem_fraction_static": 0.35,
            "disable_cuda_graph": True,
            "trust_remote_code": True,
            "skip_server_warmup": True,
            "custom_weight_loader": [DUMP_LOADER],
            "file_storage_path": str(root / "storage"),
            "download_dir": str(root / "downloads"),
        }
        atomic_write(root / "launch.json", json.dumps(server_args, indent=2).encode())
        self.engine._init_normal(server_args, apply_policy_load_plan=False)
        import requests

        info = requests.get(f"http://{host}:{port}/get_server_info", timeout=10)
        info.raise_for_status()
        atomic_write(root / "server-info.json", json.dumps(info.json(), indent=2).encode())
        atomic_write(root / "runtime.json", json.dumps(runtime_identity(), indent=2).encode())

    def evaluate(self, target: dict, published: dict, timeout: float) -> dict:
        from relax.backends.sglang.disk_weight_sync import install

        started = monotonic()
        events = []
        self.engine.pause_generation(timeout=10)
        events.append("PAUSED")
        try:
            result = install(self.engine, target, published, timeout)
            events.append("VERIFIED")
            result["paused_install_s"] = monotonic() - started
            result["events"] = events
            return result
        except Exception:
            self.engine.pause_generation(timeout=10)
            events.append("FAILED_PAUSED")
            atomic_write(self.root / "failure-events.json", json.dumps(events).encode())
            raise

    def pause(self) -> None:
        self.engine.pause_generation(timeout=10)

    def resume(self) -> None:
        self.engine.continue_generation(timeout=10)

    def probe(self) -> None:
        self.engine.disk_weight_generation_probe(30)

    def shutdown(self) -> None:
        # This harness has no router; shutdown's process-tree cleanup is reused.
        self.engine.shutdown()


@ray.remote(num_cpus=0.25)
class SourceServer:
    def __init__(self, config_path: str) -> None:
        config = WeightSyncConfig.load(config_path)
        self.server = ObjectServer(
            (ray.util.get_node_ip_address(), 0),
            SharedFSStore(config.store_root, config.disk_budget_bytes, canonical_root=config.canonical_root),
            request_timeout_s=config.sync_timeout_s,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def identity(self) -> dict:
        return {
            "endpoint": "tcp://%s:%s" % self.server.server_address,
            "node_id": ray.get_runtime_context().get_node_id(),
            "node_ip": ray.util.get_node_ip_address(),
        }

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--consumer-node-ip", required=True)
    parser.add_argument("--source-node-ip", help="A different node; enables TCP")
    parser.add_argument("--gpu-count", type=int, default=4, choices=[1, 2, 4])
    parser.add_argument("--versions", type=int, nargs="+", required=True)
    parser.add_argument("--full-only", action="store_true")
    args = parser.parse_args()
    if args.source_node_ip == args.consumer_node_ip:
        raise ValueError("TCP requires two different nodes")
    args.output.mkdir(parents=True, exist_ok=False)
    ray.init(address="auto", log_to_driver=True)
    nodes = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes() if node["Alive"]}
    if ray.util.get_node_ip_address() != args.consumer_node_ip:
        raise ValueError("Submit the replay driver from the consumer node")
    engines, source = [], None
    try:
        endpoint = None
        if args.source_node_ip:
            source = SourceServer.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[args.source_node_ip], soft=False)
            ).remote(str(args.config))
            identity = ray.get(source.identity.remote(), timeout=60)
            endpoint = identity["endpoint"]
            atomic_write(args.output / "source.json", json.dumps(identity, indent=2).encode())
        engines = [
            CheckedEngine.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[args.consumer_node_ip], soft=False)
            ).remote(rank, args.model, str(args.output))
            for rank in range(args.gpu_count)
        ]
        # The existing streaming HF replay supplies each frozen target and its
        # independent ordinary FULL export. GPU loading remains outside prepare.
        from replay_sequence import replay

        def installed(row: dict) -> None:
            started = monotonic()
            target = {
                "path": row["path"],
                "full_reference": row["reference"],
                "node": args.consumer_node_ip,
            }
            try:
                checked = ray.get([engine.evaluate.remote(target, row, 900) for engine in engines], timeout=1200)
                if any(item["status"] != "VERIFIED" or item["root"] != row["root"] for item in checked):
                    raise RuntimeError("A GPU member did not verify the requested target")
                ray.get([engine.resume.remote() for engine in engines], timeout=30)
                ray.get([engine.probe.remote() for engine in engines], timeout=45)
                for item in checked:
                    item["events"].extend(["CONTINUED", "PROBED"])
                row["gpu"] = checked
                row["group_pause_and_probe_s"] = monotonic() - started
            except Exception:
                for engine in engines:
                    try:
                        ray.get(engine.pause.remote(), timeout=15)
                    except Exception:
                        ray.kill(engine, no_restart=True)
                raise

        return replay(args.config, args.output / "receiver", args.versions, args.full_only, endpoint, installed)
    finally:
        original_error = sys.exc_info()[0]
        cleanup_errors = []
        for engine in engines:
            try:
                ray.get(engine.shutdown.remote(), timeout=30)
            except Exception as error:
                cleanup_errors.append(str(error))
            finally:
                ray.kill(engine, no_restart=True)
        if source:
            try:
                ray.get(source.shutdown.remote(), timeout=35)
            except Exception as error:
                cleanup_errors.append(str(error))
            finally:
                ray.kill(source, no_restart=True)
        atomic_write(args.output / "cleanup.json", json.dumps({"errors": cleanup_errors}).encode())
        ray.shutdown()
        if cleanup_errors and original_error is None:
            raise RuntimeError(f"Replay cleanup failed: {cleanup_errors}")


if __name__ == "__main__":
    raise SystemExit(main())
