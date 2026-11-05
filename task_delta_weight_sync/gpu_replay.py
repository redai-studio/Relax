#!/usr/bin/env python3
"""Load frozen replay checkpoints with the real SGLang disk loader and GPU
audit.

Run as a bounded Ray Job. GPU engines are pinned to the requested consumer
node; the optional object server is pinned to a different source node. The CPU
replay driver uses TCPTransport exclusively for protocol
records/manifests/objects.
"""

import argparse
import concurrent.futures
import json
import os
import socket
import sys
import threading
import uuid
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

    def reject_bad_load(self, target: dict, published: dict) -> dict:
        import requests

        try:
            self.evaluate(target, published, 60)
        except requests.HTTPError as error:
            response = error.response
            body = response.json() if response is not None else {}
            if response is None or response.status_code != 400 or body.get("success") is not False:
                raise
            if "Failed to get weights iterator" not in body.get("message", ""):
                raise
            return {"status": "REJECTED_PAUSED", "http_status": response.status_code, "response": body}
        raise AssertionError("The real loader accepted a checkpoint without weights")

    def observe_paused_generation(self) -> dict:
        import requests

        self.pause()
        url = f"http://{self.engine.server_host}:{self.engine.server_port}"
        # /health generates a token in this SGLang runtime and waits while paused.
        control_endpoint = "/model_info"
        control = requests.get(url + control_endpoint, timeout=5)
        control.raise_for_status()
        request_id = uuid.uuid4().hex
        payload = {"rid": request_id, "input_ids": [1], "sampling_params": {"max_new_tokens": 1, "temperature": 0}}
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(requests.post, url + "/generate", json=payload, timeout=15)
            observed_at = monotonic()
            try:
                try:
                    response = future.result(timeout=3)
                except concurrent.futures.TimeoutError:
                    pass
                else:
                    raise AssertionError(f"Paused generation completed: {response.status_code} {response.text[:500]}")
            finally:
                # The full unique rid avoids SGLang's prefix-abort affecting
                # any other request. Keep the original HTTP request until done.
                aborted = requests.post(
                    url + "/abort_request", json={"rid": request_id, "abort_all": False}, timeout=5
                )
                aborted.raise_for_status()
            response = future.result(timeout=8)
        body = response.json()
        meta = body.get("meta_info", {})
        if (
            meta.get("completion_tokens", 0)
            or meta.get("output_token_ids")
            or body.get("text")
            or body.get("output_ids")
        ):
            raise AssertionError(f"Paused probe produced a token: {body}")
        message = body.get("error", {}).get("message", "")
        if response.status_code != 499 or request_id not in message or "was aborted" not in message:
            raise AssertionError(f"Paused probe did not terminate as aborted: {body}")
        info = requests.get(url + "/model_info", timeout=5)
        info.raise_for_status()
        result = {
            "status": "PAUSED_NO_TOKEN_ABORTED",
            "rid": request_id,
            "observation_window_s": 3,
            "observation_and_abort_s": monotonic() - observed_at,
            "control_endpoint": control_endpoint,
            "control_status": control.status_code,
            "abort_status": aborted.status_code,
            "generate_status": response.status_code,
            "generate_response": body,
            "model_info": info.json(),
        }
        atomic_write(self.root / "paused-probe.json", json.dumps(result, indent=2).encode())
        return result

    def shutdown(self) -> None:
        # This harness has no router; shutdown's process-tree cleanup is reused.
        self.engine.shutdown()


@ray.remote(num_cpus=0.25)
class SourceServer:
    def __init__(self, config_path: str) -> None:
        config = WeightSyncConfig.load(config_path)
        self.server = ObservedObjectServer(
            (ray.util.get_node_ip_address(), 0),
            SharedFSStore(config.store_root, config.disk_budget_bytes, canonical_root=config.canonical_root),
            request_timeout_s=config.sync_timeout_s,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def identity(self) -> dict:
        from relax.distributed.weight_sync.provenance import runtime_identity

        with self.server.observation_lock:
            observed = {
                "accepted_connections": self.server.accepted_connections,
                "peer_ips": sorted(self.server.peer_ips),
            }
        return {
            "endpoint": "tcp://%s:%s" % self.server.server_address,
            "node_id": ray.get_runtime_context().get_node_id(),
            "actor_id": ray.get_runtime_context().get_actor_id(),
            "creator_job_id": ray.get_runtime_context().get_job_id(),
            "node_ip": ray.util.get_node_ip_address(),
            "runtime": runtime_identity(),
            **observed,
        }

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


class ObservedObjectServer(ObjectServer):
    """Record actual accepted peers without changing the object protocol."""

    def __init__(self, *args, **kwargs) -> None:
        self.observation_lock = threading.Lock()
        self.accepted_connections = 0
        self.peer_ips = set()
        super().__init__(*args, **kwargs)

    def process_request(self, request: socket.socket, client_address: tuple) -> None:
        with self.observation_lock:
            self.accepted_connections += 1
            self.peer_ips.add(client_address[0])
        super().process_request(request, client_address)


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
    parser.add_argument("--check-load-failure", action="store_true")
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

        last_row = None

        def installed(row: dict) -> None:
            nonlocal last_row
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
                resume_started = monotonic()
                ray.get([engine.resume.remote() for engine in engines], timeout=30)
                ray.get([engine.probe.remote() for engine in engines], timeout=45)
                row["resume_and_probe_s"] = monotonic() - resume_started
                for item in checked:
                    item["events"].extend(["CONTINUED", "PROBED"])
                row["gpu"] = checked
                row["group_pause_and_probe_s"] = monotonic() - started
                last_row = row
            except Exception:
                for engine in engines:
                    try:
                        ray.get(engine.pause.remote(), timeout=15)
                    except Exception:
                        ray.kill(engine, no_restart=True)
                raise

        result = replay(args.config, args.output / "receiver", args.versions, args.full_only, endpoint, installed)
        if result == 0 and args.check_load_failure:
            fault_started = monotonic()
            stable = dict(last_row)
            empty = args.output / "fault" / "empty"
            empty.mkdir(parents=True)
            ray.get([engine.pause.remote() for engine in engines], timeout=30)
            bad_target = {"path": str(empty), "full_reference": stable["reference"], "node": args.consumer_node_ip}
            attempt = stable | {"version": stable["version"] + 1}
            rejected = ray.get(engines[0].reject_bad_load.remote(bad_target, attempt), timeout=90)
            # A failed member never opens the group. Confirm each member's
            # actual gate while its health/control interface remains responsive.
            ray.get([engine.pause.remote() for engine in engines], timeout=30)
            probes = ray.get([engine.observe_paused_generation.remote() for engine in engines], timeout=45)
            for probe in probes:
                if str(probe["model_info"].get("weight_version")) != str(stable["version"]):
                    raise AssertionError(f"Rejected attempt changed the installed version: {probe}")
            fault = {
                "status": "REJECTED_CLOSED",
                "attempted_version": attempt["version"],
                "stable_version": stable["version"],
                "stable_root": stable["root"],
                "bad_path": str(empty),
                "rejected": rejected,
                "probes": probes,
                "resume_calls_before_recovery": 0,
                "negative_stage_s": monotonic() - fault_started,
            }
            atomic_write(args.output / "fault-recovery.json", json.dumps(fault, indent=2).encode())
            # The runner's model_path changed on rejection. Explicitly reload
            # and audit the good frozen target before the next group resume.
            installed(stable)
            fault.update(
                status="RECOVERED",
                restored_root=stable["root"],
                gpu=stable["gpu"],
                total_pause_and_recovery_s=monotonic() - fault_started,
            )
            atomic_write(args.output / "fault-recovery.json", json.dumps(fault, indent=2).encode())
        return result
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
                identity = ray.get(source.identity.remote(), timeout=10)
                atomic_write(args.output / "source-final.json", json.dumps(identity, indent=2).encode())
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
