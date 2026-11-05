#!/usr/bin/env python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Cross-node TransferQueue C0/C1/C2 acceptance benchmark.

Invoke once per fresh process: ``simple`` is C0, benchmark-only Mooncake/TCP is
C1, and Mooncake/host-RDMA is C2.  The driver produces while one persistent
consumer is hard-pinned to another Ray node.  Dtype/shape/raw-byte digests and
receive-counter wire proof are mandatory; each measured round is flushed to CSV
before either gate can fail.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as importlib_metadata
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, TextIO

import ray

from relax.utils.tq.correctness import leaf_digests, payload_nbytes, payload_rows


PROTOCOL_LABELS = {
    "simple": "C0 SimpleStorage",
    "tcp": "C1 Mooncake/TCP",
    "rdma": "C2 Mooncake/RDMA",
}
CSV_COLUMNS = (
    "protocol profile payload_mib actual_mib run measured phase status error_kind cleanup_status cleanup_error_kind "
    "byte_exact mismatch_fields wire_proven "
    "put_ms get_ms round_ms put_gbs get_gbs ib_mb tcp_mb proof_ib_mb proof_tcp_mb idle_ib_mbps idle_tcp_mbps "
    "relax_sha tq_commit mooncake_version"
).split()
_TCP_WIRE_MIN_PAYLOAD_RATIO = 0.20
_RDMA_WIRE_MIN_PAYLOAD_RATIO = 0.80
_IDLE_COUNTER_SAMPLE_SECONDS = 0.5
_GLOB_MAGIC = frozenset("*?[")

CounterMap = dict[str, int]
DigestMap = dict[str, tuple[Any, ...]]


def _is_safe_device_name(value: str) -> bool:
    return (
        bool(value)
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and not any(character in _GLOB_MAGIC for character in value)
        and all(character.isprintable() and not character.isspace() for character in value)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-node TQ byte-exact and wire-proof benchmark")
    parser.add_argument("--protocol", required=True, choices=sorted(PROTOCOL_LABELS))
    parser.add_argument("--consumer-node-id", required=True, help="Alive Ray NodeID distinct from the driver node")
    parser.add_argument(
        "--master",
        default=None,
        help="Mooncake master host:port (defaults to MC_MASTER_ADDRESS for Mooncake protocols)",
    )
    parser.add_argument("--device", default="", help="RDMA device required by the C2 wire-proof counter")
    parser.add_argument("--rdma-port", type=int, default=None, help="RDMA port required by the C2 wire-proof counter")
    parser.add_argument("--tcp-device", required=True, help="Network interface used for the TCP receive counter")
    parser.add_argument(
        "--payload-profiles", nargs="+", default=["synthetic", "multimodal"], choices=["synthetic", "multimodal"]
    )
    parser.add_argument("--payload-mib", nargs="+", type=int, default=[256, 1024, 2048, 4096])
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--segment-gib", type=int, default=16)
    parser.add_argument("--csv", required=True, type=Path, help="New per-round CSV output path")
    args = parser.parse_args()
    for name in ("num_samples", "repeats", "segment_gib"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not args.payload_mib or any(size <= 0 for size in args.payload_mib):
        parser.error("--payload-mib values must be positive")
    if args.protocol == "simple":
        if args.master is not None:
            parser.error("--master is valid only with Mooncake protocols")
        args.master = ""
    else:
        args.master = args.master or os.environ.get("MC_MASTER_ADDRESS", "")
        if not args.master:
            parser.error("--master or MC_MASTER_ADDRESS is required for Mooncake protocols")
    if not _is_safe_device_name(args.tcp_device):
        parser.error("--tcp-device must be a single printable interface name without whitespace or path separators")
    if args.protocol == "rdma":
        if not _is_safe_device_name(args.device):
            parser.error("--device is required for RDMA and must be a single printable device name")
        if args.rdma_port is None or args.rdma_port <= 0:
            parser.error("--rdma-port is required for RDMA and must be positive")
    elif args.device or args.rdma_port is not None:
        parser.error("--device and --rdma-port are valid only with --protocol rdma")
    return args


def _columns_for_budget(total_bytes: int, num_samples: int, dtype: Any) -> int:
    import torch

    return max(1, total_bytes // (num_samples * torch.tensor([], dtype=dtype).element_size()))


def make_synthetic_payload(num_samples: int, total_mib: int) -> Any:
    """Deterministic mixed-dtype payload with a fixed schema."""
    import torch
    from tensordict import TensorDict

    generator = torch.Generator().manual_seed(20260824)
    field_budget = total_mib * 1024**2 // 3
    fp32_cols = _columns_for_budget(field_budget, num_samples, torch.float32)
    bf16_cols = _columns_for_budget(field_budget, num_samples, torch.bfloat16)
    int64_cols = _columns_for_budget(field_budget, num_samples, torch.int64)
    return TensorDict(
        {
            "activations": torch.randn(num_samples, fp32_cols, dtype=torch.float32, generator=generator),
            "logits": torch.randn(num_samples, bf16_cols, dtype=torch.bfloat16, generator=generator),
            "tokens": torch.randint(0, 151_000, (num_samples, int64_cols), dtype=torch.int64, generator=generator),
        },
        batch_size=[num_samples],
    )


def make_multimodal_payload(num_samples: int, total_mib: int) -> Any:
    """Deterministic production-shaped non-tensor vision-language payload."""
    import torch

    from relax.utils.utils import dict_to_tensordict

    target_bytes = total_mib * 1024**2
    seq_len = min(4096, max(128, target_bytes // max(1, num_samples * 128 * 1024)))
    fixed_bytes = num_samples * (seq_len * 8 + 8 + 4)
    pixel_budget = max(num_samples * 1536 * 4, target_bytes - fixed_bytes)
    base_patches = max(1, pixel_budget // (num_samples * 1536 * 4))
    generator = torch.Generator().manual_seed(20260824)
    multimodal_rows = []
    for index in range(num_samples):
        variation = (index % 3) - 1
        requested_patches = max(1, base_patches + variation * max(1, base_patches // 20))
        height = max(1, int(requested_patches**0.5))
        width = max(1, (requested_patches + height - 1) // height)
        patches = height * width
        pixel_values = torch.empty((patches, 1536), dtype=torch.float32)
        pixel_values.uniform_(-1.0, 1.0, generator=generator)
        multimodal_rows.append(
            {
                "pixel_values": pixel_values,
                "image_grid_thw": torch.tensor([[1, height, width]], dtype=torch.int64),
            }
        )
    token_row = list(range(seq_len))
    payload = dict_to_tensordict(
        {
            "tokens": [token_row for _ in range(num_samples)],
            "sample_id": list(range(num_samples)),
            "rewards": [float(index) / max(1, num_samples - 1) for index in range(num_samples)],
            "multimodal_train_inputs": multimodal_rows,
        },
        batch_size=num_samples,
    )
    # TQ records scalar columns as one-element rows. Match that wire shape
    # before computing expected digests, keeping dtype/shape/byte checks strict.
    for field in ("sample_id", "rewards"):
        payload.set(field, payload.get(field).unsqueeze(-1))
    if type(payload.get("multimodal_train_inputs")).__name__ != "NonTensorStack":
        raise RuntimeError("multimodal benchmark payload did not produce a NonTensorStack column")
    return payload


def field_byte_digests(payload: Any, fields: list[str]) -> DigestMap:
    """Ordered row digests preserving dtype, shape and every raw byte."""
    return {field: tuple(leaf_digests(row) for row in payload_rows(payload.get(field))) for field in fields}


def payload_bytes(payload: Any) -> int:
    return sum(payload_nbytes(payload.get(field)) for field in payload.keys())


def _read_required_counter(path: Path, label: str, *, scale: int = 1) -> int:
    try:
        value = int(path.read_text().strip()) * scale
    except (OSError, ValueError):
        raise RuntimeError(f"{label} receive counter is unavailable or unreadable") from None
    if value < 0:
        raise RuntimeError(f"{label} receive counter returned a negative value")
    return value


def read_counters(
    tcp_device: str,
    rdma_device: str = "",
    rdma_port: int | None = None,
    sysfs_root: Path = Path("/sys/class"),
) -> CounterMap:
    """Read exact receive-byte counters or fail closed.

    ``port_rcv_data`` is expressed in four-byte words.  RDMA device and port
    are an all-or-nothing pair so no glob or multi-port aggregation can enter
    an acceptance result.
    """
    if not _is_safe_device_name(tcp_device):
        raise ValueError("TCP counter selection requires one safe interface name")
    if bool(rdma_device) != (rdma_port is not None):
        raise ValueError("RDMA counter selection requires both device and port")
    if rdma_device and (not _is_safe_device_name(rdma_device) or rdma_port is None or rdma_port <= 0):
        raise ValueError("RDMA counter selection requires one safe device and positive port")

    counters: CounterMap = {}
    if rdma_device:
        ib_path = sysfs_root / "infiniband" / rdma_device / "ports" / str(rdma_port) / "counters" / "port_rcv_data"
        counters[f"ib:{rdma_device}:{rdma_port}"] = _read_required_counter(ib_path, "RDMA", scale=4)
    tcp_path = sysfs_root / "net" / tcp_device / "statistics" / "rx_bytes"
    counters[f"tcp:{tcp_device}"] = _read_required_counter(tcp_path, "TCP")
    return counters


def _counter_delta(before: CounterMap, after: CounterMap, prefix: str) -> int:
    before_keys = {key for key in before if key.startswith(prefix)}
    after_keys = {key for key in after if key.startswith(prefix)}
    if before_keys != after_keys:
        raise RuntimeError(f"{prefix.rstrip(':').upper()} counter set changed during the measured round")
    deltas = [after[key] - before[key] for key in sorted(before_keys)]
    if any(delta < 0 for delta in deltas):
        raise RuntimeError(f"{prefix.rstrip(':').upper()} counter reset or wrapped during the measured round")
    return sum(deltas)


def _subtract_idle_noise(raw_delta: int, idle_delta: int, idle_seconds: float, round_seconds: float) -> int:
    if idle_seconds <= 0 or round_seconds <= 0:
        raise RuntimeError("counter sampling durations must be positive")
    estimated_noise = math.ceil(idle_delta * round_seconds / idle_seconds)
    return max(0, raw_delta - estimated_noise)


def wire_is_proven(protocol: str, ib_bytes: int, tcp_bytes: int, payload_nbytes: int) -> bool:
    if protocol == "rdma":
        return ib_bytes >= _RDMA_WIRE_MIN_PAYLOAD_RATIO * payload_nbytes
    if protocol == "tcp":
        return tcp_bytes >= _TCP_WIRE_MIN_PAYLOAD_RATIO * payload_nbytes
    # SimpleStorage may place some units on the consumer node, so C0 cannot
    # require a payload-volume ratio. It must still produce cross-node TCP.
    return tcp_bytes > 0


def build_conf(protocol: str, master: str, device: str, segment_gib: int) -> Any:
    from omegaconf import OmegaConf
    from transfer_queue import GRPOGroupNSampler

    from relax.utils.tq.config import (
        build_mooncake_config,
        build_simple_storage_config,
        validate_mooncake_runtime_contract,
    )

    if protocol == "simple":
        backend = build_simple_storage_config(total_storage_size=None, num_data_storage_units=2)
    else:
        validate_mooncake_runtime_contract()
        backend = build_mooncake_config(
            master_address=master,
            device=device,
            protocol=protocol,
            global_segment_size=segment_gib * 1024**3,
        )
    return OmegaConf.create(
        {
            "controller": {"sampler": GRPOGroupNSampler(n_samples_per_prompt=1), "polling_mode": True},
            "backend": backend,
        },
        flags={"allow_objects": True},
    )


def _require_commit_id(value: Any, label: str) -> str:
    commit = value if isinstance(value, str) else ""
    if len(commit) != 40 or any(character not in "0123456789abcdefABCDEF" for character in commit):
        raise RuntimeError(f"{label} commit is unavailable or not a full SHA")
    return commit.lower()


def _git_output(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError("Relax commit provenance is unavailable") from None
    if result.returncode != 0:
        raise RuntimeError("Relax commit provenance is unavailable")
    return result.stdout.strip()


def collect_provenance(repo_root: Path | None = None) -> dict[str, str]:
    """Collect exact source/dependency versions for every CSV row.

    Acceptance runs require a clean tracked Relax checkout.  Untracked result
    files are ignored so the benchmark can write artifacts below the checkout.
    TransferQueue's PEP 610 metadata supplies the installed VCS commit.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    if _git_output(repo_root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("benchmark acceptance requires a clean tracked Relax checkout")
    revision = _git_output(repo_root, "rev-parse", "HEAD")

    distribution = importlib_metadata.distribution("transferqueue")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("installed TransferQueue has no VCS provenance metadata")
    try:
        direct_url = json.loads(direct_url_text)
        tq_commit = direct_url["vcs_info"]["commit_id"]
    except (KeyError, TypeError, json.JSONDecodeError):
        raise RuntimeError("installed TransferQueue VCS provenance metadata is invalid") from None
    try:
        mooncake_version = importlib_metadata.version("mooncake-transfer-engine")
    except importlib_metadata.PackageNotFoundError:
        mooncake_version = "not-installed"

    return {
        "relax_sha": _require_commit_id(revision, "Relax"),
        "tq_commit": _require_commit_id(tq_commit, "TransferQueue"),
        "mooncake_version": mooncake_version,
    }


@ray.remote(num_cpus=0.001, max_restarts=0)
class TQConsumer:
    """Persistent, hard-pinned consumer mirroring a Relax component actor."""

    def __init__(
        self,
        conf: Any,
        tcp_device: str,
        rdma_device: str,
        rdma_port: int | None,
        protocol: str,
    ) -> None:
        selected_device = rdma_device if protocol == "rdma" else ""
        selected_port = rdma_port if protocol == "rdma" else None
        read_counters(tcp_device, selected_device, selected_port)
        if protocol == "tcp":
            os.environ["MC_TCP_ENABLE_CONNECTION_POOL"] = "1"
        from relax.utils.tq.lifecycle import attach_tq_client

        # This runs inside the pinned Ray worker.  The bounded helper applies
        # Mooncake correctness guards in this process before client creation.
        self.client = attach_tq_client(conf, role="benchmark-consumer")
        self.tcp_device = tcp_device
        self.rdma_device = selected_device
        self.rdma_port = selected_port

    def describe(self) -> tuple[str, str]:
        manager = self.client.storage_manager
        storage_client = getattr(manager, "storage_client", None)
        return type(manager).__name__, getattr(storage_client, "protocol", "")

    def counters(self) -> CounterMap:
        return read_counters(self.tcp_device, self.rdma_device, self.rdma_port)

    def sample_idle_counters(self, seconds: float) -> dict[str, Any]:
        if seconds <= 0:
            raise ValueError("idle counter sample duration must be positive")
        before = self.counters()
        started = time.perf_counter()
        time.sleep(seconds)
        after = self.counters()
        elapsed = time.perf_counter() - started
        return {
            "seconds": elapsed,
            "ib_bytes": _counter_delta(before, after, "ib:"),
            "tcp_bytes": _counter_delta(before, after, "tcp:"),
        }

    def begin_round(self) -> tuple[CounterMap, float]:
        return self.counters(), time.perf_counter()

    def fetch(
        self,
        fields: list[str],
        batch_size: int,
        partition: str,
        expected: DigestMap,
        before: CounterMap,
        counter_started: float,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        meta = self.client.get_meta(
            data_fields=fields,
            batch_size=batch_size,
            partition_id=partition,
            mode="fetch",
            task_name="tq-benchmark",
        )
        received = self.client.get_data(meta)
        get_ms = (time.perf_counter() - started) * 1000
        after = self.counters()
        round_seconds = time.perf_counter() - counter_started
        actual = field_byte_digests(received, fields)
        mismatches = [field for field in fields if actual.get(field) != expected.get(field)]
        return {
            "get_ms": get_ms,
            "ib_bytes": _counter_delta(before, after, "ib:"),
            "tcp_bytes": _counter_delta(before, after, "tcp:"),
            "round_seconds": round_seconds,
            "byte_exact": not mismatches,
            "mismatch_fields": mismatches,
        }

    def shutdown(self) -> None:
        from relax.utils.tq.lifecycle import detach_tq_client

        detach_tq_client()


def _validate_attached_backend(protocol: str, manager: str, attached_protocol: str) -> None:
    if protocol == "simple" and manager != "AsyncSimpleStorageManager":
        raise RuntimeError(f"C0 attached unexpected storage manager {manager}")
    if protocol != "simple" and (manager != "MooncakeStorageManager" or attached_protocol != protocol):
        raise RuntimeError(
            f"{PROTOCOL_LABELS[protocol]} attached manager={manager} protocol={attached_protocol or '-'}"
        )


def _throughput_gbs(nbytes: int, milliseconds: float, label: str) -> float:
    if milliseconds <= 0:
        raise RuntimeError(f"{label} duration must be positive")
    return nbytes / milliseconds / 1e6


def _write_csv_record(writer: csv.DictWriter, csv_handle: TextIO, record: dict[str, Any]) -> None:
    writer.writerow(record)
    csv_handle.flush()


def _run_round(
    *,
    producer: Any,
    consumer: Any,
    payload: Any,
    fields: list[str],
    expected: DigestMap,
    nbytes: int,
    protocol: str,
    profile: str,
    requested_mib: int,
    run: int,
    writer: csv.DictWriter,
    csv_handle: TextIO,
    provenance: dict[str, str],
) -> dict[str, float]:
    """Measure, persist and clean one warmup or acceptance round."""
    partition = f"bench-{profile}-{requested_mib}-{run}"
    record: dict[str, Any] = {column: "" for column in CSV_COLUMNS}
    record.update(
        {
            "protocol": protocol,
            "profile": profile,
            "payload_mib": requested_mib,
            "actual_mib": round(nbytes / 1024**2, 2),
            "run": run,
            "measured": run > 0,
            **provenance,
        }
    )
    try:
        idle = ray.get(consumer.sample_idle_counters.remote(_IDLE_COUNTER_SAMPLE_SECONDS))
        before, counter_started = ray.get(consumer.begin_round.remote())
        started = time.perf_counter()
        producer.put(payload, partition_id=partition)
        put_ms = (time.perf_counter() - started) * 1000
        fetched = ray.get(
            consumer.fetch.remote(
                fields,
                payload.batch_size[0],
                partition,
                expected,
                before,
                counter_started,
            )
        )
        round_seconds = fetched["round_seconds"]
        proof_ib_bytes = _subtract_idle_noise(fetched["ib_bytes"], idle["ib_bytes"], idle["seconds"], round_seconds)
        proof_tcp_bytes = _subtract_idle_noise(fetched["tcp_bytes"], idle["tcp_bytes"], idle["seconds"], round_seconds)
        proven = wire_is_proven(protocol, proof_ib_bytes, proof_tcp_bytes, nbytes)
        put_gbs = _throughput_gbs(nbytes, put_ms, "put")
        get_gbs = _throughput_gbs(nbytes, fetched["get_ms"], "get")

        if not fetched["byte_exact"]:
            status, error_kind = "fail", "ByteExactMismatch"
            gate_error: BaseException | None = AssertionError("byte-exact mismatch after TQ get")
        elif not proven:
            status, error_kind = "fail", "WireProofFailed"
            gate_error = RuntimeError(
                f"wire proof failed for protocol={protocol} profile={profile} payload={requested_mib}MiB run={run}"
            )
        else:
            status, error_kind = "pass", ""
            gate_error = None
        record.update(
            {
                "status": status,
                "error_kind": error_kind,
                "byte_exact": fetched["byte_exact"],
                "mismatch_fields": ";".join(fetched["mismatch_fields"]),
                "wire_proven": proven,
                "put_ms": round(put_ms, 2),
                "get_ms": round(fetched["get_ms"], 2),
                "round_ms": round(round_seconds * 1000, 2),
                "put_gbs": round(put_gbs, 3),
                "get_gbs": round(get_gbs, 3),
                "ib_mb": round(fetched["ib_bytes"] / 1e6, 1),
                "tcp_mb": round(fetched["tcp_bytes"] / 1e6, 1),
                "proof_ib_mb": round(proof_ib_bytes / 1e6, 1),
                "proof_tcp_mb": round(proof_tcp_bytes / 1e6, 1),
                "idle_ib_mbps": round(idle["ib_bytes"] / idle["seconds"] / 1e6, 3),
                "idle_tcp_mbps": round(idle["tcp_bytes"] / idle["seconds"] / 1e6, 3),
            }
        )
    except BaseException as error:
        record.update({"status": "error", "error_kind": type(error).__name__})
        gate_error = error

    final_status = record["status"]
    record.update({"phase": "measurement", "cleanup_status": "pending"})
    if final_status == "pass":
        record["status"] = "pending"
    try:
        _write_csv_record(writer, csv_handle, record)
    except BaseException as write_error:
        if gate_error is None:
            gate_error = write_error
            final_status = "error"
            record["error_kind"] = type(write_error).__name__

    cleanup_failure: BaseException | None = None
    try:
        producer.clear_partition(partition)
        record["cleanup_status"] = "pass"
    except BaseException as error:
        cleanup_failure = error
        record.update({"cleanup_status": "error", "cleanup_error_kind": type(error).__name__})
        if gate_error is None:
            final_status = "error"
            record["error_kind"] = type(error).__name__
    record.update({"phase": "final", "status": final_status})
    try:
        _write_csv_record(writer, csv_handle, record)
    except BaseException as write_error:
        if gate_error is not None:
            if cleanup_failure is not None:
                cleanup_failure.__cause__ = write_error
                raise gate_error from cleanup_failure
            raise gate_error from write_error
        if cleanup_failure is not None:
            raise cleanup_failure from write_error
        raise
    if gate_error is not None:
        if cleanup_failure is not None:
            raise gate_error from cleanup_failure
        raise gate_error
    if cleanup_failure is not None:
        raise cleanup_failure
    return {"put_gbs": put_gbs, "get_gbs": get_gbs}


def _teardown_benchmark(consumer: Any, producer_attached: bool, owner: Any) -> None:
    """Attempt every cleanup step and propagate the first teardown error."""
    from relax.utils.tq.lifecycle import close_tq_owner, detach_tq_client

    first_error: BaseException | None = None

    def remember(error: BaseException) -> None:
        nonlocal first_error
        if first_error is None:
            first_error = error

    if consumer is not None:
        try:
            ray.get(consumer.shutdown.remote(), timeout=10)
        except BaseException as error:
            remember(error)
        try:
            ray.kill(consumer, no_restart=True)
        except BaseException as error:
            remember(error)
    if producer_attached:
        try:
            detach_tq_client()
        except BaseException as error:
            remember(error)
    if owner is not None:
        try:
            close_tq_owner(owner)
        except BaseException as error:
            remember(error)
    try:
        ray.shutdown()
    except BaseException as error:
        remember(error)
    if first_error is not None:
        raise first_error


def main() -> None:
    args = parse_args()
    provenance = collect_provenance()
    runtime_env: dict[str, Any] = {}
    if args.protocol == "tcp":
        os.environ["MC_TCP_ENABLE_CONNECTION_POOL"] = "1"
        runtime_env = {"env_vars": {"MC_TCP_ENABLE_CONNECTION_POOL": "1"}}
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from relax.utils.tq.lifecycle import attach_tq_client, initialize_tq_with_fallback

    with args.csv.open("x", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        csv_handle.flush()
        ray.init(address="auto", logging_level="ERROR", runtime_env=runtime_env)
        consumer = None
        owner = None
        producer_attached = False
        measured_runs = 0
        try:
            driver_node_id = ray.get_runtime_context().get_node_id()
            alive_ids = {node["NodeID"] for node in ray.nodes() if node.get("Alive")}
            if args.consumer_node_id not in alive_ids:
                raise RuntimeError("--consumer-node-id is not an alive Ray node")
            if args.consumer_node_id == driver_node_id:
                raise RuntimeError("producer and consumer must run on different Ray nodes")

            conf = build_conf(args.protocol, args.master, args.device, args.segment_gib)
            init_result = initialize_tq_with_fallback(conf, mode="required")
            owner = init_result.owner
            producer = attach_tq_client(init_result.config, role="benchmark-producer")
            producer_attached = True
            strategy = NodeAffinitySchedulingStrategy(node_id=args.consumer_node_id, soft=False)
            consumer = TQConsumer.options(scheduling_strategy=strategy).remote(
                init_result.config,
                args.tcp_device,
                args.device,
                args.rdma_port,
                args.protocol,
            )
            manager, attached_protocol = ray.get(consumer.describe.remote())
            _validate_attached_backend(args.protocol, manager, attached_protocol)
            print(f"[setup] {PROTOCOL_LABELS[args.protocol]} consumer=remote-node", flush=True)

            for profile in args.payload_profiles:
                for requested_mib in args.payload_mib:
                    payload = (
                        make_multimodal_payload(args.num_samples, requested_mib)
                        if profile == "multimodal"
                        else make_synthetic_payload(args.num_samples, requested_mib)
                    )
                    fields = sorted(payload.keys())
                    expected = field_byte_digests(payload, fields)
                    nbytes = payload_bytes(payload)
                    put_rates: list[float] = []
                    get_rates: list[float] = []
                    for run in range(args.repeats + 1):
                        rates = _run_round(
                            producer=producer,
                            consumer=consumer,
                            payload=payload,
                            fields=fields,
                            expected=expected,
                            nbytes=nbytes,
                            protocol=args.protocol,
                            profile=profile,
                            requested_mib=requested_mib,
                            run=run,
                            writer=writer,
                            csv_handle=csv_handle,
                            provenance=provenance,
                        )
                        if run == 0:
                            continue
                        put_rates.append(rates["put_gbs"])
                        get_rates.append(rates["get_gbs"])
                        measured_runs += 1
                    print(
                        f"[{profile} {requested_mib}MiB] byte_exact=PASS wire=PASS "
                        f"put={statistics.mean(put_rates):.2f}GB/s "
                        f"get_mean={statistics.mean(get_rates):.2f}GB/s "
                        f"get_median={statistics.median(get_rates):.2f}GB/s "
                        f"get_std={statistics.pstdev(get_rates):.2f}GB/s",
                        flush=True,
                    )
            print(f"[csv] wrote {measured_runs} measured rounds", flush=True)
        except BaseException as primary_error:
            try:
                _teardown_benchmark(consumer, producer_attached, owner)
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
            raise
        else:
            _teardown_benchmark(consumer, producer_attached, owner)


if __name__ == "__main__":
    main()
