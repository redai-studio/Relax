# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Receiver-side disk preparation and checked loading for the synchronous
MVP."""

import fcntl
import json
import shutil
import uuid
from pathlib import Path
from time import monotonic

from relax.distributed.weight_sync.config import WeightSyncConfig
from relax.distributed.weight_sync.consumer import Consumer
from relax.distributed.weight_sync.store import SharedFSStore, atomic_write
from relax.distributed.weight_sync.tensor_diff import compare


def prepare(engine, config_path: str, published: dict, reference: str) -> dict:
    import ray

    started = monotonic()
    config = WeightSyncConfig.load(config_path)
    node = ray.util.get_node_ip_address()
    root = Path(config_path).resolve().parent / "receivers" / node
    root.mkdir(parents=True, exist_ok=True)
    consumer = Consumer(
        SharedFSStore(config.store_root, config.disk_budget_bytes, canonical_root=config.canonical_root),
        SharedFSStore(root / "cache", config.disk_budget_bytes),
        root / "checkpoints",
        disk_budget_bytes=config.disk_budget_bytes,
        timeout_s=config.sync_timeout_s,
    )
    prepared = consumer.prepare(published["stream"], published["version"], full=config.full_only)
    if prepared.root != published["root"]:
        raise RuntimeError("Prepared checkpoint has the wrong target root")
    # Independent reference: copy the ordinary Bridge export, rather than
    # deriving the correctness oracle through this package's encoder/decoder.
    full = root / "full" / published["root"]
    with (root / ".full-reference.lock").open("ab") as lock:
        deadline = monotonic() + config.sync_timeout_s
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if monotonic() >= deadline:
                    raise TimeoutError("FULL reference copy lock deadline exceeded") from None
                from time import sleep

                sleep(0.05)
        if not full.exists():
            full.parent.mkdir(parents=True, exist_ok=True)
            staging = full.with_name(f".copy-{uuid.uuid4().hex}")
            shutil.copytree(reference, staging)
            staging.rename(full)
    hf_diff = compare(full, Path(prepared.path))
    diff_path = root / "diffs" / f"{published['version']:06d}-hf.json"
    atomic_write(diff_path, json.dumps(hf_diff, indent=2).encode())
    if not hf_diff["equal"]:
        raise RuntimeError(f"Independent FULL versus disk checkpoint mismatch: {diff_path}")
    # All six short-run versions are retained for replay/acceptance. Releasing
    # the transient pin does not GC artifacts or the prepared directory.
    consumer.release(prepared)
    return {
        "path": prepared.path,
        "full_reference": str(full),
        "root": prepared.root,
        "node": node,
        "prepare_s": monotonic() - started,
        "transport": dict(consumer.last_metrics),
        "hf_diff": str(diff_path),
    }


def install(engine, target: dict, published: dict, timeout: float) -> dict:
    import torch
    from sglang.srt.utils import MultiprocessingSerializer

    from .disk_weight_dump import DUMP_LOADER

    started = monotonic()
    deadline = started + timeout
    dump_root = Path(target["path"]).parents[1] / "gpu" / f"engine-{engine.rank}" / uuid.uuid4().hex
    dump_root.mkdir(parents=True)

    def request(endpoint: str, payload: dict) -> dict:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("Disk loader/readback deadline exceeded")
        result = engine._make_request(endpoint, payload, timeout=remaining)
        if not isinstance(result, dict) or result.get("success") is not True:
            raise RuntimeError(f"SGLang {endpoint} failed: {result}")
        return result

    def load(path: str) -> float:
        load_started = monotonic()
        request(
            "update_weights_from_disk",
            {
                "model_path": path,
                "load_format": "auto",
                "weight_version": str(published["version"]),
                "keep_pause": True,
                "flush_cache": True,
                "recapture_cuda_graph": False,
            },
        )
        return monotonic() - load_started

    def dump(label: str) -> Path:
        path = dump_root / label
        # A zero-length control tensor carries no weight values. The registered
        # hook only reads persistent GPU state; it never applies tensor updates.
        serialized = MultiprocessingSerializer.serialize([(str(path), torch.empty(0))], output_str=True)
        tp = engine.num_gpus_per_engine or engine.args.rollout_num_gpus_per_engine
        request(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": [serialized] * tp,
                "load_format": DUMP_LOADER,
                "flush_cache": False,
                "selector": "target",
            },
        )
        return path

    delta_load_s = load(target["path"])
    delta_dump = dump("delivered")  # Must survive before reference FULL reload.
    reference_started = monotonic()
    reference_load_s = load(target["full_reference"])
    full_dump = dump("reference")
    delta_ranks = {path.name for path in delta_dump.glob("rank-*")}
    full_ranks = {path.name for path in full_dump.glob("rank-*")}
    expected = engine.num_gpus_per_engine or engine.args.rollout_num_gpus_per_engine
    if delta_ranks != full_ranks or len(delta_ranks) != expected:
        raise RuntimeError("Missing GPU rank readback")
    checks = []
    for rank in sorted(delta_ranks):
        check = compare(full_dump / rank, delta_dump / rank)
        left = json.loads((full_dump / rank / "coverage.json").read_text())
        right = json.loads((delta_dump / rank / "coverage.json").read_text())
        if left != right:
            raise RuntimeError(f"GPU persistent coverage or aliases differ: {rank}")
        check_path = dump_root / f"{rank}-diff.json"
        atomic_write(check_path, json.dumps(check, indent=2).encode())
        if not check["equal"]:
            raise RuntimeError(f"Actual GPU parameters differ: {check_path}")
        checks.append({"rank": rank, "diff": str(check_path), "bytes": check["compared_bytes"]})
    result = {
        "status": "VERIFIED",
        "root": published["root"],
        "version": published["version"],
        "engine": engine.rank,
        "node": target["node"],
        "delta_load_s": delta_load_s,
        "reference_load_s": reference_load_s,
        "reference_validation_s": monotonic() - reference_started,
        "install_and_check_s": monotonic() - started,
        "readback": checks,
        "prepared": target,
    }
    return result
