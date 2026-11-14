# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import hashlib
import json
import runpy
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from relax.distributed.weight_sync.config import WeightSyncConfig
from relax.distributed.weight_sync.consumer import Consumer
from relax.distributed.weight_sync.publisher import Publisher
from relax.distributed.weight_sync.store import SharedFSStore
from relax.distributed.weight_sync.tensor_diff import compare


@pytest.mark.parametrize("fault", ["missing-base", "corrupt-object"])
def test_fault_copy_recovers_without_mutating_original_material(tmp_path, monkeypatch, fault):
    source = SharedFSStore(tmp_path / "source", canonical_root=tmp_path / "owner")
    publisher = Publisher(source, "frozen", {}, chunk_bytes=256)
    weights = torch.arange(128, dtype=torch.float32)
    publisher.finalize_pending(publisher.submit("initial", [("x", weights)]))
    weights[3] = -7
    target = publisher.finalize_pending(publisher.submit("update", [("x", weights)]))
    publisher.close()
    source.seal("offline", "frozen", [1])
    config = tmp_path / "fault.json"
    config.write_text(
        json.dumps({"mode": "delta", "stream": "frozen", "store_root": "source", "canonical_root": "owner"})
    )
    config_bytes = config.read_bytes()

    def source_hashes():
        return {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for root in [source.root, source.canonical_root]
            for path in root.rglob("*")
            if path.is_file()
        }

    original_hashes = source_hashes()
    copied = tmp_path / "fault-delivery"
    fault_config = tmp_path / "recovered-config.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fault_store",
            "--config",
            str(config),
            "--output-config",
            str(fault_config),
            "--delivery-root",
            str(copied),
            "--version",
            "1",
            "--fault",
            fault,
        ],
    )
    with pytest.raises(SystemExit) as completed:
        runpy.run_path(
            str(Path(__file__).resolve().parents[4] / "task_delta_weight_sync/fault_store.py"), run_name="__main__"
        )
    assert completed.value.code == 0
    assert config.read_bytes() == config_bytes
    actual = WeightSyncConfig.load(fault_config)
    consumer = Consumer(
        SharedFSStore(actual.store_root, canonical_root=actual.canonical_root),
        SharedFSStore(tmp_path / "cache"),
        tmp_path / "checkpoints",
    )
    prepared = consumer.prepare(actual.stream, 1)
    assert prepared.root == target.root
    assert prepared.artifact_id == source.record("frozen", 1)["artifacts"]["FULL"]
    assert consumer.last_metrics["fallback_reason"]
    reference = tmp_path / "ordinary-reference.safetensors"
    save_file({"x": weights}, reference)
    assert compare(reference, Path(prepared.path))["equal"]
    consumer.release(prepared)
    assert source_hashes() == original_hashes
