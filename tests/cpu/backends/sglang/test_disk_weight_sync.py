# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from dataclasses import asdict
from pathlib import Path

import ray
import torch
from safetensors.torch import save_file

from relax.backends.sglang.disk_weight_sync import prepare
from relax.distributed.weight_sync.config import WeightSyncConfig
from relax.distributed.weight_sync.publisher import Publisher
from relax.distributed.weight_sync.store import SharedFSStore
from relax.distributed.weight_sync.tensor_diff import compare


def test_receiver_repairs_corrupt_delta_using_external_canonical_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: "receiver-test")
    config_path = tmp_path / "sync.json"
    config_path.write_text(
        json.dumps(
            {
                "mode": "delta",
                "stream": "trained",
                "store_root": str(tmp_path / "delivery"),
                "canonical_root": str(tmp_path / "source-only"),
                "chunk_bytes": 256,
                "host_budget_bytes": 1000000,
                "disk_budget_bytes": 1000000,
            }
        )
    )
    config = WeightSyncConfig.load(config_path)
    store = SharedFSStore(config.store_root, canonical_root=config.canonical_root)
    pub = Publisher(store, config.stream, {"model_config": {"model_type": "qwen3"}}, **config.publisher_kwargs())
    x = torch.arange(512, dtype=torch.float32)
    pub.finalize_pending(pub.submit("0", [("x", x)]))
    x[0] = -1
    published = pub.finalize_pending(pub.submit("1", [("x", x)]))
    pub.close()
    manifest = store.resolve(config.stream, 1)
    changed = next(block for block in manifest["blocks"] if block["object"] is not None)
    (store.root / "objects" / changed["object"]).write_bytes(b"truncated")
    reference = tmp_path / "hf-reference"
    reference.mkdir()
    (reference / "config.json").write_text('{"model_type":"qwen3"}')
    save_file({"x": x}, reference / "model.safetensors")
    target = prepare(None, str(config_path), asdict(published), str(reference))
    assert target["root"] == published.root
    assert compare(reference, Path(target["path"]))["equal"]
    assert "FULL" in store.record(config.stream, 1)["artifacts"]
