# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from relax.distributed.weight_sync.__main__ import main
from relax.distributed.weight_sync.config import WeightSyncConfig
from relax.distributed.weight_sync.format import WeightSyncError


def test_cli_publishes_seals_and_consumes_from_new_process(tmp_path):
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"qwen3"}')
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "mode": "delta",
                "stream": "cli",
                "store_root": str(tmp_path / "store"),
                "canonical_root": str(tmp_path / "canonical"),
                "chunk_bytes": 256,
                "disk_budget_bytes": 1000000,
                "host_budget_bytes": 1000000,
                "anchor_interval": 2,
            }
        )
    )
    weights = {"x": torch.arange(512, dtype=torch.bfloat16)}
    for version in range(2):
        weights["x"][version] = -version
        save_file(weights, checkpoint / "model.safetensors")
        main(
            [
                "--store",
                str(tmp_path / "store"),
                "publish",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--update-id",
                str(version),
                "--result",
                str(tmp_path / "published.json"),
            ]
        )
    main(
        [
            "--store",
            str(tmp_path / "store"),
            "--canonical-root",
            str(tmp_path / "canonical"),
            "seal",
            "--stream",
            "cli",
            "--versions",
            "1",
            "--seal-id",
            "offline",
        ]
    )
    # This reader cannot access the owner's canonical root. Its sealed FULL is
    # sufficient after producer exit, with no live source service or Ray cluster.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "relax.distributed.weight_sync",
            "--store",
            str(tmp_path / "store"),
            "prepare",
            "--stream",
            "cli",
            "--version",
            "1",
            "--full",
            "--cache",
            str(tmp_path / "cache"),
            "--checkpoints",
            str(tmp_path / "prepared"),
            "--disk-budget-bytes",
            "1000000",
            "--result",
            str(tmp_path / "prepared.json"),
        ],
        check=True,
        timeout=20,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    prepared = json.loads((tmp_path / "prepared.json").read_text())
    actual = load_file(next(Path(prepared["path"]).glob("*.safetensors")))
    assert torch.equal(actual["x"], weights["x"])
    main(["--store", str(tmp_path / "store"), "query", "--result", str(tmp_path / "inventory.json")])
    assert "offline" in json.loads((tmp_path / "inventory.json").read_text())["seals"]


@pytest.mark.parametrize(
    "data",
    [
        {"unexpected": True},
        {"anchor_interval": True},
        {"compression": "true"},
        {"chunk_bytes": 2**30},
        {"mode": "delta"},
        {"mode": "delta", "stream": "s", "store_root": "/tmp/store", "canonical_root": "/tmp/store/canonical"},
    ],
)
def test_config_rejects_ambiguous_or_unbounded_settings(data):
    with pytest.raises(WeightSyncError):
        WeightSyncConfig.from_dict(data)


def test_incomplete_hf_checkpoint_is_rejected_before_publication(tmp_path):
    from relax.distributed.weight_sync.__main__ import checkpoint_tensors
    from relax.distributed.weight_sync.publisher import Publisher
    from relax.distributed.weight_sync.store import SharedFSStore

    save_file({"present": torch.ones(3)}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"present": "model.safetensors", "missing": "model.safetensors"}})
    )
    store = SharedFSStore(tmp_path / "store")
    pub = Publisher(store, "incomplete", {})
    with pytest.raises(WeightSyncError, match="missing"):
        pub.submit("0", checkpoint_tensors(tmp_path))
    assert store.head("incomplete") is None
    pub.close()
