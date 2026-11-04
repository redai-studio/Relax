# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise CPU staging/group gates; these tests do not simulate GPU audit."""

import importlib.util
import json
import threading
from argparse import Namespace
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest
import ray
import torch
from safetensors.torch import save_file

from relax.backends.sglang.disk_weight_sync import prepare
from relax.distributed.weight_sync.config import WeightSyncConfig
from relax.distributed.weight_sync.publisher import Publisher
from relax.distributed.weight_sync.store import SharedFSStore
from relax.distributed.weight_sync.tensor_diff import compare


class DirectRPC:
    def __init__(self, function):
        self.remote = function


@pytest.mark.parametrize("empty_node", [False, True])
def test_disk_install_prepares_per_node_and_preserves_each_member_gate(tmp_path, monkeypatch, empty_node):
    # Load this CPU-safe module without the unrelated optional Megatron package
    # initializer. Only Ray dispatch is adapted; staging and file checks are real.
    path = Path(__file__).resolve().parents[4] / "relax/backends/megatron/weight_update/update_weight_from_disk.py"
    spec = importlib.util.spec_from_file_location("disk_updater_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.ray, "get", lambda value, timeout: value)
    config_path = tmp_path / "sync.json"
    config_path.write_text(
        json.dumps({"mode": "delta", "stream": "s", "store_root": "store", "canonical_root": "canonical"})
    )
    config = WeightSyncConfig.load(config_path)
    store = SharedFSStore(config.store_root, canonical_root=config.canonical_root)
    pub = Publisher(store, config.stream, {"model_config": {"model_type": "qwen3"}})
    weights = torch.arange(32, dtype=torch.float32)
    published = asdict(pub.finalize_pending(pub.submit("0", [("x", weights)])))
    pub.close()
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "config.json").write_text('{"model_type":"qwen3"}')
    save_file({"x": weights}, reference / "model.safetensors")
    preparations = Counter()
    engines = []

    class Engine:
        def __init__(self, index):
            self.index, self.node = index, f"node-{index // 2}"
            self.paused, self.loaded = False, None
            self.get_pid_and_node_id = DirectRPC(lambda: {"node_id": "" if empty_node and index == 1 else self.node})
            for name in (
                "prepare_disk_checkpoint",
                "pause_generation",
                "install_disk_checkpoint",
                "continue_generation",
                "disk_weight_generation_probe",
            ):
                setattr(self, name, DirectRPC(getattr(self, name)))

        def prepare_disk_checkpoint(self, config_path, published, reference):
            preparations[self.node] += 1
            with monkeypatch.context() as patch:
                patch.setattr(ray.util, "get_node_ip_address", lambda: self.node)
                return prepare(config_path, published, reference)

        def pause_generation(self, timeout):
            self.paused = True

        def install_disk_checkpoint(self, target, published, timeout):
            assert all(engine.paused for engine in engines)
            assert target["node"] == self.node
            assert compare(reference, Path(target["path"]))["equal"]
            self.loaded = published["root"]
            # CPU staging acknowledgement only; no GPU correctness claim.
            return {"status": "VERIFIED", "root": self.loaded, "engine": self.index}

        def continue_generation(self, timeout):
            assert all(engine.loaded == published["root"] for engine in engines)
            self.paused = False

        def disk_weight_generation_probe(self, timeout):
            assert not self.paused and self.loaded == published["root"]

    engines.extend(Engine(index) for index in range(4))
    lock = threading.Lock()
    remote_lock = Namespace(acquire=DirectRPC(lambda: lock.acquire(False)), release=DirectRPC(lock.release))
    updater = object.__new__(module.UpdateWeightFromDisk)
    updater.config, updater.config_path = config, config_path
    if empty_node:
        with pytest.raises(RuntimeError, match="node identity"):
            updater._install(tuple(engines), remote_lock, published, str(reference))
        assert not preparations and all(engine.loaded is None and not engine.paused for engine in engines)
    else:
        result = updater._install(tuple(engines), remote_lock, published, str(reference))
        assert preparations == {"node-0": 1, "node-1": 1}
        assert len(result["members"]) == 4
        assert all(engine.loaded == published["root"] and not engine.paused for engine in engines)
    assert not lock.locked()
