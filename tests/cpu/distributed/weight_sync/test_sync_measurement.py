# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real Gloo gates around an opt-in benchmark's identity and result I/O."""

from datetime import timedelta
from pathlib import Path
from time import monotonic
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class FakeEngine:
    def __init__(self):
        self.probes = 0
        self.disk_weight_generation_probe = SimpleNamespace(remote=lambda timeout: self.probe)

    def probe(self):
        self.probes += 1
        return {"success": True}


class FakeUpdater:
    def __init__(self):
        self.weight = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
        self.named = [("weight", self.weight)]
        self.rollout_engines = [FakeEngine() for _ in range(8)]
        self.updated = False

    def _update_bucket_weights_from_distributed(self, named):
        named.clear()

    def update_weights(self):
        self.updated = True
        self._update_bucket_weights_from_distributed(self.named)
        return "ordinary-result"


def _measurement_rank(rank, rendezvous, directory):
    import json
    import os

    import ray

    from relax.distributed.weight_sync import provenance, store
    from task_delta_weight_sync.measure_sync import install

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    module = ModuleType("relax.backends.megatron.weight_update.update_weight_from_distributed")
    module.UpdateWeightFromDistributed = FakeUpdater
    original_update = FakeUpdater.update_weights
    args = SimpleNamespace(memory_weight_sync_config=None, disk_weight_sync_config=None)
    os.environ["TENSORBOARD_DIR"] = str(Path(directory) / "logs/tensorboard")
    write = store.atomic_write
    fail_write = True

    def result_write(path, data):
        if rank == 0 and fail_write and path.name == "000000.json":
            raise OSError("injected metrics write failure")
        return write(path, data)

    identity = provenance.runtime_identity

    def failed_identity():
        if rank == 0:
            raise OSError("injected identity failure")
        return identity()

    try:
        with (
            patch.dict("sys.modules", {module.__name__: module}),
            patch("relax.utils.distributed_utils.get_gloo_group", return_value=dist.group.WORLD),
            patch.object(ray, "get", side_effect=lambda refs, **kwargs: [f() for f in refs]),
        ):
            with patch.object(provenance, "runtime_identity", side_effect=failed_identity):
                with pytest.raises(RuntimeError, match="Measurement initialization failed.*identity failure"):
                    install(args)
            assert FakeUpdater.update_weights is original_update
            with patch.object(store, "atomic_write", side_effect=result_write):
                install(args)
                # The closure keeps the instrument's writer, including the
                # injected root-only failure. All ranks must leave together.
                with pytest.raises(RuntimeError, match="Weight sync measurement failed.*metrics write failure"):
                    FakeUpdater().update_weights()
                assert not (Path(directory) / "sync-measurements/000000.json").exists()
                # Finish both failure assertions before root can publish the
                # same version filename in the following success scenario.
                dist.barrier(group=dist.group.WORLD)
                fail_write = False
                updater = FakeUpdater()
                before = updater.weight.view(torch.uint8).numpy().tobytes()
                assert updater.update_weights() == "ordinary-result"
                assert updater.updated and not updater.named
                assert updater.weight.view(torch.uint8).numpy().tobytes() == before
                if rank == 0:
                    assert all(engine.probes == 1 for engine in updater.rollout_engines)
                    metrics = json.loads((Path(directory) / "sync-measurements/000000.json").read_text())
                    assert metrics["payload_bytes"] == 4 and metrics["elements"] == 2
                    assert metrics["members"] == 8 and metrics["version"] == 0
    finally:
        dist.destroy_process_group()


def test_measurement_io_errors_reach_every_rank_and_success_preserves_weights(tmp_path):
    context = mp.spawn(
        _measurement_rank, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=2, join=False
    )
    deadline = monotonic() + 35
    try:
        while not context.join(timeout=1):
            if monotonic() > deadline:
                pytest.fail("Measurement I/O errors stranded a rank past the 35 second deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3)
