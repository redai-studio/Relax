# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two real Gloo ranks must converge after root-only audit failures."""

import json
import os
from datetime import timedelta
from pathlib import Path
from time import monotonic
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class Remote:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return lambda: self.function(*args, **kwargs)


class FakeUpdater:
    def __init__(self, args, model):
        self.args, self.model = args, model
        self.failed = False
        self.source = SimpleNamespace(close=lambda: None)
        self.members = {str(n) for n in range(8)}
        self.encoder = SimpleNamespace(version=0, schema="schema")
        self.config = SimpleNamespace(stream="audit-test")
        self._dws_details = dict(receivers=[dict(update="live-sha")] * 8)
        self.quarantined = False

    def update_weights(self, manager):
        if self.args.stage in ("update", "closure"):
            raise OSError("injected measurement failure")
        return "ordinary-result"

    def _close_generation(self, manager):
        self.quarantined = True
        if self.args.stage == "closure":
            raise OSError("injected closure failure")
        return []


def _audit_rank(rank, rendezvous, directory):
    import ray

    from relax.distributed.weight_sync import provenance, store
    from task_delta_weight_sync import measure_sync, memory_audit

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    model_module = ModuleType("relax.backends.megatron.model")
    updater_module = ModuleType("relax.backends.megatron.weight_update.update_weight_from_memory")
    updater_module.UpdateWeightFromMemory = FakeUpdater
    stage = ""
    events = []
    original_write = store.atomic_write

    def export(args, version, model, **kwargs):
        events.append("export")
        if stage == "export" and rank == 1:
            raise OSError("injected peer export failure")
        if rank == 0:
            path = Path(args.save_hf.format(rollout_id=version))
            path.mkdir(parents=True, exist_ok=False)
            from safetensors.torch import save_file

            save_file({"weight": model}, path / "model.safetensors")

    model_module.save_hf_model = export

    def identity():
        if stage == "identity" and rank == 1:
            raise OSError("injected peer identity failure")
        return dict(pid=os.getpid())

    def write(path, data):
        if stage == "receipt" and rank == 0 and path.parent.name == "receipts":
            raise OSError("injected receipt failure")
        return original_write(path, data)

    def resume(**kwargs):
        if stage == "resume":
            raise OSError("injected resume failure")

    def snapshot(function, destination, update):
        events.append("dump")
        if stage == "dump":
            raise OSError("injected dump failure")
        assert update == "live-sha"
        return dict(directory=destination, update=update)

    lock = SimpleNamespace(acquire=Remote(lambda: True), release=Remote(lambda: None))
    engines = [
        SimpleNamespace(
            _actor_id=SimpleNamespace(hex=lambda n=n: str(n)),
            pause_generation=Remote(lambda **kwargs: None),
            continue_generation=Remote(resume),
            __ray_call__=Remote(snapshot),
        )
        for n in range(8)
    ]
    manager = SimpleNamespace(get_rollout_engines_and_lock=Remote(lambda: (engines, lock)))

    def get(refs, **kwargs):
        return [ref() for ref in refs] if isinstance(refs, list) else refs()

    try:
        with (
            patch.dict("sys.modules", {model_module.__name__: model_module, updater_module.__name__: updater_module}),
            patch("relax.utils.distributed_utils.get_gloo_group", return_value=dist.group.WORLD),
            patch.object(measure_sync, "install"),
            patch.object(provenance, "runtime_identity", side_effect=identity),
            patch.object(store, "atomic_write", side_effect=write),
            patch.object(ray, "get", side_effect=get),
        ):
            memory_audit.install(SimpleNamespace(memory_weight_sync_config="enabled"))
            for stage in ("update", "closure", "identity", "dump", "export", "receipt", "resume", "success"):
                events.clear()
                case = Path(directory) / stage
                os.environ["TENSORBOARD_DIR"] = str(case / "logs/tensorboard")
                weight = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
                original = weight.view(torch.uint8).numpy().tobytes()
                updater = FakeUpdater(
                    SimpleNamespace(stage=stage, memory_weight_sync_config=str(case / "memory.json")), weight
                )
                if stage == "success":
                    assert updater.update_weights(manager) == "ordinary-result"
                    assert not updater.failed and not updater.quarantined
                    if rank == 0:
                        assert events == ["dump"] * 8 + ["export"]
                        receipt = json.loads((case / "gpu-audit/receipts/000000.json").read_text())
                        assert receipt["status"] == "CAPTURED_NOT_COMPARED"
                        assert receipt["update"] == "live-sha" and len(receipt["captured"]) == 8
                else:
                    with pytest.raises(RuntimeError, match="GPU memory audit") as error:
                        updater.update_weights(manager)
                    assert updater.failed
                    if stage == "closure":
                        assert "injected closure failure" in str(error.value)
                    if rank == 0:
                        assert updater.quarantined and updater.source is None
                    if stage in ("update", "closure", "identity", "dump"):
                        assert "export" not in events
                assert weight.view(torch.uint8).numpy().tobytes() == original
                dist.barrier(group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()


def test_audit_failures_converge_and_capture_precedes_independent_export(tmp_path):
    context = mp.spawn(_audit_rank, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=2, join=False)
    deadline = monotonic() + 35
    try:
        while not context.join(timeout=1):
            if monotonic() > deadline:
                pytest.fail("Audit failure stranded a rank past the 35 second deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3)
