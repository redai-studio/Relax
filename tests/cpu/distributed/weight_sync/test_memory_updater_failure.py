# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded two-rank regression for errors before converter collectives."""

from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.distributed.weight_sync.memory_delta import MemoryDeltaEncoder


def _updater_class():
    # The backend package initializer imports unrelated GPU extensions. Load
    # this exact file to exercise its CPU Gloo/error paths without those hooks.
    path = Path(__file__).resolve().parents[4] / "relax/backends/megatron/weight_update/update_weight_from_memory.py"
    spec = spec_from_file_location("memory_updater_failure_subject", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UpdateWeightFromMemory


def _getter_failure_rank(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    try:
        updater = object.__new__(_updater_class())
        updater.failed, updater.source = False, None
        updater.encoder = MemoryDeltaEncoder()
        updater._close_generation = lambda manager: []

        def getter():
            if rank == 1:
                raise ValueError("snapshot unavailable")
            return {}

        def converter(weights):
            raise AssertionError("Peers must reject the snapshot before converter collectives")

        updater.weights_getter = getter
        updater.iterator = SimpleNamespace(get_hf_weight_chunks=converter)
        with patch("relax.utils.distributed_utils.get_gloo_group", return_value=dist.group.WORLD):
            with pytest.raises(RuntimeError, match="snapshot unavailable"):
                updater.update_weights(None)
        assert updater.failed and updater.encoder.version == -1
    finally:
        dist.destroy_process_group()


def test_snapshot_getter_failure_reaches_every_rank_without_entering_converter(tmp_path):
    # A single-rank mock cannot expose divergent collective sequences. Two CPU
    # Gloo ranks suffice; impose an outer deadline as well as the PG timeout.
    context = mp.spawn(_getter_failure_rank, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=False)
    deadline = monotonic() + 35
    try:
        while not context.join(timeout=1):
            if monotonic() > deadline:
                pytest.fail("Snapshot error did not reach every rank within 35 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3)


def test_failed_kill_does_not_leave_other_inference_members_active(monkeypatch):
    import ray

    class Engine:
        def __init__(self, name, unreachable=False):
            self._actor_id = SimpleNamespace(hex=lambda: name)
            self.active, self.unreachable = True, unreachable
            self.pause_generation = SimpleNamespace(remote=lambda **kwargs: self.pause)

        def pause(self):
            if self.unreachable:
                raise OSError("pause unreachable")
            self.active = False

    missing, known, discovered = Engine("missing", True), Engine("known"), Engine("discovered")
    manager = SimpleNamespace(
        get_rollout_engines_and_lock=SimpleNamespace(remote=lambda: lambda: ([discovered], None))
    )
    monkeypatch.setattr(ray, "get", lambda result, **kwargs: result())

    def failed_kill(engine, **kwargs):
        raise OSError("kill unreachable")

    monkeypatch.setattr(ray, "kill", failed_kill)
    updater = object.__new__(_updater_class())
    updater.engines = (missing, known)
    errors = updater._close_generation(manager)
    assert not known.active and not discovered.active
    assert missing.active and len(errors) == 1 and "missing" in errors[0]
