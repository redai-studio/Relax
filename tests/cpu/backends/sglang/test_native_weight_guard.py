# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import time
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.backends.sglang.weight_sync_guard import (
    _NATIVE_WEIGHT_OPERATIONS,
    _guard_native_weights,
    commit_native_weight_lease,
    install_native_weight_guard,
)


def _guarded_manager_type():
    class Manager:
        def __init__(self, group):
            self.tp_cpu_group, self.effects = group, []
            self.runners = []
            for _ in range(2):
                runner = SimpleNamespace(model=torch.nn.Linear(2, 2))
                runner.weight_updater = SimpleNamespace(get_model=lambda runner=runner: runner.model)
                self.runners.append(runner)
            self.tp_worker = SimpleNamespace()
            self.action = None

        def _select_runners(self, selector="all"):
            return list(zip(("target", "draft"), self.runners, strict=True))

    def operation(name):
        def body(self, recv_req):
            self.effects.append(name)
            if self.action is not None:
                result = self.action(self)
                if result is not None:
                    return result
            return name

        return body

    def finalize(self):
        self.effects.append("finalize_pending_bf16_weights")

    for name in _NATIVE_WEIGHT_OPERATIONS:
        setattr(Manager, name, finalize if name == "finalize_pending_bf16_weights" else operation(name))
    return Manager


@pytest.fixture
def native_guard(monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(dist, "all_gather_object", lambda output, value, group: output.__setitem__(0, value))
    manager_type = _guarded_manager_type()
    install_native_weight_guard(manager_type)
    manager = manager_type(object())
    guard = _guard_native_weights(manager, "initial")
    return manager, guard


def _native_guard_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        manager_type = _guarded_manager_type()
        install_native_weight_guard(manager_type)
        checked = []
        for name in _NATIVE_WEIGHT_OPERATIONS:
            manager = manager_type(dist.group.WORLD)
            guard = _guard_native_weights(manager, "initial")
            if rank == 0:
                guard.lease.begin("one remote reader", 2, "delivery")
            try:
                getattr(manager, name)(*(() if name == "finalize_pending_bf16_weights" else (None,)))
            except RuntimeError as error:
                assert "lifetime guard rejected" in str(error)
            else:
                raise AssertionError("A peer freed an allocation before remote completion")
            assert not manager.effects and guard.lease.failed
            checked.append(name)
        # Completing this rank's native registrations alone cannot unlock it.
        # A peer still waiting on DONE must keep the whole group quarantined.
        for complete in (False, True):
            manager = manager_type(dist.group.WORLD)
            guard = _guard_native_weights(manager, "initial")
            identity = guard.lease.begin("session", 2, f"delivery-{rank}")
            target = object()
            guard.lease.add(target, 2)
            if rank == 0 or complete:
                guard.lease.complete(target)
            if complete:
                commit_native_weight_lease(manager.runners[0].model, identity, dist.group.WORLD)
                assert manager.release_memory_occupation(None) == "release_memory_occupation"
                assert guard.lease.active is None and not guard.lease.failed
            else:
                with pytest.raises(RuntimeError, match="completion was not agreed"):
                    commit_native_weight_lease(manager.runners[0].model, identity, dist.group.WORLD)
                assert guard.lease.active == identity and guard.lease.failed and not manager.effects
        # A wrong caller group must still reject on the actual worker TP group.
        singleton = dist.new_group([0], backend="gloo", timeout=timedelta(seconds=10))
        manager = manager_type(dist.group.WORLD)
        guard = _guard_native_weights(manager, "initial")
        identity = guard.lease.begin("session", 2, f"delivery-{rank}")
        with pytest.raises(RuntimeError, match="completion was not agreed"):
            commit_native_weight_lease(
                manager.runners[0].model, identity, singleton if rank == 0 else dist.group.WORLD
            )
        assert guard.lease.active == identity and guard.lease.failed and not manager.effects
        from pathlib import Path

        Path(output, f"{rank}.json").write_text(json.dumps(checked))
    finally:
        dist.destroy_process_group()


def test_native_guard_one_busy_rank_rejects_all_tp_mutations_without_hanging(tmp_path):
    rendezvous = (tmp_path / "gloo-store").as_uri()
    context = mp.spawn(_native_guard_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            assert time.monotonic() < deadline, "Native lifetime guard failed to converge"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
    for rank in range(2):
        assert json.loads((tmp_path / f"{rank}.json").read_text()) == list(_NATIVE_WEIGHT_OPERATIONS)


def test_real_scheduler_manager_rejects_mutations_before_native_body(monkeypatch):
    updater = pytest.importorskip("sglang.srt.managers.scheduler_components.weight_updater")
    manager_type = updater.SchedulerWeightUpdaterManager
    operations = [name for name in _NATIVE_WEIGHT_OPERATIONS if callable(getattr(manager_type, name, None))]
    for name in operations:
        monkeypatch.setattr(manager_type, name, getattr(manager_type, name))
    install_native_weight_guard(manager_type)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(dist, "all_gather_object", lambda output, value, group: output.__setitem__(0, value))
    target, draft = (SimpleNamespace(model=torch.nn.Linear(2, 2)) for _ in range(2))
    if callable(getattr(manager_type, "_select_runners", None)):
        for runner in (target, draft):
            runner.weight_updater = SimpleNamespace(get_model=lambda runner=runner: runner.model)
        tp_worker = SimpleNamespace(weight_update_runners=lambda: [("target", target)])
        draft_worker = SimpleNamespace(weight_update_runners=lambda: [("draft", draft)])
    else:
        tp_worker = SimpleNamespace(model_runner=target)
        draft_worker = SimpleNamespace(draft_model_runner=draft)
    manager = manager_type(
        tp_worker=tp_worker,
        draft_worker=draft_worker,
        tp_cpu_group=object(),
        memory_saver_adapter=Mock(),
        flush_cache=Mock(),
        is_fully_idle=Mock(),
    )
    guard = _guard_native_weights(manager, "initial")
    assert draft.model._relax_native_weight_lease is target.model._relax_native_weight_lease
    guard.lease.begin("session", 2, "delivery")
    for name in operations:
        with pytest.raises(RuntimeError, match="lifetime guard rejected"):
            getattr(manager, name)(*(() if name == "finalize_pending_bf16_weights" else (SimpleNamespace(),)))
    assert not manager.offload_tags
    manager.memory_saver_adapter.pause.assert_not_called()
    manager.memory_saver_adapter.resume.assert_not_called()
    manager.flush_cache.assert_not_called()


@pytest.mark.parametrize("success", [True, False])
def test_prepare_reply_uses_real_sglang_response_and_never_replays(native_guard, success):
    from sglang.srt.managers.communicator import FanOutCommunicator
    from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqOutput

    manager, guard = native_guard
    model = guard.models[0]
    model._relax_weight_sync_reply = "stale"
    payload = json.dumps({"shard_p2p_receivers": [{"identity": {"nonce": "current"}}]})

    def prepare(manager):
        assert not hasattr(model, "_relax_weight_sync_reply")
        model._relax_weight_sync_reply = payload
        return UpdateWeightsFromTensorReqOutput(success=success, message="native result")

    manager.action = prepare
    result = manager.update_weights_from_tensor(None)
    assert FanOutCommunicator.merge_results([result]) == (success, payload if success else "native result")
    assert not hasattr(model, "_relax_weight_sync_reply")
    manager.action = None
    assert manager.update_weights_from_tensor(None) == "update_weights_from_tensor"

    def fail(manager):
        model._relax_weight_sync_reply = payload
        raise RuntimeError("native failure")

    manager.action = fail
    with pytest.raises(RuntimeError, match="native failure"):
        manager.update_weights_from_tensor(None)
    assert not hasattr(model, "_relax_weight_sync_reply") and guard.lease.failed
