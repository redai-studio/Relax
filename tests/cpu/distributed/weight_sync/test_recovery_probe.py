# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression for the SGLang failure response and Ray proxy serialization."""

import sys
from types import SimpleNamespace

import cloudpickle
import pytest

from task_delta_weight_sync.recovery_probe import (
    InjectedRecoveryFailure,
    _DeferredEngine,
    _DeferredManager,
    catch_injected_fault,
    worker_control,
)


def test_only_explicit_injected_failure_becomes_a_normal_tp_failure_response():
    def original(self, error=None):
        if error:
            raise error
        return True, "ordinary result"

    wrapped = catch_injected_fault(original)
    assert wrapped(None) == (True, "ordinary result")
    success, message = wrapped(None, InjectedRecoveryFailure("partial GPU write"))
    assert not success and "InjectedRecoveryFailure" in message
    with pytest.raises(ValueError, match="unrelated"):
        wrapped(None, ValueError("unrelated"))


def test_serialized_manager_tracks_actual_proxy_calls_without_resuming_real_engine(monkeypatch):
    def copied(value):
        return cloudpickle.loads(cloudpickle.dumps(value))

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(put=copied))
    actor = SimpleNamespace(_actor_id="identity", live_method="forwarded")
    original = _DeferredEngine(actor)
    manager = copied(_DeferredManager((original,), "lock"))
    engines, lock, _, counts, offsets = manager.get_rollout_engines_and_lock.remote()
    proxy = engines[0]
    assert proxy._actor_id == "identity" and proxy.live_method == "forwarded"
    assert lock == "lock" and counts == [2] * 4 and offsets == [0, 2, 4, 6]
    assert proxy.continue_generation.remote()["deferred"]
    assert proxy.disk_weight_generation_probe.remote()["deferred"]
    assert proxy.calls == ["continue", "probe"] and not original.calls


def test_successful_peer_records_current_install_nonce_without_swallowing_errors(tmp_path, monkeypatch):
    import json

    import torch

    from task_delta_weight_sync import recovery_probe

    class Updater:
        def update_weights_from_tensor(self):
            return True, "ordinary"

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.model_runner_components.weight_updater",
        SimpleNamespace(WeightUpdater=Updater),
    )
    monkeypatch.setattr(
        recovery_probe, "scheduler_state", lambda: dict(paused=True, fully_idle=True, running_requests=0)
    )
    model = torch.nn.Linear(2, 2)
    receiver = SimpleNamespace(poisoned=False, version=3, _apply_bucket=lambda: None)

    def apply(group, receive):
        receiver._apply_bucket()
        receiver.version = group["version"]
        return {"success": True}

    receiver.apply_group = apply
    model._relax_sharded_memory_receiver = receiver
    control = dict(command="arm_failure", member="engine1:tp:0", directory=str(tmp_path), inject=False)
    worker_control(model, control)
    assert receiver.apply_group(dict(group_update="fresh-nonce", version=4), None)["success"]
    receipt = json.loads((tmp_path / "failure-probe/engine1:tp:0-installed.json").read_text())
    assert receipt["group_update"] == "fresh-nonce" and receipt["applied_buckets"] == 1
    control["command"] = "disarm_failure"
    assert worker_control(model, control)["receiver_version"] == 4
    assert model(torch.ones(1, 2)).shape == (1, 2)
    assert Updater().update_weights_from_tensor() == (True, "ordinary")


def test_fault_is_not_armed_with_a_retained_overlap_batch(tmp_path, monkeypatch):
    import torch

    from task_delta_weight_sync import recovery_probe

    monkeypatch.setattr(
        recovery_probe, "scheduler_state", lambda: dict(paused=True, fully_idle=False, running_requests=1)
    )
    model = torch.nn.Linear(2, 2)
    receiver = SimpleNamespace(_apply_bucket=lambda: None)
    model._relax_sharded_memory_receiver = receiver
    original_bucket = receiver._apply_bucket
    control = dict(command="arm_failure", member="engine1:tp:0", directory=str(tmp_path), inject=False)
    with pytest.raises(RuntimeError, match="paused and drained"):
        worker_control(model, control)
    assert receiver._apply_bucket is original_bucket
    assert not hasattr(model, "_relax_recovery_probe")
    assert model(torch.ones(1, 2)).shape == (1, 2)
