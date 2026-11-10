# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Native wire operation accounting without allocating full model replicas."""

import json
import sys
import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from task_delta_weight_sync import prepared_nccl, recovery_probe
from task_delta_weight_sync.prepared_nccl import plan_capacity, wire_bucket
from task_delta_weight_sync.sharded_gpu_audit import AUDIT_LOADER


def test_prepared_plan_keeps_native_order_dtype_and_bytes_with_bounded_reuse():
    first = [
        ("a", torch.empty(3, dtype=torch.bfloat16)),
        ("b", torch.empty(3, dtype=torch.bfloat16)),
        ("c", torch.empty(2, dtype=torch.float32)),
        ("d", torch.empty(3, dtype=torch.bfloat16)),
    ]
    bucket = wire_bucket(first)
    assert bucket == [["bfloat16", 3, 2], ["float32", 2, 1], ["bfloat16", 3, 1]]
    capacities, totals = plan_capacity([bucket, bucket])
    assert capacities == {"bfloat16": 9, "float32": 2}
    assert totals == {"bfloat16": 36, "float32": 16}


@pytest.mark.parametrize(
    "plan", [[], [[["int8", 1, 1]]], [[["bfloat16", -1, 1]]], [[["bfloat16", 1, 0]]], [[["float32", 1 << 30, 1]]]]
)
def test_invalid_or_unbounded_transfer_plan_is_rejected_before_allocation(plan):
    with pytest.raises(ValueError):
        plan_capacity(plan)


def test_prepared_worker_supports_frozen_slots_updater_and_releases_owned_scratch(monkeypatch, tmp_path):
    @dataclass(frozen=True, slots=True, kw_only=True)
    class WeightUpdater:
        device: str
        _model_update_group: dict

        def update_weights_from_tensor(self, named_tensors, load_format=None):
            return True, "ordinary update"

    class Scratch(dict):
        pass

    scratch_refs = []

    def allocate(plan, device):
        assert device == "cpu"
        state = Scratch(plan=plan)
        scratch_refs.append(weakref.ref(state))
        return state

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.model_runner_components.weight_updater",
        SimpleNamespace(WeightUpdater=WeightUpdater),
    )
    monkeypatch.setattr(prepared_nccl, "prepare", allocate)
    monkeypatch.setattr(prepared_nccl, "replay", lambda state, group: {"bytes": 12, "plan": state["plan"]})
    monkeypatch.setattr(recovery_probe, "scheduler_state", lambda: {"paused": True})
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    original = WeightUpdater.update_weights_from_tensor
    owner = WeightUpdater(device="cpu", _model_update_group={"test": object()})
    other = WeightUpdater(device="cpu", _model_update_group={"test": object()})
    prepared_nccl.install_worker()

    def call(updater, command, **extra):
        control = dict(
            command=command,
            member="worker:tp:0",
            phase="transfer",
            update_id="test",
            paused=True,
            directory=str(tmp_path / "audit"),
            group_name="test",
            **extra,
        )
        return updater.update_weights_from_tensor([(json.dumps(control), torch.empty(0))], load_format=AUDIT_LOADER)

    assert owner.update_weights_from_tensor([], load_format="ordinary") == (True, "ordinary update")
    with pytest.raises(RuntimeError, match="not allocated"):
        call(owner, "run_transfer", round=0)
    assert call(owner, "prepare_transfer", plan=[[["bfloat16", 6, 1]]])[0]
    with pytest.raises(RuntimeError, match="already allocated"):
        call(owner, "prepare_transfer", plan=[])
    for command in ("run_transfer", "close_transfer"):
        with pytest.raises(RuntimeError, match="another updater"):
            call(other, command, round=0)
    assert call(owner, "run_transfer", round=0)[0]
    report = json.loads((tmp_path / "audit/worker:tp:0/prepared-0.json").read_text())
    assert report["bytes"] == 12
    assert scratch_refs[0]() is not None
    assert call(owner, "close_transfer")[0]
    assert scratch_refs[0]() is None
    assert WeightUpdater.update_weights_from_tensor is original
    assert not hasattr(WeightUpdater, "_relax_prepared_transport")
