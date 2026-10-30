# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import pytest
import torch


# backup() imports megatron's TP-attribute helpers lazily on every call.
pytest.importorskip("megatron.core")

from relax.utils.training.tensor_backper import TensorBackuper


@pytest.fixture(autouse=True)
def cpu_only_snapshot_copies(monkeypatch):
    from relax.utils.training import tensor_backper

    monkeypatch.setattr(tensor_backper, "_PIN_MEMORY", False)
    monkeypatch.setattr(tensor_backper.device_module, "synchronize", lambda: None)


def _mark_tp(tensor: torch.Tensor, *, parallel_mode="column", partition_dim=0, partition_stride=1):
    """Attach Megatron TP attributes the way Megatron marks a column-parallel
    param."""
    tensor.tensor_model_parallel = True
    tensor.partition_dim = partition_dim
    tensor.partition_stride = partition_stride
    tensor.parallel_mode = parallel_mode
    return tensor


def _source_of(params: dict[str, torch.Tensor]):
    # backup() iterates the source twice (allocate-missing, then copy), so the
    # getter must return a fresh iterable on every call.
    return lambda: params.items()


def test_backup_cpu_snapshot_attaches_tp_attrs():
    param = _mark_tp(torch.nn.Parameter(torch.randn(4, 3)))
    backuper = TensorBackuper.create(_source_of({"module.layer.weight": param}), single_tag=None)

    backuper.backup("actor")

    snapshot = backuper.get("actor")["module.layer.weight"]
    assert snapshot.device.type == "cpu"
    assert snapshot.tensor_model_parallel is True
    assert snapshot.partition_dim == 0
    assert snapshot.partition_stride == 1
    assert snapshot.parallel_mode == "column"
    assert torch.equal(snapshot, param.detach())


def test_backup_cpu_snapshot_backfills_defaults_for_unmarked_params():
    param = torch.nn.Parameter(torch.randn(2, 5))
    backuper = TensorBackuper.create(_source_of({"w": param}), single_tag=None)

    backuper.backup("actor")

    snapshot = backuper.get("actor")["w"]
    # Values are Megatron's set_defaults_if_not_set_tensor_model_parallel_attributes
    # defaults; parallel_mode has no Megatron default, so it is not asserted.
    assert snapshot.tensor_model_parallel is False
    assert snapshot.partition_dim == -1
    assert snapshot.partition_stride == 1
    assert torch.equal(snapshot, param.detach())


def test_actor_snapshot_survives_live_reference_switch_and_reuses_allocation():
    param = _mark_tp(torch.nn.Parameter(torch.tensor([1.0, 2.0])))
    backuper = TensorBackuper.create(_source_of({"weight": param}), single_tag=None)
    backuper.backup("actor")
    actor = backuper.get("actor")["weight"]
    with torch.no_grad():
        param.fill_(9.0)
    backuper.backup("ref")
    assert torch.equal(actor, torch.tensor([1.0, 2.0]))
    backuper.restore("actor")
    assert torch.equal(param, actor)
    with torch.no_grad():
        param.add_(1.0)
    backuper.backup("actor")
    assert backuper.get("actor")["weight"] is actor
    assert torch.equal(actor, torch.tensor([2.0, 3.0]))
    assert actor.tensor_model_parallel
