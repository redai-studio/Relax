"""Unit tests for TensorBackuper snapshot support used by hybrid-DCS weight
sync.

Covers ``_TensorBackuperNormal`` (created via ``TensorBackuper.create(source_getter,
single_tag=None)``):

- Megatron TP attributes (``tensor_model_parallel`` / ``partition_dim`` /
  ``partition_stride`` / ``parallel_mode``) are attached to snapshots, with Megatron's
  own defaults backfilled for unmarked params, so downstream TP-gather consumers
  (e.g. DCS's ``all_gather_param``) can rely on the attributes always being present.
- ``backup(tag, on_device=True)`` keeps snapshots device-resident as views into one
  contiguous per-dtype buffer (avoiding per-parameter cudaMalloc fragmentation) and
  reuses that buffer across calls.
- The default ``backup(tag)`` path stays on (optionally pinned) host memory.
- ``_TensorBackuperNoop`` accepts the same ``backup(tag, on_device=...)`` signature.
"""

from __future__ import annotations

import pytest
import torch


# backup() imports megatron's TP-attribute helpers lazily on every call.
pytest.importorskip("megatron.core")

from relax.utils import device as device_utils
from relax.utils.training.tensor_backper import TensorBackuper, _TensorBackuperNoop


# Host snapshots allocate pinned memory and backup synchronizes the accelerator.
# Fake CUDA tensors also need a CUDA context for view operations in supported PyTorch builds.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="snapshot allocation/synchronization requires CUDA"
)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="on_device backup requires CUDA")
def test_backup_on_device_layout():
    p1 = _mark_tp(torch.nn.Parameter(torch.randn(4, 3, device="cuda")))
    p2 = torch.nn.Parameter(torch.randn(5, device="cuda"))
    p3 = torch.nn.Parameter(torch.randn(2, 2, device="cuda", dtype=torch.bfloat16))
    params = {"p1": p1, "p2": p2, "p3": p3}
    backuper = TensorBackuper.create(_source_of(params), single_tag=None)

    backuper.backup("actor", on_device=True)

    snapshots = backuper.get("actor")
    # (a) All snapshots stay device-resident.
    for snapshot in snapshots.values():
        assert snapshot.device.type == "cuda"
    # (b) Same-dtype snapshots are views into one contiguous flat buffer.
    assert snapshots["p1"].untyped_storage().data_ptr() == snapshots["p2"].untyped_storage().data_ptr()
    assert snapshots["p2"].data_ptr() == snapshots["p1"].data_ptr() + p1.numel() * p1.element_size()
    # A different dtype gets its own buffer.
    assert snapshots["p3"].untyped_storage().data_ptr() != snapshots["p1"].untyped_storage().data_ptr()
    # (c) Values were copied into the snapshots.
    for name, param in params.items():
        assert torch.equal(snapshots[name], param.detach())
    # (d) TP attributes: marked param keeps its values; unmarked params get backfilled defaults.
    assert snapshots["p1"].tensor_model_parallel is True
    assert snapshots["p1"].partition_dim == 0
    assert snapshots["p1"].partition_stride == 1
    assert snapshots["p1"].parallel_mode == "column"
    assert snapshots["p2"].tensor_model_parallel is False
    assert snapshots["p3"].partition_dim == -1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="on_device backup requires CUDA")
def test_backup_on_device_second_call_reuses_buffer():
    p1 = torch.nn.Parameter(torch.randn(4, device="cuda"))
    p2 = torch.nn.Parameter(torch.randn(6, device="cuda"))
    backuper = TensorBackuper.create(_source_of({"p1": p1, "p2": p2}), single_tag=None)

    backuper.backup("actor", on_device=True)
    first = dict(backuper.get("actor"))
    storage_ptrs = {name: snapshot.untyped_storage().data_ptr() for name, snapshot in first.items()}

    with torch.no_grad():
        p1.add_(1.0)
        p2.mul_(2.0)
    backuper.backup("actor", on_device=True)

    second = backuper.get("actor")
    for name in ("p1", "p2"):
        # No fresh allocation: the same snapshot tensor (and its storage) is reused.
        assert second[name] is first[name]
        assert second[name].untyped_storage().data_ptr() == storage_ptrs[name]
    # Values reflect the updated sources.
    assert torch.equal(second["p1"], p1.detach())
    assert torch.equal(second["p2"], p2.detach())


def test_backup_default_unchanged_cpu_pinned():
    param = torch.nn.Parameter(torch.randn(3, 4))
    backuper = TensorBackuper.create(_source_of({"w": param}), single_tag=None)

    backuper.backup("actor")

    snapshot = backuper.get("actor")["w"]
    assert snapshot.device.type == "cpu"
    assert snapshot.is_pinned() == device_utils.use_pinned_host_memory()
    assert torch.equal(snapshot, param.detach())


def test_noop_backuper_signature_compatible():
    param = torch.nn.Parameter(torch.randn(2, 3))
    backuper = TensorBackuper.create(_source_of({"w": param}), single_tag="actor")
    assert isinstance(backuper, _TensorBackuperNoop)

    # The on_device kwarg must be accepted (and ignored) for call-site compatibility.
    backuper.backup("actor", on_device=True)

    snapshot = backuper.get("actor")["w"]
    # get() returns a live detach of the source, not a copy.
    assert not snapshot.requires_grad
    assert snapshot.data_ptr() == param.data_ptr()
    assert torch.equal(snapshot, param.detach())


def _assert_backup_on_device_preserves_each_source_device(monkeypatch, devices: tuple[str, str]) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    from relax.utils.training import tensor_backper

    monkeypatch.setattr(tensor_backper.device_module, "synchronize", lambda: None)
    with FakeTensorMode():
        params = {name: torch.empty(3, device=device) for name, device in zip(("a", "b"), devices, strict=True)}
        backuper = TensorBackuper.create(_source_of(params), single_tag=None)
        backuper.backup("actor", on_device=True)
        first = dict(backuper.get("actor"))
        backuper.backup("actor", on_device=True)
        for name, param in params.items():
            assert first[name].device == param.device
            assert backuper.get("actor")[name] is first[name]


def test_backup_on_device_preserves_each_source_device(monkeypatch):
    _assert_backup_on_device_preserves_each_source_device(monkeypatch, ("cpu", "cuda:0"))
