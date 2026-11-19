# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CUDA snapshot staging and TP metadata regression tests."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from relax.distributed.checkpoint_service.backends.device_direct import DeviceDirectBackend


def _mark_tp(tensor: torch.Tensor, *, parallel_mode="column", partition_dim=0, partition_stride=1):
    """Attach Megatron TP attributes the way Megatron marks a column-parallel
    param."""
    tensor.tensor_model_parallel = True
    tensor.partition_dim = partition_dim
    tensor.partition_stride = partition_stride
    tensor.parallel_mode = parallel_mode
    return tensor


def _bare_backend(*, weights_getter, device, use_bridge=False) -> DeviceDirectBackend:
    """Build a DeviceDirectBackend without running its heavy ``__init__``.

    ``_named_params_and_buffers`` only reads the handful of attributes set
    here, so we bypass the constructor (which pulls in httpx, megatron dep
    loading, and thread pools) and wire them by hand.
    """
    backend = DeviceDirectBackend.__new__(DeviceDirectBackend)
    backend._weights_getter = weights_getter
    backend.device = device
    backend.args = Namespace()
    backend.model = [torch.nn.Linear(2, 2)]
    backend._use_bridge = use_bridge
    backend._megatron = SimpleNamespace()  # live-path tests replace this
    return backend


@pytest.mark.skipif(not torch.cuda.is_available(), reason="H2D coercion requires CUDA")
def test_weights_getter_host_snapshot_h2d_copies_tp_attrs():
    cpu_tensor = _mark_tp(torch.randn(4, 3))
    backend = _bare_backend(
        weights_getter=lambda: {"module.layer.weight": cpu_tensor},
        device=torch.device("cuda:0"),
    )

    [(name, tensor)] = list(backend._named_params_and_buffers())

    assert name == "module.layer.weight"
    assert tensor.device.type == "cuda"
    assert torch.equal(tensor.cpu(), cpu_tensor)
    # TP attributes must survive the .to(device) reallocation.
    assert tensor.tensor_model_parallel is True
    assert tensor.partition_dim == 0
    assert tensor.partition_stride == 1
    assert tensor.parallel_mode == "column"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device passthrough requires CUDA")
def test_weights_getter_device_snapshot_passthrough():
    gpu_tensor = _mark_tp(torch.randn(2, 2, device="cuda"))
    backend = _bare_backend(
        weights_getter=lambda: {"w": gpu_tensor},
        device=torch.device("cuda:0"),
    )

    [(name, tensor)] = list(backend._named_params_and_buffers())

    assert name == "w"
    # Already device-resident: yielded verbatim, no reallocation.
    assert tensor is gpu_tensor


class _ExplodingH2DTensor(torch.Tensor):
    """Raises on any ``.to()`` copy — stands in for a filtered-out CPU snapshot
    tensor."""

    def to(self, *args, **kwargs):
        raise AssertionError("filtered-out tensor must never be moved to device")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="name_filter/H2D ordering requires CUDA")
def test_weights_getter_name_filter_applied_before_h2d():
    keep1 = torch.randn(4)
    keep2 = torch.randn(2, 2)
    expert = torch.randn(8).as_subclass(_ExplodingH2DTensor)
    snapshot = {
        "layer0.mlp.weight": keep1,
        "layer1.mlp.weight": keep2,
        "layer0.mlp.experts.w1": expert,
    }
    backend = _bare_backend(weights_getter=lambda: snapshot, device=torch.device("cuda:0"))

    pairs = list(backend._named_params_and_buffers(name_filter=lambda n: ".experts." not in n))

    # Only the non-expert tensors are yielded; the expert tensor was filtered out
    # before the H2D branch (its .to() would have raised otherwise).
    assert [name for name, _ in pairs] == ["layer0.mlp.weight", "layer1.mlp.weight"]
    for _, tensor in pairs:
        assert tensor.device.type == "cuda"
