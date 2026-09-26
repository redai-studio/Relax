# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Contract tests for the cross-node DCS weight-update transport."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


class _Tensor:
    def __init__(self, values, dtype):
        self.values = list(values)
        self.dtype = dtype
        self.shape = (len(self.values),)
        self.data = self
        self.view_dtypes = []

    def flatten(self):
        return self

    def view(self, _dtype):
        self.view_dtypes.append(_dtype)
        return self


class _FlatTensor:
    def __init__(self, values):
        self.values = list(values)


class _Torch:
    Tensor = _Tensor
    uint8 = "uint8"

    @staticmethod
    def cat(tensors, dim=0):
        assert dim == 0
        values = []
        for tensor in tensors:
            values.extend(tensor.values)
        return _FlatTensor(values)


class _Dist:
    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def broadcast(self, tensor, src, group, **kwargs):
        self.calls.append((tensor, src, group, kwargs))
        if self.fail:
            raise RuntimeError("broadcast failed")
        return SimpleNamespace(wait=lambda: None)


def _load_update_bucket(dist):
    path = ROOT / "relax/distributed/checkpoint_service/backends/device_direct.py"
    tree = ast.parse(path.read_text())
    backend = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeviceDirectBackend")
    method = next(
        node for node in backend.body if getattr(node, "name", None) == "_update_bucket_weights_from_distributed"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = {
        "torch": _Torch,
        "dist": dist,
        "ray": SimpleNamespace(get=lambda ref: True),
        "time": SimpleNamespace(sleep=lambda _seconds: None),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["_update_bucket_weights_from_distributed"]


def _backend(dist, *, fully_async=True):
    payloads = []
    lock_events = []
    backend = SimpleNamespace(
        args=SimpleNamespace(fully_async=fully_async),
        lock=SimpleNamespace(
            acquire=SimpleNamespace(remote=lambda: lock_events.append("acquire") or object()),
            release=SimpleNamespace(remote=lambda: lock_events.append("release") or object()),
        ),
        _group_name="slime-pp_0",
        weight_version=1,
        _model_update_groups="group",
        _batch_request=lambda endpoint, payload: payloads.append((endpoint, payload)) or [object()],
    )
    backend._payloads = payloads
    backend._lock_events = lock_events
    backend._dist = dist
    return backend


def test_fully_async_weight_bucket_uses_one_flattened_collective():
    dist = _Dist()
    method = _load_update_bucket(dist)
    backend = _backend(dist)
    tensors = [_Tensor([1, 2], "bf16"), _Tensor([3], "bf16")]
    method(backend, [("a", tensors[0]), ("b", tensors[1])])

    payload = backend._payloads[0][1]
    assert payload["load_format"] == "flattened_bucket"
    assert len(dist.calls) == 1
    assert dist.calls[0][0].values == [1, 2, 3]
    assert tensors[0].view_dtypes == ["uint8"]
    assert tensors[1].view_dtypes == ["uint8"]


def test_mixed_dtype_weight_bucket_uses_byte_flattened_protocol():
    dist = _Dist()
    method = _load_update_bucket(dist)
    backend = _backend(dist)
    method(backend, [("a", _Tensor([1], "bf16")), ("b", _Tensor([2], "fp32"))])

    payload = backend._payloads[0][1]
    assert payload["load_format"] == "flattened_bucket"
    assert len(dist.calls) == 1


def test_weight_bucket_releases_lock_after_broadcast_failure():
    dist = _Dist(fail=True)
    method = _load_update_bucket(dist)
    backend = _backend(dist)

    with pytest.raises(RuntimeError, match="broadcast failed"):
        method(backend, [("a", _Tensor([1], "bf16"))])
    assert backend._lock_events == ["acquire", "release"]
