# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""ART wire equivalence, receiver ownership and bounded parallel copies."""

import pickle

import pytest
import torch

from relax.utils.art_tensor_transport import TensorBuffer, pack, unpack, unpack_shards


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    ],
)
def test_transport_preserves_bytes_and_alias_with_owned_storage(dtype):
    x = torch.arange(24).to(dtype).reshape(2, 3, 4)
    shared = {"a": x}
    obj = {"a": x, "nested": [x, ("tag", 4)], "metadata": shared, "env": {"metadata": shared}}
    encoded = pack(obj, min_bytes=0)
    assert isinstance(encoded["a"], TensorBuffer)
    # Protocol-5 out-of-band buffers model the relevant Ray numpy transport.
    buffers = []
    serialized = pickle.dumps(encoded, protocol=5, buffer_callback=buffers.append)
    received = pickle.loads(serialized, buffers=[b.raw().toreadonly() for b in buffers])
    decoded = unpack(received)
    assert decoded["a"] is decoded["nested"][0]
    assert decoded["metadata"] is decoded["env"]["metadata"]
    assert decoded["a"].dtype == x.dtype and decoded["a"].shape == x.shape
    assert torch.equal(decoded["a"].reshape(-1).view(torch.uint8), x.reshape(-1).view(torch.uint8))
    before = x.clone()
    decoded["a"].zero_()
    assert torch.equal(x, before)


def test_transport_preserves_special_float_bits():
    x = torch.tensor([0.0, -0.0, float("nan"), float("inf"), -float("inf")])
    assert torch.equal(unpack(pack(x, min_bytes=0)).view(torch.uint8), x.view(torch.uint8))


def test_transport_fallback_and_scalar():
    for x in (
        torch.empty(0),
        torch.arange(20).reshape(4, 5).T,
        torch.ones(2, requires_grad=True),
        torch.ones(2, dtype=torch.complex64),
    ):
        assert pack(x, min_bytes=0) is x
    x = torch.tensor(3.0, dtype=torch.bfloat16)
    assert unpack(pack(x, min_bytes=0)).shape == torch.Size([])


def test_distinct_views_retain_original_transport():
    source = torch.arange(20.0)
    a, b = source[:10], source[5:15]
    encoded = pack({"a": a, "b": b}, min_bytes=0)
    assert encoded["a"] is a and encoded["b"] is b


def test_threshold_and_nonzero_storage_offset():
    below = torch.ones(1048576 - 1, dtype=torch.uint8)
    at = torch.ones(1048576, dtype=torch.uint8)
    assert pack(below) is below
    assert isinstance(pack(at), TensorBuffer)
    source = torch.arange(30.0, dtype=torch.float32)
    view = source[5:20]
    result = unpack(pack(view, min_bytes=0))
    assert view.storage_offset() > 0 and torch.equal(result, view)
    result.zero_()
    assert torch.equal(source, torch.arange(30.0, dtype=torch.float32))


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_unpacks_are_equal_and_independent(workers):
    x = torch.arange(120, dtype=torch.float32)
    items = [pack({"a": x + i}, min_bytes=0) for i in range(8)]
    restored = unpack_shards(items, max_workers=workers)
    for i, value in enumerate(restored):
        assert torch.equal(value["a"], x + i)
        value["a"].zero_()
        assert torch.equal(unpack(items[i])["a"], x + i)


def test_parallel_completion_order_and_bound(monkeypatch):
    import threading
    import time

    import relax.utils.art_tensor_transport as transport

    lock = threading.Lock()
    active = peak = 0

    def fake(value):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(active, peak)
        time.sleep(0.03 if value % 2 == 0 else 0.005)
        with lock:
            active -= 1
        return value

    monkeypatch.setattr(transport, "unpack", fake)
    assert transport.unpack_shards(list(range(8)), max_workers=2) == list(range(8))
    assert peak == 2 and active == 0


def test_parallel_failure_joins_running_copy(monkeypatch):
    import threading
    import time

    import relax.utils.art_tensor_transport as transport

    entered, finished = threading.Event(), threading.Event()

    def fake(value):
        if value == 0:
            assert entered.wait(2)
            raise ValueError("injected copy failure")
        entered.set()
        time.sleep(0.03)
        finished.set()
        return value

    monkeypatch.setattr(transport, "unpack", fake)
    with pytest.raises(ValueError, match="injected copy failure"):
        transport.unpack_shards([0, 1, 2], max_workers=2)
    assert finished.is_set()
