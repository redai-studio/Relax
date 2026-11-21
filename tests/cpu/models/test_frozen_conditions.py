# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Cache regressions: bounded backing storage, keys, autograd and rank
misses."""

import weakref

import pytest
import torch
from PIL import Image

from relax.models.frozen_conditions import FrozenConditionCache, agree_condition_request, image_condition_key


def test_storage_budget_lru_and_oversized_admission():
    cache = FrozenConditionCache(64, 2)
    backing = torch.arange(8, dtype=torch.float32)
    cache.put(b"a", {"first": backing[:1], "second": backing[2:]})
    assert cache.resident_bytes == 32  # backing bytes, shared views counted once
    cache.put(b"b", {"x": torch.zeros(8)})
    assert cache.lookup(b"a").tensors["first"].untyped_storage().data_ptr() == backing.data_ptr()
    cache.put(b"c", {"x": torch.ones(8)})
    assert cache.lookup(b"b").tensors is None
    assert cache.resident_bytes == cache.peak_bytes == 64
    cache.put(b"large", {"x": torch.zeros(100)})
    assert cache.resident_bytes == 64
    assert cache.metrics()["oversized"] == cache.metrics()["evictions"] == 1
    cache.clear()
    assert cache.metrics()["entries"] == cache.resident_bytes == 0


def test_entry_limit_graph_rejection_and_release():
    cache = FrozenConditionCache(1024, 1)
    tensor = torch.ones(1, dtype=torch.bfloat16)
    reference = weakref.ref(tensor)
    cache.put(b"a", {"x": tensor})
    del tensor
    assert reference() is not None
    cache.put(b"b", {"x": torch.zeros(1)})
    assert reference() is None
    with pytest.raises(ValueError, match="autograd"):
        cache.put(b"c", {"x": torch.ones(1, requires_grad=True) * 2})


def test_borrowed_frozen_tensor_can_participate_in_generation_backward():
    cache = FrozenConditionCache(1024, 2)
    with torch.no_grad():
        condition = torch.randn(3, 4)
    cache.put(b"key", {"x": condition})
    cached = cache.lookup(b"key").tensors["x"]
    assert cached is condition and cached.dtype == condition.dtype
    weight = torch.nn.Parameter(torch.randn(4, 2))
    (cached @ weight).square().sum().backward()
    assert torch.isfinite(weight.grad).all()
    assert agree_condition_request(cache.lookup(b"key"), False).tensors is None
    assert agree_condition_request(cache.lookup(b"key"), True).tensors["x"] is condition


def test_actual_inputs_and_geometry_determine_key():
    data = {"prompt": "edit", "image": Image.new("RGB", (32, 32)), "edit_image": [Image.new("RGB", (32, 32))]}

    def key(value=data, version="v1", dtype=torch.bfloat16):
        return image_condition_key(value, version=version, dtype=dtype)

    original = key()
    assert key({**data, "image": Image.new("RGB", (32, 32), "red")}) == original
    assert key({**data, "prompt": ""}) != original
    assert key({**data, "image": Image.new("RGB", (64, 32))}) != original
    assert key(version="v2") != original
    assert key(dtype=torch.float32) != original
    data["edit_image"][0].putpixel((0, 0), (255, 0, 0))
    assert key() != original
