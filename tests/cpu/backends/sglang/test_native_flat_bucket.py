# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import torch
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

from relax.backends.sglang import native_flat_bucket as module


def test_flat_native_receive_preserves_mixed_weight_scale_bytes(monkeypatch):
    values = [
        ("weight", torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)),
        ("scale", torch.tensor([0, 128, 255, 1], dtype=torch.uint8)),
        ("bias", torch.tensor([-1.25], dtype=torch.float32)),
    ]
    assert module.aligned_bucket(values)
    source = FlattenedTensorBucket(named_tensors=values).get_flattened_tensor()
    loaded, calls = [], []
    group = object()

    def broadcast(buffer, *, src, group):
        calls.append((src, group))
        buffer.copy_(source)

    monkeypatch.setattr(module.dist, "broadcast", broadcast)
    updater = SimpleNamespace(
        device="cpu",
        _model_update_group={"native": group},
        get_model=lambda: SimpleNamespace(load_weights=lambda weights: loaded.extend(weights)),
    )
    result, _ = module.receive_flat_bucket(
        updater,
        [name for name, _ in values],
        [str(value.dtype).removeprefix("torch.") for _, value in values],
        [list(value.shape) for _, value in values],
        "native",
    )
    assert result and calls == [(0, group)]
    assert [name for name, _ in loaded] == [name for name, _ in values]
    for (_, expected), (_, actual) in zip(values, loaded, strict=True):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert not module.aligned_bucket([("packed", torch.ones(3, dtype=torch.uint8)), ("scale", torch.ones(1))])
