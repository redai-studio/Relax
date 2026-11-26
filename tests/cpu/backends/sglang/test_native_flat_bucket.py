# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.model_runner_components.weight_updater import WeightUpdater
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


def test_receive_only_weight_updater_advertises_single_buffer_transport(monkeypatch):
    if not hasattr(WeightUpdater, "receive_weights_from_distributed"):
        pytest.skip("This SGLang version receives and loads weights in one call")
    method = "_receive_bucketed_weights_from_distributed"
    monkeypatch.setattr(WeightUpdater, method, getattr(WeightUpdater, method))
    module.install_native_flat_bucket()
    model, group = torch.nn.Module(), object()
    updater = WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=None,
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=None,
        recapture_cuda_graph=None,
        get_model_runner=None,
        _model_update_group={"native": group},
    )
    runner = SimpleNamespace(weight_updater=updater)
    manager = SimpleNamespace(_select_runners=lambda _: [("target", runner)])
    model._relax_native_weight_guard = SimpleNamespace(manager=manager)
    assert module.native_flat_bucket_supported(model)

    packed = torch.arange(12, dtype=torch.uint8)
    calls = []

    def broadcast(buffer, *, src, group):
        calls.append((buffer, src, group))
        buffer.copy_(packed)

    monkeypatch.setattr(module.dist, "broadcast", broadcast)
    weights = updater.receive_weights_from_distributed(
        names=["weight", "scale"],
        dtypes=["bfloat16", "uint8"],
        shapes=[[2, 2], [4]],
        group_name="native",
        load_format="flattened_bucket",
    )
    assert len(calls) == 1 and calls[0][1:] == (0, group)
    assert [(name, weight.dtype, tuple(weight.shape)) for name, weight in weights] == [
        ("weight", torch.bfloat16, (2, 2)),
        ("scale", torch.uint8, (4,)),
    ]
    assert torch.equal(weights[0][1].view(torch.uint8).flatten(), packed[:8])
    assert torch.equal(weights[1][1], packed[8:])
    assert all(weight.untyped_storage().data_ptr() == calls[0][0].data_ptr() for _, weight in weights)
