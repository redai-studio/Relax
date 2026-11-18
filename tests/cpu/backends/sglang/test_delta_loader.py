# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest
import torch

from relax.backends.sglang import delta_loader as loader_module
from relax.backends.sglang.delta_loader import load_delta_weights
from relax.distributed.weight_sync.codec import encode


class TinyModel(torch.nn.Module):
    def __init__(self, dtype, *, transform=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2, dtype=dtype), requires_grad=False)
        self.transform = transform

    def load_weights(self, weights):
        for name, value in weights:
            assert name == "weight"
            # Exercise the normal model-owned layout responsibility.
            self.weight.copy_(value.transpose(0, 1))
            if self.transform:
                self.weight.exp_()


@pytest.fixture(autouse=True)
def known_cpu_copy_adapter(monkeypatch):
    # Stand in only for the external SG capability boundary. The actual codec,
    # receiver, layout operations and writes under test remain production code.
    original = loader_module.sparse_loader_identity

    def identity(model):
        if type(model).load_weights is TinyModel.load_weights and not model.transform:
            return (TinyModel.load_weights,)
        return original(model)

    monkeypatch.setattr(loader_module, "sparse_loader_identity", identity)


def control(model, **fields):
    if fields["op"] == "finalize":
        fields.setdefault("version", model._relax_delta_receiver.active["control"]["version"])
    load_delta_weights(model, [(json.dumps(fields), torch.empty(0))])


def frame(model, values, *, positions=None, ordinal=0):
    metadata, raw = encode([("weight", (2, 2), values.dtype, positions, values.flatten())], max_bytes=1024)
    status = torch.zeros(1, dtype=torch.uint8)
    payload = dict(
        op="consume",
        version=model._relax_delta_receiver.active["control"]["version"],
        frame=ordinal,
        metadata=metadata,
    )
    load_delta_weights(model, [(json.dumps(payload), raw), ("__delta_capability__", status)])
    return status


def begin(model, version, *, seed=False, base=None):
    control(
        model,
        op="begin",
        version=version,
        base=base,
        seed=seed,
        receiver_gpu_budget_bytes=4096,
        staging_budget_bytes=1024,
    )


@pytest.mark.parametrize(
    "dtype,words,nan1,nan2,minus_zero",
    [
        (torch.bfloat16, torch.int16, 0x7FC1, 0x7FC2, -32768),
        (torch.float32, torch.int32, 0x7FC00001, 0x7FC00002, -2147483648),
    ],
)
def test_delta_receiver_matches_independent_full_loader_with_true_nan_bits(dtype, words, nan1, nan2, minus_zero):
    receiver, reference = TinyModel(dtype), TinyModel(dtype)
    raw = torch.tensor([nan1, 0, 3, 4], dtype=words)
    begin(receiver, "0", seed=True)
    status = frame(receiver, raw.view(dtype))
    assert status.tolist() == [0]
    control(receiver, op="finalize", frames=1)
    changed = torch.tensor([nan2, minus_zero], dtype=words)
    raw[:2] = changed
    begin(receiver, "1", base="0")
    frame(receiver, changed.view(dtype), positions=torch.tensor([0, 1], dtype=torch.int32))
    control(receiver, op="finalize", frames=1)
    with torch.no_grad():
        reference.load_weights([("weight", raw.view(dtype).reshape(2, 2))])
    assert torch.equal(receiver.weight.view(words), reference.weight.view(words))
    begin(receiver, "2", base="1")
    frame(receiver, torch.empty(0, dtype=dtype), positions=torch.empty(0, dtype=torch.int32))
    control(receiver, op="finalize", frames=1)
    assert torch.equal(receiver.weight.view(words), reference.weight.view(words))


def test_full_seed_marks_post_copy_numeric_conversion_for_full_fallback():
    receiver = TinyModel(torch.float32, transform=True)
    begin(receiver, "0", seed=True)
    status = frame(receiver, torch.arange(4, dtype=torch.float32))
    assert status.tolist() == [1]
    control(receiver, op="finalize", frames=1)
    assert torch.equal(receiver.weight, torch.arange(4, dtype=torch.float32).reshape(2, 2).T.exp())
    before = receiver.weight.detach().clone()
    begin(receiver, "1", base="0")
    with pytest.raises(ValueError, match="complete conversion group as FULL"):
        frame(receiver, torch.tensor([3.0]), positions=torch.tensor([0], dtype=torch.int32))
    assert torch.equal(receiver.weight, before)


def test_invalid_sparse_coordinates_fail_before_mutation_and_retire_receiver():
    receiver = TinyModel(torch.float32)
    begin(receiver, "0", seed=True)
    frame(receiver, torch.ones(4))
    control(receiver, op="finalize", frames=1)
    before = receiver.weight.detach().clone()
    begin(receiver, "1", base="0")
    with pytest.raises(ValueError, match="sorted, unique"):
        frame(receiver, torch.ones(2), positions=torch.tensor([2, 2], dtype=torch.int32))
    assert torch.equal(receiver.weight, before)
    with pytest.raises(RuntimeError, match="failed update"):
        begin(receiver, "2", seed=True)


def test_unknown_storage_loader_uses_ordinary_full_without_speculative_writes():
    class ReplacingModel(TinyModel):
        def load_weights(self, weights):
            for _name, value in weights:
                self.weight.data = value.reshape(2, 2)
                self.weight.mul_(2)

    receiver = ReplacingModel(torch.float32)
    begin(receiver, "0", seed=True)
    status = frame(receiver, torch.arange(4, dtype=torch.float32))
    control(receiver, op="finalize", frames=1)
    assert status.tolist() == [1]
    assert torch.equal(receiver.weight, torch.arange(4, dtype=torch.float32).reshape(2, 2) * 2)


def test_schema_change_is_rejected_before_an_ordinary_loader_can_mutate():
    class ReshapingModel(TinyModel):
        def load_weights(self, weights):
            for _name, value in weights:
                self.weight.copy_(value.reshape(2, 2))

    receiver = ReshapingModel(torch.float32)
    begin(receiver, "0", seed=True)
    frame(receiver, torch.ones(4))
    control(receiver, op="finalize", frames=1)
    before = receiver.weight.detach().clone()
    begin(receiver, "1", base="0")
    metadata, raw = encode([("weight", (4,), torch.float32, None, torch.arange(4.0))], max_bytes=1024)
    payload = dict(op="consume", version="1", frame=0, metadata=metadata)
    with pytest.raises(ValueError, match="acknowledged schema"):
        load_delta_weights(
            receiver, [(json.dumps(payload), raw), ("__delta_capability__", torch.zeros(1, dtype=torch.uint8))]
        )
    assert torch.equal(receiver.weight, before)


def test_unknown_full_loader_buffer_side_effect_runs_once_and_cache_stays_ordinary():
    class StatefulModel(TinyModel):
        def __init__(self):
            super().__init__(torch.float32)
            self.register_buffer("loads", torch.zeros((), dtype=torch.int64))

        def load_weights(self, weights):
            for _name, value in weights:
                self.weight.copy_(value.reshape(2, 2))
                self.loads.add_(1)
                self.cached = value.clone()

    model = StatefulModel()
    begin(model, "0", seed=True)
    status = frame(model, torch.arange(4, dtype=torch.float32))
    control(model, op="finalize", frames=1)
    assert status.tolist() == [1]
    assert int(model.loads) == 1
    assert type(model.cached) is torch.Tensor
    assert float(model.cached.sum()) == 6
