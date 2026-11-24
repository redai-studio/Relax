# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang import delta_loader as loader_module
from relax.backends.sglang.delta_loader import load_delta_weights
from relax.distributed.weight_sync.codec import WireBucket, encode


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
    ack = []
    if fields["op"] == "finalize":
        ack.append(("__delta_capability__", torch.full((1,), 255, dtype=torch.uint8)))
        fields.setdefault("version", model._relax_delta_receiver.active["control"]["version"])
    load_delta_weights(model, [(json.dumps(fields), torch.empty(0)), *ack])


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


def test_publication_finishes_before_sglang_publishes_rank_context(monkeypatch):
    class UninitializedParallel:
        @property
        def tp_rank(self):
            raise RuntimeError("parallel name 'tp_rank' has not been written in this process")

    model = TinyModel(torch.bfloat16)
    weights = torch.arange(4, dtype=torch.bfloat16)
    begin(model, "0", seed=True)
    frame(model, weights)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", SimpleNamespace(get_parallel=UninitializedParallel))
    control(model, op="finalize", frames=1)
    assert model._relax_delta_receiver.version == "0"
    assert torch.equal(model.weight, weights.view(2, 2).T)


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


def zipped_control(data):
    import base64
    import hashlib
    import zlib

    return dict(
        op="consume",
        version="1",
        frame=0,
        metadata_id=hashlib.sha256(data).hexdigest(),
        metadata_zlib={"bytes": len(data), "data": base64.b64encode(zlib.compress(data)).decode("ascii")},
    )


@pytest.mark.parametrize(
    "kind",
    [
        "sha",
        "truncated",
        "trailing",
        "concatenated",
        "size",
        "bomb",
        "too-big",
        "negative",
        "non-ascii",
        "schema",
        "bad-base64",
        "excess-base64",
        "mixed",
        "staged",
    ],
)
def test_invalid_inline_directory_fails_before_writes_and_cannot_replay(kind):
    import base64
    import zlib

    model = TinyModel(torch.bfloat16)
    begin(model, "0", seed=True)
    frame(model, torch.ones(4, dtype=torch.bfloat16))
    control(model, op="finalize", frames=1)
    begin(model, "1", base="0")
    metadata, raw = encode(
        [
            (
                "weight",
                (2, 2),
                torch.bfloat16,
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([-32768], dtype=torch.int16).view(torch.bfloat16),
            )
        ],
        max_bytes=1024,
    )
    data = json.dumps(metadata, separators=(",", ":")).encode("ascii")
    original = zipped_control(data)
    candidate = zipped_control(data)
    packed = candidate["metadata_zlib"]
    if kind in ("truncated", "trailing", "concatenated"):
        blob = base64.b64decode(packed["data"])
        packed["data"] = base64.b64encode(
            {"truncated": blob[:-1], "trailing": blob + b"junk", "concatenated": blob + blob}[kind]
        ).decode()
    elif kind == "sha":
        candidate["metadata_id"] = "0" * 64
    elif kind == "bomb":
        packed.update(bytes=32, data=base64.b64encode(zlib.compress(b"x" * (1 << 20))).decode())
    elif kind in ("size", "too-big", "negative"):
        packed["bytes"] = {"size": len(data) + 1, "too-big": (16 << 20) + 1, "negative": -1}[kind]
    elif kind == "non-ascii":
        candidate = zipped_control(b'"\xff"')
    elif kind == "schema":
        candidate = zipped_control(b'{"format":1,"bytes":1,"tensors":[]}')
    elif kind == "mixed":
        candidate["metadata"] = metadata
    elif kind == "staged":
        control(
            model, op="document", kind="frame", id=original["metadata_id"], index=0, count=2, data=data[:10].decode()
        )
    else:
        packed["data"] = "?" if kind == "bad-base64" else "A" * 65537
    before = model.weight.detach().clone()
    status = torch.full((1,), 73, dtype=torch.uint8)
    with pytest.raises(ValueError):
        load_delta_weights(model, [(json.dumps(candidate), raw), ("__delta_capability__", status)])
    assert torch.equal(model.weight, before) and status.tolist() == [73]
    assert model._relax_delta_receiver.version == "0"
    with pytest.raises(RuntimeError, match="failed update"):
        load_delta_weights(model, [(json.dumps(original), raw), ("__delta_capability__", status)])
    assert torch.equal(model.weight, before) and status.tolist() == [73]


def test_inline_consume_rejects_duplicate_frame_in_source_order():
    model = TinyModel(torch.float32)
    metadata, raw = encode(
        [("weight", (2, 2), torch.float32, None, torch.arange(4, dtype=torch.float32))], max_bytes=1024
    )
    begin(model, "1", seed=True)
    ctl = zipped_control(json.dumps(metadata, separators=(",", ":")).encode("ascii"))
    weights = [(json.dumps(ctl), raw), ("__delta_capability__", torch.zeros(1, dtype=torch.uint8))]
    load_delta_weights(model, weights)
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="once in source order"):
        load_delta_weights(model, weights)
    assert torch.equal(before, model.weight)


def test_packed_wire_keeps_ordinary_loader_side_effects_and_rejects_invalid_boundaries():
    class Recorder(TinyModel):
        def __init__(self):
            super().__init__(torch.float32)
            self.calls = []

        def load_weights(self, weights):
            self.calls.append([(n, v.tolist()) for n, v in weights])

    bucket = WireBucket(16)
    for i in range(3):
        item = (str(i), (1,), torch.int16, None, torch.tensor([i], dtype=torch.int16))
        result = bucket.offer(*encode([item], max_bytes=16))
        if i < 2:
            assert result is None
    directory, raw = result
    assert directory["bytes"] == 10
    assert directory["loader_ends"] == [1, 2]
    assert raw[2:8].tolist() == [0] * 6
    final, _ = bucket.finish()
    assert "loader_ends" not in final
    assert bucket.finish() is None

    def install(model, metadata):
        begin(model, "seed", seed=True)
        command = dict(op="consume", version="seed", frame=0, metadata=metadata)
        status = torch.zeros(2, dtype=torch.uint8)
        load_delta_weights(model, [(json.dumps(command), raw), ("__delta_capability__", status)])
        control(model, op="finalize", frames=1)
        assert status.tolist() == [1, 1]

    model = Recorder()
    install(model, directory)
    assert model.calls == [[("0", [0])], [("1", [1])]]
    for invalid_ends in ([0, 2], [1, 1], [1], [False, 2]):
        model = Recorder()
        with pytest.raises(ValueError, match="boundaries"):
            install(model, dict(directory, loader_ends=invalid_ends))
        assert not model.calls


@pytest.mark.parametrize("fault", ["digest", "length", "compressed_bytes", "status_alias", "loader_ends"])
def test_ipc_directory_rejects_invalid_tail_before_writing(fault):
    import hashlib
    import zlib

    from relax.backends.sglang.weight_sync_proxy import _status_buffer

    model = TinyModel(torch.bfloat16)
    control = dict(op="begin", seed=True, version="seed", receiver_gpu_budget_bytes=1 << 20, staging_budget_bytes=1024)
    load_delta_weights(model, [(json.dumps(control), torch.empty(0))])
    values = torch.tensor([11, 12, 13, 14], dtype=torch.bfloat16)
    metadata, raw = encode([("weight", (2, 2), values.dtype, None, values)], max_bytes=1024)
    if fault == "loader_ends":
        metadata["loader_ends"] = [2]
    canonical = json.dumps(metadata).encode()
    compressed = zlib.compress(canonical)
    ctl = dict(
        op="consume",
        version="seed",
        frame=0,
        metadata_id=hashlib.sha256(canonical).hexdigest(),
        metadata_ipc=[1, len(compressed), len(canonical)],
    )
    ack = _status_buffer(1, compressed, "cpu")
    ack[0] = 99
    if fault == "digest":
        ctl["metadata_id"] = "0" * 64
    elif fault == "length":
        ctl["metadata_ipc"][2] -= 1
    elif fault == "compressed_bytes":
        ack[-1] ^= 127
    elif fault == "status_alias":
        raw = ack[: raw.numel()]
    with pytest.raises(ValueError):
        load_delta_weights(model, [(json.dumps(ctl), raw), ("__delta_capability__", ack)])
    assert int(ack[0]) == 99 and not model.weight.any()
    assert model._relax_delta_receiver.failed and model._relax_delta_receiver.version is None


def test_receiver_replays_complete_native_loader_inputs_from_bounded_host_seed(monkeypatch):
    class Derived(TinyModel):
        def __init__(self):
            super().__init__(torch.float32)
            self.register_buffer("derived", torch.zeros_like(self.weight))

        def load_weights(self, weights):
            for name, value in weights:
                assert name == "weight" and type(value) is torch.Tensor
                self.weight.copy_(value.T)
                self.derived.copy_(value.square())

    monkeypatch.setattr(loader_module, "sparse_loader_identity", lambda model: (Derived.load_weights,))
    monkeypatch.setattr(loader_module, "requires_full_input", lambda model, name: True)
    model, reference = Derived(), Derived()
    raw = torch.tensor([0x7FC00017, 0, 1065353216, -2147483648], dtype=torch.int32)
    for version, seed in enumerate((True, False, False, True, False, False)):
        cur = str(version)
        control(
            model,
            op="begin",
            version=cur,
            base=str(version - 1),
            seed=seed,
            receiver_gpu_budget_bytes=4096,
            receiver_host_cache_bytes=raw.nbytes,
            staging_budget_bytes=1024,
        )
        pos = torch.tensor([0, 1], dtype=torch.int32)
        raw[0] += 1
        raw[1] = -2147483648 if version % 2 else 0
        status = frame(
            model, raw.view(torch.float32) if seed else raw[:2].view(torch.float32), positions=None if seed else pos
        )
        assert status.tolist() == [0]
        control(model, op="finalize", frames=1)
        reference.load_weights([("weight", raw.view(torch.float32).view(2, 2))])
        for name in ("weight", "derived"):
            assert torch.equal(getattr(model, name).view(torch.int32), getattr(reference, name).view(torch.int32))
        assert model._relax_delta_receiver._host_input_bytes == raw.nbytes

    control(
        model,
        op="begin",
        version="6",
        seed=True,
        receiver_gpu_budget_bytes=4096,
        receiver_host_cache_bytes=raw.nbytes - 1,
        staging_budget_bytes=1024,
    )
    assert frame(model, raw.view(torch.float32)).tolist() == [1]
    assert model._relax_delta_receiver._host_input_bytes == 0


def test_ordinary_delta_requires_final_ack_before_advancing_version():
    receiver = TinyModel(torch.float32)
    begin(receiver, "0", seed=True)
    frame(receiver, torch.arange(4, dtype=torch.float32))
    control(receiver, op="finalize", frames=1)
    begin(receiver, "1", base="0")
    frame(receiver, torch.ones(1), positions=torch.tensor([0], dtype=torch.int32))
    with pytest.raises(ValueError, match="acknowledged capability"):
        load_delta_weights(receiver, [(json.dumps(dict(op="finalize", version="1", frames=1)), torch.empty(0))])
    assert receiver._relax_delta_receiver.version == "0" and receiver._relax_delta_receiver.failed
