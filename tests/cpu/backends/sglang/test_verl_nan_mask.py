# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU weight publication through the production verl wire and scheduler
loader."""

import json
import math

import pytest
import torch

from relax.backends.sglang import delta_loader, verl_nan_mask
from relax.distributed.weight_sync import codec


class NativeCopyModel(torch.nn.Module):
    """The native dispatcher owns each TP slice and every model loader
    invocation."""

    def __init__(self, templates, *, tp_axis=0, tp_rank=0, tp_size=1):
        super().__init__()
        self.tp_axis, self.tp_rank, self.tp_size = tp_axis, tp_rank, tp_size
        self.weights = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(torch.zeros_like(self.local_slice(value)), requires_grad=False)
                for name, value in templates.items()
            }
        )
        self.register_buffer("untargeted", torch.tensor([17, 29], dtype=torch.int32))
        self.calls = []

    def local_slice(self, value):
        width, remainder = divmod(value.shape[self.tp_axis], self.tp_size)
        assert not remainder
        return value.narrow(self.tp_axis, self.tp_rank * width, width)

    def load_weights(self, weights):
        self.calls.append([name for name, _ in weights])
        for name, value in weights:
            self.weights[name].copy_(self.local_slice(value))


def row(name, shape, values, positions=None):
    indices = None if positions is None else torch.tensor(positions, dtype=torch.int32)
    return name, shape, values.dtype, indices, values.reshape(-1)


def packed(*groups, budget=1024):
    bucket = codec.WireBucket(budget)
    frames = []
    for group in groups:
        ready = bucket.offer(*codec.encode(group, max_bytes=budget))
        if ready is not None:
            frames.append(ready)
    tail = bucket.finish()
    if tail is not None:
        frames.append(tail)
    return frames


def install(model, frames, *, budget=1024):
    for metadata, raw in frames:
        codec.validate_directory(metadata, max_bytes=budget)
        assert raw.nbytes == metadata["bytes"]
        for spec, positions, values in codec.frame_flushes(metadata, raw):
            for blob in (positions, values):
                if blob.numel():
                    assert blob.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
            verl_nan_mask.apply_flush(model, spec, positions, values)


@pytest.mark.parametrize(
    ("dtype", "word", "one", "two", "minus_zero", "old_nan", "new_nan", "new_payload", "untouched_nan"),
    [
        (torch.bfloat16, torch.int16, 0x3F80, 0x4000, -32768, 0x7FC1, 0x7FC2, 0x7FC3, 0x7F81),
        (
            torch.float32,
            torch.int32,
            0x3F800000,
            0x40000000,
            -(2**31),
            0x7FC00001,
            0x7FC00002,
            0x7FC00003,
            0x7F800001,
        ),
    ],
)
def test_full_writes_true_nan_and_sparse_treats_it_as_an_unmodified_coordinate(
    dtype, word, one, two, minus_zero, old_nan, new_nan, new_payload, untouched_nan
):
    initial = torch.tensor([minus_zero, 0, one, one, old_nan, untouched_nan], dtype=word)
    reference = torch.tensor([0, minus_zero, new_nan, two, new_payload, untouched_nan], dtype=word)
    names = ("weight", "empty", "unlisted")
    model = NativeCopyModel({name: initial.view(dtype) for name in names})
    install(model, packed([row(name, (6,), initial.view(dtype)) for name in names]))
    for parameter in model.weights.values():
        assert torch.equal(parameter.view(word), initial)

    # Positions 2 and 4 deliberately carry real NaNs. Fixed verl treats both as
    # placeholders; the expected sparse bits differ from the independent FULL.
    changed = reference[:5].view(dtype)
    empty = changed[:0]
    frames = packed([row("weight", (6,), changed, [0, 1, 2, 3, 4]), row("empty", (6,), empty, [])])
    install(model, frames)

    dense_oracle = NativeCopyModel({"weight": reference.view(dtype)})
    install(dense_oracle, packed([row("weight", (6,), reference.view(dtype))]))
    sparse_bits = model.weights["weight"].view(word)
    assert torch.equal(sparse_bits, torch.tensor([0, minus_zero, one, two, old_nan, untouched_nan], dtype=word))
    assert (sparse_bits != dense_oracle.weights["weight"].view(word)).nonzero().flatten().tolist() == [2, 4]
    assert torch.equal(model.weights["empty"].view(word), initial)
    assert torch.equal(model.weights["unlisted"].view(word), initial)
    assert model.calls == [list(names), ["weight", "empty"]]
    assert model.untargeted.tolist() == [17, 29]

    # Updating an existing NaN to a non-NaN is still representable.
    install(model, packed([row("weight", (6,), torch.tensor([2.0], dtype=dtype), [4])]))
    assert model.weights["weight"].view(word)[4] == two


@pytest.mark.parametrize(("shape", "axis", "tp_size"), [((6, 5), 0, 3), ((2, 8, 3), 1, 2)])
def test_native_tp_slice_and_loader_invocations_survive_packing(shape, axis, tp_size):
    before = (torch.arange(math.prod(shape), dtype=torch.float32) + 1).reshape(shape)
    indices = [0, before.numel() // 2, before.numel() - 1]
    changed = torch.tensor([-0.0, -2.0, 5.0], dtype=torch.float32)
    expected = before.clone()
    expected.view(torch.int32).reshape(-1)[indices] = changed.view(torch.int32)
    frames = packed(
        [row("table", shape, before)],
        [row("table", shape, changed, indices)],
        [row("table", shape, changed[:0], [])],
    )
    assert len(frames) == 1
    assert [group["encoding"] for group in frames[0][0]["flushes"]] == ["dense", "indices", "indices"]
    for rank in range(tp_size):
        model = NativeCopyModel({"table": before}, tp_axis=axis, tp_rank=rank, tp_size=tp_size)
        install(model, frames)
        local_width = shape[axis] // tp_size
        oracle = expected.view(torch.int32).narrow(axis, rank * local_width, local_width)
        assert torch.equal(model.weights["table"].view(torch.int32), oracle)
        assert model.calls == [["table"], ["table"], ["table"]]


def test_mixed_dense_and_sparse_flushes_are_typed_and_ordered():
    float_words = torch.tensor([0x7FC00005, 0x3F800000], dtype=torch.int32)
    index_values = torch.tensor([11, 22], dtype=torch.int32)
    floating = float_words.view(torch.float32)
    model = NativeCopyModel({"float_weight": floating, "int_weight": index_values})
    model.weights["float_weight"].fill_(1.0)
    frames = packed(
        [
            row("float_weight", (2,), floating),
            row("int_weight", (2,), index_values),
            row("float_weight", (2,), torch.tensor([-0.0]), [1]),
        ]
    )
    ((metadata, _),) = frames
    assert [group["encoding"] for group in metadata["flushes"]] == ["dense", "dense", "indices"]
    install(model, frames)
    assert model.calls == [["float_weight"], ["int_weight"], ["float_weight"]]
    assert model.weights["int_weight"].dtype == torch.int32
    assert torch.equal(model.weights["int_weight"], index_values)
    assert model.weights["float_weight"].view(torch.int32).tolist() == [0x7FC00005, -(2**31)]

    # The NaN-sentinel codec supports floating sparse storage. Integer FULL is
    # eligible; encoding sparse integer or complex input fails before delivery.
    for dtype in (torch.int32, torch.complex64):
        with pytest.raises(ValueError, match="sparse storage dtype"):
            codec.encode([row("not_a_floating_sparse_slot", (2,), torch.ones(1, dtype=dtype), [0])], max_bytes=1024)


@pytest.mark.parametrize("corrupt", ["positions", "values"])
def test_wire_checksum_rejects_corruption_before_the_native_loader(corrupt):
    data = torch.tensor([7.0], dtype=torch.bfloat16)
    model = NativeCopyModel({"weight": torch.zeros(2, dtype=data.dtype)})
    ((metadata, raw),) = packed([row("weight", (2,), data, [1])])
    altered = raw.clone()
    group = metadata["flushes"][0]
    offset = group["offset" if corrupt == "positions" else "value_offset"]
    altered[offset] ^= 1
    with pytest.raises(RuntimeError, match="delta checksum mismatch"):
        install(model, [(metadata, altered)])
    assert not model.calls
    assert not model.weights["weight"].any()
    install(model, [(metadata, raw)])
    assert torch.equal(model.weights["weight"], torch.tensor([0.0, 7.0], dtype=data.dtype))


def test_loader_failure_restores_normal_copy_before_next_full_publication():
    class FailingLoader(NativeCopyModel):
        def load_weights(self, weights):
            assert torch.Tensor.copy_ is not original
            raise RuntimeError("native loader failed")

    original = torch.Tensor.copy_
    values = torch.tensor([2.0])
    frames = packed([row("weight", (2,), values, [0])])
    with pytest.raises(RuntimeError, match="native loader failed"):
        install(FailingLoader({"weight": torch.zeros(2)}), frames)
    assert torch.Tensor.copy_ is original

    model = NativeCopyModel({"weight": torch.zeros(2)})
    full = torch.tensor([0x7FC00003, 0x40000000], dtype=torch.int32)
    install(model, packed([row("weight", (2,), full.view(torch.float32))]))
    assert torch.equal(model.weights["weight"].view(torch.int32), full)


def test_wire_budget_includes_alignment_between_packed_groups():
    values = torch.tensor([1.0])
    group = [row("weight", (2,), values, [0])]
    with pytest.raises(ValueError, match="budget"):
        codec.encode(group, max_bytes=11)

    single = codec.encode(group, max_bytes=12)
    assert single[1].nbytes == 12
    with pytest.raises(ValueError, match="bounds"):
        codec.validate_directory(single[0], max_bytes=11)

    # Two 12-byte flushes need 28 bytes with alignment. A cap of 26
    # admits each flush but forces the bucket to release the first.
    frames = packed(group, group, budget=26)
    assert [raw.nbytes for _, raw in frames] == [12, 12]
    ((metadata, raw),) = packed(group, group, budget=28)
    assert raw.nbytes == 28
    assert [group["offset"] for group in metadata["flushes"]] == [0, 16]
    codec.validate_directory(metadata, max_bytes=28)


@pytest.mark.parametrize("has_pending_frame", [False, True])
def test_bucket_rejects_an_oversize_packet_without_changing_existing_state(has_pending_frame):
    values = torch.arange(8, dtype=torch.float32)
    oversized = codec.encode([row("large", (8,), values)], max_bytes=32)
    legal = codec.encode([row("small", (4,), values[:4])], max_bytes=16)
    bucket = codec.WireBucket(16)
    if has_pending_frame:
        assert bucket.offer(*legal) is None
    with pytest.raises(ValueError, match="budget|limit"):
        bucket.offer(*oversized)
    pending = bucket.finish()
    if has_pending_frame:
        assert pending is not None
        assert pending[0] == legal[0] and torch.equal(pending[1], legal[1])
    else:
        assert pending is None


@pytest.mark.parametrize("sparse", [False, True])
def test_receiver_decoded_chunk_budget_preserves_complete_tensor_order(sparse, monkeypatch):
    monkeypatch.setattr(verl_nan_mask, "CHUNK_BYTES", 40)
    shapes = {"first": (4,), "second": (2, 3), "third": (8,)}
    templates = {name: torch.zeros(shape) for name, shape in shapes.items()}
    model = NativeCopyModel(templates)
    entries = [
        row(
            name,
            shape,
            torch.tensor([7.0]) if sparse else torch.full(shape, 7.0),
            [0] if sparse else None,
        )
        for name, shape in shapes.items()
    ]
    install(model, packed(entries))
    assert model.calls == [["first", "second"], ["third"]]
    for name, parameter in model.weights.items():
        expected = templates[name].reshape(-1).clone()
        expected[: 1 if sparse else len(expected)] = 7
        assert torch.equal(parameter.reshape(-1), expected)


def command(model, control, raw=None, status=None):
    weights = [(json.dumps(control), raw if raw is not None else torch.empty(0, dtype=torch.uint8))]
    if status is not None:
        weights.append(("__delta_capability__", status))
    delta_loader.load_delta_weights(model, weights)


def publish(model, frames, version, *, base=None):
    control = dict(
        version=version, base=base, seed=base is None, staging_budget_bytes=1024, receiver_gpu_budget_bytes=8192
    )
    command(model, dict(op="begin", **control))
    for frame, (metadata, raw) in enumerate(frames):
        status = torch.full((len(metadata["tensors"]),), 255, dtype=torch.uint8)
        command(model, dict(op="consume", version=version, frame=frame, metadata=metadata), raw, status)
        assert status.tolist() == [0] * len(metadata["tensors"])
    status = torch.full((1,), 255, dtype=torch.uint8)
    command(model, dict(op="finalize", version=version, frames=len(frames)), status=status)
    assert status.item() == 0
    assert model._relax_delta_receiver.version == version


@pytest.mark.parametrize(
    ("tp_rank", "own_rows"),
    [(0, [0, 1, 4, 5, 8, 9, 10]), (1, [2, 3, 6, 7, 11, 12, 13])],
)
def test_mamba_native_assignment_preserves_unchanged_bf16_tp_shards(tp_rank, own_rows):
    mamba = pytest.importorskip("sglang.srt.layers.attention.mamba.mamba", exc_type=ImportError)

    class ConvModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Parameter(torch.zeros(7, 1, 3, dtype=torch.bfloat16), requires_grad=False)
            # The Qwen3.5 GDN convolution uses the same query, key, value
            # loader registration, including unequal query/key and value widths.
            self.conv.weight_loader = mamba.mamba_v2_sharded_weight_loader(
                [(4, 0, False), (4, 0, False), (6, 0, False)], 2, tp_rank
            )

        def load_weights(self, items):
            params = dict(self.named_parameters())
            for name, loaded in items:
                param = params[name]
                param.weight_loader(param, loaded)

    before = torch.arange(20, 62).to(torch.bfloat16).reshape(14, 1, 3).view(torch.int16)
    before[0, 0, 1], before[2, 0, 1] = 0x7FC1, 0x7FC2
    before[1, 0, 2], before[3, 0, 2] = -32768, -32768
    model = ConvModel()
    publish(model, packed([row("conv", before.shape, before.view(torch.bfloat16))]), "1")
    assert torch.equal(model.conv.view(torch.int16), before[own_rows])
    initial_ptr = model.conv.data_ptr()

    # Change one element in every rank's query, key and value slices.
    positions = [0, 6, 12, 18, 24, 33]
    changed = torch.tensor([-32768, 0, 0x4000, 0x4040, 0x4080, 0x40A0], dtype=torch.int16)
    publish(model, packed([row("conv", before.shape, changed.view(torch.bfloat16), positions)]), "2", base="1")
    expected = before.clone()
    expected.reshape(-1)[positions] = changed
    assert model.conv.data_ptr() == initial_ptr
    assert torch.equal(model.conv.view(torch.int16), expected[own_rows])

    # A later complete seed writes actual NaN words through the very same native loader.
    expected[4, 0, 0], expected[6, 0, 0] = 0x7FC3, 0x7FC4
    publish(model, packed([row("conv", before.shape, expected.view(torch.bfloat16))]), "3")
    assert torch.equal(model.conv.view(torch.int16), expected[own_rows])


class AssignmentModel(torch.nn.Module):
    def __init__(self, operation, *, fail_sparse=False):
        super().__init__()
        self.operation = operation
        self.fail_sparse = fail_sparse
        self.table = torch.nn.Parameter(torch.full((4, 3), -7.0), requires_grad=False)

    def load_weights(self, weights):
        for name, value in weights:
            table = self.get_parameter(name).data
            if self.operation == "copy":
                table[1:3, :].copy_(value)
            elif self.operation == "slice":
                table[1:3, ...] = value
            else:
                table[[3, 1], :] = value
            if self.fail_sparse:
                raise RuntimeError("native assignment loader failed")


@pytest.mark.parametrize("stage_copy", [False, True])
def test_staged_native_loader_propagates_placeholders_to_resident_buffer(stage_copy):
    class StagedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("table", torch.zeros(3))

        def load_weights(self, weights):
            for _, source in weights:
                staged = torch.zeros_like(source)
                if stage_copy:
                    staged.copy_(source)
                else:
                    staged[:] = source
                self.table[:].copy_(staged)

    model = StagedModel()
    seed = torch.tensor([4.0, 6.0, 8.0])
    install(model, packed([row("table", seed.shape, seed)]))
    install(model, packed([row("table", seed.shape, torch.tensor([9.0]), [1])]))
    assert torch.equal(model.table, torch.tensor([4.0, 9.0, 8.0]))


@pytest.mark.parametrize(("operation", "destination"), [("slice", [1, 2]), ("copy", [1, 2]), ("advanced", [3, 1])])
def test_native_weight_assignment_masks_broadcast_values(operation, destination):
    model = AssignmentModel(operation)
    dense = torch.tensor([2.0, -0.0]).reshape(2, 1)
    publish(model, packed([row("table", dense.shape, dense)]), "1")
    baseline = model.table.detach().view(torch.int32).clone()
    publish(model, packed([row("table", dense.shape, torch.tensor([7.0]), [0])]), "2", base="1")

    expected = baseline.clone()
    expected[destination[0], :].fill_(0x40E00000)
    assert torch.equal(model.table.view(torch.int32), expected)

    real_nan = torch.tensor([0x7FC00013, -(2**31)], dtype=torch.int32).view(torch.float32).reshape(2, 1)
    publish(model, packed([row("table", real_nan.shape, real_nan)]), "3")
    expected[destination[0], :].fill_(0x7FC00013)
    assert torch.equal(model.table.view(torch.int32), expected)


def test_failed_native_assignment_keeps_prior_ack_and_restores_normal_writes():
    model = AssignmentModel("slice")
    before = torch.tensor([2.0, -0.0]).reshape(2, 1)
    publish(model, packed([row("table", before.shape, before)]), "1")
    model.fail_sparse = True
    ((metadata, raw),) = packed([row("table", before.shape, torch.tensor([4.0]), [0])])
    command(
        model,
        dict(op="begin", version="2", base="1", seed=False, staging_budget_bytes=1024, receiver_gpu_budget_bytes=8192),
    )
    status = torch.full((1,), 255, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="native assignment loader failed"):
        command(model, dict(op="consume", version="2", frame=0, metadata=metadata), raw, status)
    assert status.item() == 255 and model._relax_delta_receiver.version == "1"
    assert model._relax_delta_receiver.failed
    assert model.table.view(torch.int32)[2].tolist() == [-(2**31)] * 3

    real_nan_word = 0x7FC00017
    real_nan = torch.tensor([real_nan_word], dtype=torch.int32).view(torch.float32)
    model.table.data[1:2, :] = real_nan
    assert model.table.view(torch.int32)[1].tolist() == [real_nan_word] * 3
    fresh = AssignmentModel("slice")
    full = torch.cat((real_nan, before.reshape(-1)[1:])).reshape(2, 1)
    publish(fresh, packed([row("table", full.shape, full)]), "1")
    assert fresh.table.view(torch.int32)[1].tolist() == [real_nan_word] * 3


@pytest.mark.parametrize("workspace_slack", [-1, 0])
def test_native_receiver_acknowledges_only_admitted_decoded_workspace(workspace_slack):
    original = torch.arange(16, dtype=torch.float32)
    model = NativeCopyModel({"weight": original})
    ((full, full_raw),) = packed([row("weight", (16,), original)])
    seed = dict(version="1", seed=True, staging_budget_bytes=64, receiver_gpu_budget_bytes=1024)
    command(model, dict(op="begin", **seed))
    seed_status = torch.full((1,), 7, dtype=torch.uint8)
    command(model, dict(op="consume", version="1", frame=0, metadata=full), full_raw, seed_status)
    assert seed_status.item() == 0 and model._relax_delta_receiver.version is None
    command(model, dict(op="finalize", version="1", frames=1), status=seed_status)
    assert model._relax_delta_receiver.version == "1"

    ((metadata, raw),) = packed([row("weight", (16,), torch.tensor([21.0]), [3])])
    staging_bytes = 16
    assert raw.nbytes <= staging_bytes
    budget = staging_bytes + delta_loader.workspace_bytes(metadata) + workspace_slack
    control = dict(
        version="2", base="1", seed=False, staging_budget_bytes=staging_bytes, receiver_gpu_budget_bytes=budget
    )
    command(model, dict(op="begin", **control))
    status = torch.full((1,), 91, dtype=torch.uint8)
    consume = dict(op="consume", version="2", frame=0, metadata=metadata)
    if workspace_slack < 0:
        with pytest.raises(ValueError, match="workspace"):
            command(model, consume, raw, status)
        assert model.calls == [["weight"]]
        assert torch.equal(model.weights["weight"], original)
        assert status.item() == 91 and model._relax_delta_receiver.version == "1"
        assert model._relax_delta_receiver.failed
    else:
        command(model, consume, raw, status)
        assert status.item() == 0 and model._relax_delta_receiver.version == "1"
        command(model, dict(op="finalize", version="2", frames=1), status=status)
        assert model._relax_delta_receiver.version == "2"
        expected = original.clone()
        expected[3] = 21.0
        assert model.calls == [["weight"], ["weight"]]
        assert torch.equal(model.weights["weight"], expected)
