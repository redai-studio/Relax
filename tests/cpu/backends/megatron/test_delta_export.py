# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Snapshot/ACK regression without Megatron hardware or optional
dependencies."""

import pytest
import torch

from relax.backends.megatron.weight_update.delta_export import ShardDeltaExporter


class LayoutIterator:
    """Conversion boundary: two chunks with a real transpose and numeric
    map."""

    def get_hf_weight_chunks(self, shards):
        yield [("hf.weight", shards["weight"].transpose(0, 1))]
        yield [("hf.derived", shards["derived"].exp())]


def exporter(budget=1 << 20):
    source = ShardDeltaExporter(LayoutIterator(), snapshot_budget_bytes=budget)
    # These isolated tests exercise snapshot lifetimes. The conversion index
    # and sparse gather run against actual Gloo in test_bridge_sharded_publication.
    source.sharded.prepare = lambda _: None
    return source


def test_shard_delta_capture_isolated_and_incomplete_export_cannot_commit():
    source = exporter()
    live = {"weight": torch.ones(2, 2), "derived": torch.zeros(2)}
    source.begin(live)
    live["weight"].zero_()
    stream = source.export(_owner_group=object())
    first = next(stream)[0]
    assert torch.equal(first.values, torch.ones(4))
    with pytest.raises(RuntimeError, match="incomplete"):
        source.commit()
    list(stream)
    source.commit()
    source.invalidate()
    source.begin(live)
    assert all(entry.full for chunk in source.export(_owner_group=object()) for entry in chunk)


def test_shard_delta_admits_both_pending_and_committed_host_allocations():
    source = exporter(budget=24)
    live = {"weight": torch.ones(2, 2), "derived": torch.zeros(2)}
    source.begin(live)
    list(source.export(_owner_group=object()))
    source.commit()
    with pytest.raises(ValueError, match="host budget"):
        source.begin(live)


def test_retired_export_cannot_acknowledge_a_new_pending_snapshot():
    source = exporter()
    live = {"weight": torch.ones(2, 2), "derived": torch.zeros(2)}
    source.begin(live)
    old_stream = source.export(_owner_group=object())
    next(old_stream)
    source.abort()
    live["weight"].fill_(3)
    source.begin(live)
    assert list(old_stream) == []
    with pytest.raises(RuntimeError, match="incomplete"):
        source.commit()
    list(source.export(_owner_group=object()))
    source.commit()


def test_aligned_shard_bank_reuses_bytes_without_changing_payload_bits():
    from relax.backends.megatron.weight_update.shard_snapshot import _view, shard_layout

    shards = {
        "byte": torch.tensor(231, dtype=torch.uint8),
        "nan": torch.tensor([0x7FC00001, -2147483648], dtype=torch.int32).view(torch.float32),
        "empty": torch.empty(0, dtype=torch.bfloat16),
        "integer": torch.tensor(2**60, dtype=torch.int64),
    }
    entries, size = shard_layout(shards.items())
    assert size == 24
    bank = torch.zeros(size, dtype=torch.uint8)
    for name, value, offset, length in entries:
        target = _view(bank, value, offset, length)
        assert target.shape == value.shape
        target.copy_(value)
        assert torch.equal(bank[offset : offset + length], shards[name].reshape(-1).view(torch.uint8))


def test_packed_bank_copies_adjacent_mixed_types_and_skips_nonowners(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    from relax.backends.megatron.weight_update import shard_snapshot
    from relax.backends.megatron.weight_update.shard_snapshot import (
        PinnedShardBank,
        _copy_bank_slices,
        _view,
        shard_bank_layout,
        shard_layout,
    )

    raw = {
        "bf16": torch.tensor([0x7FC1], dtype=torch.int16).view(torch.bfloat16),
        "fp32": torch.tensor([-2147483648], dtype=torch.int32).view(torch.float32),
        "unused": torch.ones(13, dtype=torch.uint8),
        "int": torch.tensor([2**60 + 3], dtype=torch.int64),
        "empty": torch.empty(0, dtype=torch.int64),
    }
    entries, charged = shard_bank_layout(raw.items(), 16)
    sizes = tuple(capacity for _, capacity in entries)
    assert sizes == (8, 16, 8) and charged == 32
    monkeypatch.setattr(shard_snapshot, "_allocate_bank", lambda size, **_: torch.full((size,), 17, dtype=torch.uint8))
    bank = PinnedShardBank(sizes)
    selected = []
    for (group, _), slab in zip(entries, bank.slabs, strict=True):
        for name, tensor, offset, length in group:
            host = _view(slab, tensor, offset, length)
            host.copy_(tensor)
            if name != "unused":
                selected.append((name, host))
    layout, size = shard_layout(selected)
    slot = torch.empty(size, dtype=torch.uint8)
    copies = []

    class CountCopies(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten.copy_.default:
                copies.append(args[1].numel())
            return func(*args, **(kwargs or {}))

    with CountCopies():
        _copy_bank_slices(bank, slot, layout)
    assert copies == [8, 8]
    for name, host, offset, length in layout:
        actual = _view(slot, host, offset, length)
        assert torch.equal(actual.view(torch.uint8), raw[name].view(torch.uint8))


def test_shard_prefetch_splits_matching_offsets_in_distinct_host_slabs(monkeypatch):
    from relax.backends.megatron.weight_update import shard_snapshot
    from relax.backends.megatron.weight_update.shard_snapshot import PinnedShardBank, _copy_bank_slices, shard_layout

    monkeypatch.setattr(shard_snapshot, "_allocate_bank", lambda size, **_: torch.arange(size, dtype=torch.uint8))
    bank = PinnedShardBank((8, 8))
    first, second = bank.slabs
    second.add_(64)
    # The source and destination offsets alone would allow a single DMA,
    # accidentally reading the second word from the first slab.
    layout, size = shard_layout([("a", first[:4]), ("b", second[4:])])
    slot = torch.empty(size, dtype=torch.uint8)
    _copy_bank_slices(bank, slot, layout)
    assert slot.tolist() == [0, 1, 2, 3, 68, 69, 70, 71]


def test_pipeline_inputs_isolate_inplace_converter_and_preserve_unlocked_snapshot():
    from relax.backends.megatron.weight_update.shard_snapshot import CudaShardInputs

    original = torch.tensor([1, 5, 13])
    mapping = CudaShardInputs({"w": original}, original.device)
    mapping["w"].zero_()
    assert original.tolist() == [1, 5, 13]
    s = exporter()
    s._supports_live_shards = True
    assert not s._can_read_live({"w": original})
    assert not s.sender_pipeline_active


def test_live_publication_reads_source_until_ack_and_preserves_last_bank_on_abort(monkeypatch):
    from types import SimpleNamespace

    from relax.backends.megatron.weight_update import delta_export, shard_snapshot
    from relax.backends.megatron.weight_update.shard_snapshot import PinnedShardBank, _view

    monkeypatch.setattr(shard_snapshot, "_allocate_bank", lambda size, **_: torch.zeros(size, dtype=torch.uint8))

    class DelayedCopy:
        def __init__(self, entries, size, spare, budget, _bucket):
            assert size <= budget
            self.storage = spare if spare is not None else PinnedShardBank(tuple(capacity for _, capacity in entries))
            self.targets, self.inputs = {}, {}
            for (group, _), slab in zip(entries, self.storage.slabs, strict=True):
                slab.zero_()
                for n, t, offset, length in group:
                    self.targets[n] = _view(slab, t, offset, length)
                    self.inputs[n] = t.clone()

        def finish(self):
            for n, copy in self.inputs.items():
                self.targets[n].copy_(copy)

    # Deterministic copy completion at the owner's final gate; physical
    # cross-stream behavior is exercised separately in the single-GPU test.
    monkeypatch.setattr(delta_export, "CudaShardBackup", DelayedCopy)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(record_event=lambda: SimpleNamespace(synchronize=lambda: None)),
    )
    s = exporter()
    s.config = SimpleNamespace(bucket_bytes=256, gpu_budget_bytes=4096)
    live = {"weight": torch.arange(12.0).view(3, 4), "derived": torch.ones(2)}
    s.begin(live, live_locked=True)
    assert s.sender_pipeline_active
    assert s._pending["weight"].count_nonzero() == 0
    seed = [item for chunk in s.export(_owner_group=object()) for item in chunk]
    assert torch.equal(seed[0].values, live["weight"].T.contiguous().flatten())
    s.commit()
    prior = s._ack_bank
    saved = s._acknowledged["weight"].clone()
    live["weight"][0, 0] = 30
    s.begin(live, live_locked=True)
    attempt = s._backup.storage
    stream = s.export(full=True, _owner_group=object())
    assert next(stream)[0].values[0] == 30
    s.abort()
    assert not s.sender_pipeline_active and s._ack_bank is prior
    assert torch.equal(s._acknowledged["weight"], saved)
    s.begin(live, live_locked=True)
    assert s._backup.storage is attempt
    assert all(item.full for chunk in s.export(_owner_group=object()) for item in chunk)
    s.commit()
    assert s._ack_bank is attempt and s._spare_bank is prior
    assert torch.equal(s._acknowledged["weight"], live["weight"])

    s.begin(live, live_locked=True)

    def copy_failure():
        raise RuntimeError("delayed copy error")

    monkeypatch.setattr(s._backup, "finish", copy_failure)
    with pytest.raises(RuntimeError, match="delayed copy error"):
        list(s.export(full=True, _owner_group=object()))
    with pytest.raises(RuntimeError, match="incomplete"):
        s.commit()
    with pytest.raises(RuntimeError, match="delayed copy error"):
        s.abort()
    assert s._ack_bank is attempt and not s.sender_pipeline_active and s._pending is None


@pytest.mark.parametrize(
    "case",
    [
        "reuse",
        "budget",
        "oom",
        "device_error",
        "transfer_error",
        "socket_error",
        "peer_error",
        "empty",
        "mixed",
        "connection_reused",
    ],
)
def test_first_live_seed_prepares_one_spare_under_its_existing_budget(case, monkeypatch):
    from contextlib import nullcontext
    from threading import Event, current_thread
    from types import SimpleNamespace

    from relax.backends.megatron.weight_update import delta_export, shard_snapshot
    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    model = {"byte": torch.ones(1, dtype=torch.uint8), "weight": torch.ones(1, 1), "derived": torch.ones(1)}
    # The 9 model bytes require a 12-byte aligned bank.
    source = exporter(23 if case == "budget" else 24)
    transport = "tcp" if case == "transfer_error" else "nccl_socket"
    source.config = DeltaSyncConfig(stream="test", sender_pipeline=True, transport=transport, staging_budget_bytes=8)
    defer = case not in {"transfer_error", "mixed", "connection_reused"}
    if case == "empty":
        monkeypatch.setattr(source.sharded.iterator, "get_hf_weight_chunks", lambda _: iter(()))
    monkeypatch.setattr(source, "_can_read_live", lambda _: True)
    monkeypatch.setattr(delta_export.dist, "get_rank", lambda group: 0)
    world = 2
    monkeypatch.setattr(delta_export.dist, "get_world_size", lambda group: world)
    monkeypatch.setattr(delta_export.dist, "get_process_group_ranks", lambda group: [0])
    monkeypatch.setattr(delta_export.dist, "get_backend", lambda group: "gloo")
    submitted, agreed = Event(), Event()

    def reduce(state, op, group):
        state.mul_(world)
        if submitted.is_set():
            if case == "peer_error":
                state[0] = 1
            else:
                agreed.set()

    def gather(out, obj, group):
        out[:] = [obj] * world
        if case == "peer_error" and submitted.is_set():
            out[1] = "peer rejected the frame"

    monkeypatch.setattr(delta_export.dist, "all_reduce", reduce)
    monkeypatch.setattr(delta_export.dist, "all_gather_object", gather)
    monkeypatch.setattr(delta_export.dist, "broadcast_object_list", lambda *args, **kwargs: None)

    stream = SimpleNamespace(
        wait_stream=lambda _: None, record_event=lambda: SimpleNamespace(synchronize=lambda: None)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    devices = []
    monkeypatch.setattr(torch.cuda, "device", lambda device: devices.append(device) or nullcontext())

    begun, copying, release = Event(), Event(), Event()
    worker = []
    allocations = []

    def allocate(size, *, spare=False):
        assert size in (8, 4)
        if spare:
            assert begun.is_set() and devices == [model["weight"].device]
            if defer and case != "empty":
                assert agreed.is_set()
            if not worker:
                worker.append(current_thread())
            copying.set()
            assert release.wait(5)
            if case == "oom" and size == 4:
                raise torch.OutOfMemoryError("test host allocation")
            if case == "device_error":
                raise RuntimeError("device unavailable")
        bank = torch.empty(size, dtype=torch.uint8)
        allocations.append(bank)
        return bank

    monkeypatch.setattr(shard_snapshot, "_allocate_bank", allocate)

    def submit(*args, **kwargs):
        assert case != "empty"
        if defer and not submitted.is_set():
            assert not copying.is_set()
        elif not defer:
            assert copying.wait(5)
        if case in {"transfer_error", "socket_error"}:
            raise RuntimeError("delivery interrupted")
        submitted.set()
        release.set()
        return set()

    def begin(**kwargs):
        source.transfer._applying = True
        begun.set()

    if case == "empty":
        release.set()
    source.transfer = DeltaTransfer(source.config, "cpu")
    source.transfer.group = object()
    source.transfer.tcp = object() if case == "mixed" else None
    source.transfer.connect = lambda *args, **kwargs: case != "connection_reused"
    source.transfer.begin = begin
    source.transfer.prepare_frame = lambda _: None
    source.transfer.drain_frame = set
    source.transfer.discard_prepared = release.set
    source.transfer.submit_frame = submit
    source.transfer.apply = set
    source.transfer.finalize = lambda: {"receivers": [{"force_full_seed": False}], "final_full_names": []}
    args = dict(engines=[], gpu_counts=[], version="0", group=object())
    errors = {
        "device_error": "device unavailable",
        "transfer_error": "delivery interrupted",
        "socket_error": "delivery interrupted",
        "peer_error": "peer rejected the frame",
    }
    if case in errors:
        message = errors[case]
        with pytest.raises(RuntimeError, match=message):
            source.synchronize(lambda: model, **args)
        assert len(worker) == (case not in {"socket_error", "peer_error"})
        assert all(not thread.is_alive() for thread in worker)
        assert source._ack_bank is None and source._pending is None and source._spare_bank.slabs[0] is allocations[0]
        return
    source.synchronize(lambda: model, **args)
    assert submitted.is_set() == (case != "empty")
    assert len(worker) == (case != "budget") and all(not thread.is_alive() for thread in worker)
    assert source._ack_bank is None and not source._acknowledged
    source.commit()
    assert source._ack_bank.slabs[0] is allocations[0]
    if case == "budget":
        assert source._spare_bank is None and len(allocations) == 2
        with pytest.raises(ValueError, match="host budget"):
            source.begin(model, live_locked=True)
        return
    if case == "oom":
        assert source._spare_bank is None and len(allocations) == 3
    else:
        assert source._spare_bank.slabs[0] is allocations[2]
    source.synchronize(lambda: model, **dict(args, version="1", force_full=True))
    second_bank = source._backup.storage
    assert second_bank.slabs[0] is allocations[-2] and len(allocations) == (5 if case == "oom" else 4)
    assert len(worker) == 1
    source.commit()
    assert source._ack_bank is second_bank and source._spare_bank.slabs[0] is allocations[0]
    if case == "reuse":
        # Equal total bytes with different slab boundaries must replace the spare.
        prior = source._spare_bank
        source.begin({"a": torch.ones(4, dtype=torch.uint8), "b": torch.ones(6, dtype=torch.uint8)}, live_locked=True)
        assert prior.sizes == (8, 4) and source._backup.storage.sizes == (4, 8)
        assert source._backup.storage is not prior
        source.abort()
        assert source._ack_bank is second_bank


def test_requested_delta_admits_single_node_tp_pp_mesh(tmp_path):
    from types import SimpleNamespace

    from relax.backends.megatron.weight_update.delta_export import delta_sync_config
    from relax.distributed.weight_sync.config import supports_delta_layout

    settings = tmp_path / "delta.json"
    settings.write_text('{"stream":"test"}')
    args = SimpleNamespace(memory_weight_sync_config=settings, sglang_pp_size=2, rollout_num_gpus_per_engine=4)
    assert delta_sync_config(args, None).stream == "test"
    args.megatron_to_hf_mode = "raw"
    assert delta_sync_config(args, None) is None
    args.megatron_to_hf_mode = "bridge"
    layout = dict(nnodes=1, pp_size=2, tp_size=2, dp_size=1)
    assert supports_delta_layout(layout, 4)
    assert not supports_delta_layout(layout, 2)
    for changed in [
        dict(nnodes=2),
        dict(dp_size=2),
        dict(pp_size=0),
        dict(tp_size=False),
        dict(tp_size=None),
        dict(speculative_algorithm="NEXTN"),
    ]:
        assert not supports_delta_layout(dict(layout, **changed), 4)
