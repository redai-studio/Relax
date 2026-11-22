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
    return ShardDeltaExporter(LayoutIterator(), LayoutIterator(), snapshot_budget_bytes=budget)


@pytest.mark.parametrize(
    "dtype,words,nan1,nan2,minus_zero",
    [
        (torch.bfloat16, torch.int16, 0x7FC1, 0x7FC2, -32768),
        (torch.float32, torch.int32, 0x7FC00001, 0x7FC00002, -2147483648),
    ],
)
def test_shard_delta_retains_raw_bits_and_advances_only_after_ack(dtype, words, nan1, nan2, minus_zero):
    model = {
        "weight": torch.tensor([nan1, 0, 3, 4, 5, 6, 7, 8], dtype=words).view(dtype).reshape(4, 2),
        "derived": torch.zeros(2, dtype=dtype),
    }
    source = exporter()
    source.begin(model)
    seed = [entry for chunk in source.export() for entry in chunk]
    with pytest.raises(RuntimeError, match="already pending"):
        source.begin(model)
    source.commit()
    assert all(entry.full for entry in seed)

    model["weight"].view(words)[0] = torch.tensor([nan2, minus_zero], dtype=words)
    model["derived"][1] = 1
    source.begin(model)
    delta = [entry for chunk in source.export() for entry in chunk]
    assert delta[0].positions.tolist() == [0, 4]
    assert delta[0].values.view(words).tolist() == [nan2, minus_zero]
    assert delta[1].full  # Two values are cheaper than a sparse index/value pair.
    torch.testing.assert_close(delta[1].values, model["derived"].exp(), rtol=0, atol=0)
    source.abort()

    # A failed receiver cannot silently become the next source baseline.
    source.begin(model)
    retry_seed = [entry for chunk in source.export() for entry in chunk]
    assert all(entry.full for entry in retry_seed)
    source.commit()
    source.begin(model)
    empty = [entry for chunk in source.export() for entry in chunk]
    assert all(entry.positions.numel() == entry.values.numel() == 0 for entry in empty)
    source.commit()


def test_shard_delta_capture_isolated_and_incomplete_export_cannot_commit():
    source = exporter()
    live = {"weight": torch.ones(2, 2), "derived": torch.zeros(2)}
    source.begin(live)
    live["weight"].zero_()
    stream = source.export()
    first = next(stream)[0]
    assert torch.equal(first.values, torch.ones(4))
    with pytest.raises(RuntimeError, match="incomplete"):
        source.commit()
    list(stream)
    source.commit()
    source.invalidate()
    source.begin(live)
    assert all(entry.full for chunk in source.export() for entry in chunk)


def test_shard_delta_admits_both_pending_and_committed_host_allocations():
    source = exporter(budget=24)
    live = {"weight": torch.ones(2, 2), "derived": torch.zeros(2)}
    source.begin(live)
    list(source.export())
    source.commit()
    with pytest.raises(ValueError, match="host budget"):
        source.begin(live)


def test_retired_export_cannot_acknowledge_a_new_pending_snapshot():
    source = exporter()
    live = {"weight": torch.ones(2, 2), "derived": torch.zeros(2)}
    source.begin(live)
    old_stream = source.export()
    next(old_stream)
    source.abort()
    live["weight"].fill_(3)
    source.begin(live)
    assert list(old_stream) == []
    with pytest.raises(RuntimeError, match="incomplete"):
        source.commit()
    list(source.export())
    source.commit()


def _compare_owner_export(rank, world, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist

    from relax.backends.megatron.weight_update.expert_gather import Bf16ExpertGather, OwnerExpert
    from relax.distributed.weight_sync.codec import encode

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world, timeout=timedelta(seconds=30))
    group = dist.group.WORLD
    device = torch.device("cpu")

    class ExpertIterator:
        def __init__(self):
            self.gather = Bf16ExpertGather()

        def items(self, shards):
            return [(f"{name}.{rank}", tensor) for name, tensor in shards.items()]

        def get_hf_weight_chunks(self, shards):
            yield self.gather(self.items(shards), group=group, device=device)

        def _owner_delta_group(self, metadata_group):
            return group

        def _owner_delta_chunks(self, shards, metadata_group):
            items = self.items(shards)
            # Mixed entries, including a replicated raw source, exercise both
            # the original gather and owner locations in a single HF frame.
            sources = [None, "replica", *(f"{name}.{rank}" for name in ("sparse", "dense", "forced", "empty"))]
            owned = [name for name in sources if name is not None]
            result = self.gather.owner_slots(
                items, sources, owned, group=group, metadata_group=metadata_group, device=device
            )
            assert [isinstance(tensor, OwnerExpert) for _, tensor in result] == [
                False,
                False,
                True,
                True,
                True,
                True,
            ] * world
            yield result

    def wire(chunk):
        return encode(
            ((entry.name, entry.shape, entry.dtype, entry.positions, entry.values) for entry in chunk),
            max_bytes=1 << 20,
        )

    reference = ShardDeltaExporter(ExpertIterator(), ExpertIterator(), snapshot_budget_bytes=1 << 20)
    candidate = ShardDeltaExporter(ExpertIterator(), ExpertIterator(), snapshot_budget_bytes=1 << 20)
    candidate._owner_local_diff = True
    initial = {
        name: torch.zeros(12, dtype=torch.bfloat16)
        for name in ("ordinary", "replicated", "sparse", "dense", "forced", "empty")
    }
    initial["sparse"].view(torch.int16)[1] = 0x7FC1
    updated = {name: tensor.clone() for name, tensor in initial.items()}
    if rank:
        updated["sparse"].view(torch.int16)[:2] = torch.tensor([-32768, 0x7FC2], dtype=torch.int16)
    updated["ordinary"].view(torch.int16)[3] = -32768
    updated["dense"][:4] = 1  # 4 * (4 + 2) == 12 * 2 triggers the original FULL threshold.
    try:
        for publication, shards in enumerate((initial, updated, updated, initial)):
            outputs = []
            for source in (reference, candidate) if publication % 2 == 0 else (candidate, reference):
                source.begin(shards)
                names = frozenset(f"forced.{owner}" for owner in range(world)) if publication != 2 else frozenset()
                chunks = list(source.export(full_names=names, emit=rank == 0, _owner_group=group))
                assert len(chunks) == 1
                if rank:
                    assert chunks[0] == []
                else:
                    outputs.append(wire(chunks[0]))
                source.commit()
            if rank == 0:
                (ref_meta, ref_data), (got_meta, got_data) = outputs
                assert got_meta == ref_meta and torch.equal(got_data, ref_data)
                if publication == 2:
                    assert got_data.numel() == 0
        # One owner can fail before its first payload collective. All peers must
        # leave the same pending attempt without acknowledging its contents.
        failing = Bf16ExpertGather()
        with pytest.raises(RuntimeError, match="conversion failed"):
            failing.owner_slots(
                [],
                [],
                [],
                group=group,
                metadata_group=group,
                device=device,
                error=ValueError("failed owner conversion") if rank == 1 else None,
            )
        candidate.begin(updated)
        previous = candidate._acknowledged
        # This name mismatch is visible on just one rank. The original owner
        # diff reports it to every peer before a payload exchange.
        new = OwnerExpert(0, (12,), updated["sparse"] if rank == 0 else None, device)
        with pytest.raises(RuntimeError, match="comparison failed"):
            candidate._encode_owner_chunk(
                [("x", new)], [("bad" if rank == 1 else "x", new)], frozenset(), group, group
            )
        candidate.abort()
        assert candidate._acknowledged is previous
        candidate.begin(updated)
        retry = list(candidate.export(emit=rank == 0, _owner_group=group))
        if rank == 0:
            assert all(entry.full for entry in retry[0])
        candidate.commit()
        # One rank declines the owner layout. Both must take the ordinary
        # converter rather than entering incompatible collective sequences.
        candidate.current_iterator._owner_delta_group = lambda _: group if rank == 0 else None
        candidate.begin(updated)
        unchanged = list(candidate.export(emit=rank == 0, _owner_group=group))
        if rank == 0:
            assert all(not item.full and item.values.numel() == 0 for item in unchanged[0])
        candidate.commit()
    finally:
        dist.destroy_process_group()


def test_owner_export_matches_original_wire_across_real_gloo_group(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _compare_owner_export, args=(2, (tmp_path / "rendezvous").as_uri()), nprocs=2, start_method="fork"
    )


def _check_owner_target_lanes(rank, world, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist

    from relax.backends.megatron.weight_update.expert_gather import OwnerExpert
    from relax.backends.megatron.weight_update.target_shard import encode_frame
    from relax.distributed.weight_sync.codec import tensors

    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=rendezvous, timeout=timedelta(seconds=20))
    group, device = dist.group.WORLD, torch.device("cpu")
    try:
        current, baseline, complete = [], [], []
        # Real owner gather of two projection axes, zero lane/whole entry,
        # forced ordinary FULL, raw NaN payload and negative zero.
        suffixes = ["down_proj", "gate_proj", "up_proj", "down_proj", "gate_proj", "up_proj", "gate_proj", "down_proj"]
        for index, projection in enumerate(suffixes):
            name = f"model.layers.0.mlp.experts.{index}.{projection}.weight"
            old = torch.zeros((8, 12), dtype=torch.bfloat16)
            new = old.clone()
            if index != 3:
                new.view(torch.int16).flatten()[[0, 5, 12, 46, 95]] = torch.tensor(
                    [-32768, 0x7FC1, 0x7FC2, -1, 0x3F80], dtype=torch.int16
                )
            owner = index % world
            baseline.append((name, OwnerExpert(owner, old.shape, old if owner == rank else None, device)))
            current.append((name, OwnerExpert(owner, new.shape, new if owner == rank else None, device)))
            complete.append((old, new))
        ordinary = torch.tensor([3, 4], dtype=torch.int64)
        current.append(("ordinary", ordinary))
        baseline.append(("ordinary", ordinary.clone()))
        forced = {current[4][0]}
        changes = ShardDeltaExporter._encode_owner_chunk(current, baseline, forced, group, group, target_lanes=True)
        if rank:
            assert changes == []
            return
        directories, raw = encode_frame(changes, 4096)
        moved = 0
        for tp, (directory, slab) in enumerate(zip(directories, raw, strict=True)):
            marker = directory.pop("target_shard")
            assert marker == dict(format=1, tp_rank=tp, tp_size=4, entries=[0, 1, 2, 3, 5, 6, 7])
            decoded = list(tensors(directory, slab))
            assert [entry["name"] for entry, _, _ in decoded] == [key for key, _ in current]
            for index, (entry, positions, values) in enumerate(decoded):
                if index == len(suffixes):
                    assert torch.equal(values, ordinary)
                    continue
                old, new = complete[index]
                if index == 4:
                    assert entry["full"] and torch.equal(values.view(torch.int16), new.view(torch.int16).flatten())
                    continue
                axis = 1 if suffixes[index] == "down_proj" else 0
                actual = old.chunk(4, dim=axis)[tp].contiguous().view(torch.int16).clone()
                actual.flatten().index_copy_(0, positions.long(), values.view(torch.int16))
                assert torch.equal(actual, new.chunk(4, dim=axis)[tp].view(torch.int16))
                assert positions.tolist() == sorted(set(positions.tolist()))
                moved += positions.numel()
            assert not decoded[3][1].numel()
        assert moved == 30
    finally:
        dist.destroy_process_group()


def test_owner_target_lanes_reconstruct_every_original_word(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _check_owner_target_lanes, args=(8, (tmp_path / "lane-owners").as_uri()), nprocs=8, start_method="fork"
    )


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
    seed = [item for chunk in s.export() for item in chunk]
    assert torch.equal(seed[0].values, live["weight"].T.contiguous().flatten())
    s.commit()
    prior = s._ack_bank
    saved = s._acknowledged["weight"].clone()
    live["weight"][0, 0] = 30
    s.begin(live, live_locked=True)
    attempt = s._backup.storage
    stream = s.export()
    assert next(stream)[0].positions.tolist() == [0]
    s.abort()
    assert not s.sender_pipeline_active and s._ack_bank is prior
    assert torch.equal(s._acknowledged["weight"], saved)
    s.begin(live, live_locked=True)
    assert s._backup.storage is attempt
    assert all(item.full for chunk in s.export() for item in chunk)
    s.commit()
    assert s._ack_bank is attempt and s._spare_bank is prior
    assert torch.equal(s._acknowledged["weight"], live["weight"])

    s.begin(live, live_locked=True)

    def copy_failure():
        raise RuntimeError("delayed copy error")

    monkeypatch.setattr(s._backup, "finish", copy_failure)
    with pytest.raises(RuntimeError, match="delayed copy error"):
        list(s.export())
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
        "ordered",
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
    defer = case not in {"transfer_error", "mixed", "connection_reused", "ordered"}
    if case == "empty":
        monkeypatch.setattr(source.current_iterator, "get_hf_weight_chunks", lambda _: iter(()))
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
    source.transfer.ordered_target_consume = case == "ordered"
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


def _export_empty_pipeline_stage(rank, rendezvous):
    from datetime import timedelta
    from types import SimpleNamespace

    import torch.distributed as dist

    from relax.distributed.weight_sync.config import DeltaSyncConfig

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    group = dist.group.WORLD

    class PipelineIterator:
        def get_hf_weight_chunks(self, shards):
            values = shards["layer"].clone() if rank == 1 else torch.empty(8)
            dist.broadcast(values, src=1, group=group)
            yield [("model.layers.3.weight", values)]

    try:
        source = ShardDeltaExporter(
            PipelineIterator(), PipelineIterator(), snapshot_budget_bytes=256, config=DeltaSyncConfig(stream="pp")
        )
        frames = []
        source.transfer = SimpleNamespace(
            connect=lambda *_args, **_kwargs: False,
            begin=lambda **kwargs: None,
            send=lambda directory, _raw, **_kwargs: frames.append(directory) or set(),
            ordered_target_consume=False,
            finalize=lambda: {"receivers": [{"force_full_seed": False}], "final_full_names": []},
        )
        model = {"layer": torch.zeros(8)} if rank == 1 else {}
        args = dict(engines=[], gpu_counts=[], group=group)
        for version in range(2):
            if version and rank == 1:
                model["layer"][2] = 2.5
            status = source.synchronize(lambda: model, version=str(version), **args)
            assert status["full"] is (version == 0)
            source.commit()
        if rank == 0:
            full, delta = (frame["tensors"][0] for frame in frames)
            assert full["full"] and not delta["full"] and delta["count"] == 1
        if rank == 1:
            model["extra"] = torch.zeros(1)
        with pytest.raises(RuntimeError, match="Source shard schema changed"):
            source.synchronize(lambda: model, version="2", **args)
        assert source._version == "1" and source._pending is None
    finally:
        dist.destroy_process_group()


def test_empty_pipeline_stage_participates_in_both_delta_conversions(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _export_empty_pipeline_stage, args=((tmp_path / "pp-ranks").as_uri(),), nprocs=2, start_method="fork"
    )


def _check_frame_control(rank, rendezvous):
    from datetime import timedelta
    from types import SimpleNamespace

    import torch.distributed as dist

    from relax.distributed.weight_sync.config import DeltaSyncConfig

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    group = dist.group.WORLD
    model = {"w": torch.zeros(8)}

    class Groups:
        def get_hf_weight_chunks(self, shards):
            yield [("a", shards["w"])]
            if rank == 1 and fault == "iterator":
                raise LookupError("rank one failed during conversion")
            if rank == 1 and fault == "short":
                return
            yield [("b", shards["w"])]

    def can_stage(metadata, raw):
        if fault == "admission":
            raise ValueError("publisher admission failed")
        return not events

    try:
        for transport, requested_fault in (
            ("tcp", None),
            ("nccl_socket", None),
            ("tcp", "admission"),
            ("nccl_socket", "short"),
            ("tcp", "iterator"),
        ):
            events = []
            config = DeltaSyncConfig(stream="pp", transport=transport, staging_budget_bytes=32, receive_overlap=True)
            source = ShardDeltaExporter(Groups(), Groups(), snapshot_budget_bytes=256, config=config)
            source.transfer = SimpleNamespace(
                connect=lambda *args, **kwargs: False,
                can_reuse=lambda *_: True,
                begin=lambda **kwargs: None,
                stage=lambda meta, raw: events.append(("stage", [t["name"] for t in meta["tensors"]])),
                send=lambda meta, raw, **kwargs: (
                    events.append(("send", [t["name"] for t in meta["tensors"]])) or set()
                ),
                activate=lambda: events.append("activate"),
                apply=lambda: events.append("apply") or set(),
                can_stage=can_stage,
                ordered_target_consume=False,
                finalize=lambda: events.append("finalize") or {"receivers": [], "final_full_names": []},
            )
            args = dict(engines=[], gpu_counts=[], group=group)
            fault = None
            source.synchronize(lambda: model, version="0", **args)
            source.commit()
            events.clear()
            fault = requested_fault
            if fault:
                error, text = {
                    "short": (ValueError, "disagree on the complete HF group directory"),
                    "iterator": (RuntimeError, "LookupError: rank one failed during conversion"),
                    "admission": (RuntimeError, "ValueError: publisher admission failed"),
                }[fault]
                with pytest.raises(error, match=text):
                    source.synchronize(
                        lambda: model, version="1", before_first_apply=lambda: events.append("pause"), **args
                    )
                assert "pause" not in events and "finalize" not in events
                assert source.failed and source._version == "0" and source._pending is None
                continue
            source.synchronize(lambda: model, version="1", before_first_apply=lambda: events.append("pause"), **args)
            if rank == 1:
                assert events == ["pause"]
            elif transport == "tcp":
                assert events == [("stage", ["a"]), "pause", "activate", "apply", ("send", ["b"]), "finalize"]
            else:
                assert events == [("stage", ["a", "b"]), "pause", "activate", "apply", "finalize"]
            assert source._version == "0"
            source.commit()
            assert source._version == "1"
    finally:
        dist.destroy_process_group()


def test_frame_control_propagates_publisher_and_peer_decisions_before_final_ack(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(_check_frame_control, args=((tmp_path / "control").as_uri(),), nprocs=2, start_method="fork")


def test_requested_delta_admits_single_node_tp_pp_mesh(tmp_path):
    from types import SimpleNamespace

    from relax.backends.megatron.weight_update.delta_export import delta_sync_config
    from relax.distributed.weight_sync.config import supports_delta_layout

    settings = tmp_path / "delta.json"
    settings.write_text('{"stream":"test"}')
    args = SimpleNamespace(memory_weight_sync_config=settings, sglang_pp_size=2, rollout_num_gpus_per_engine=4)
    assert delta_sync_config(args, None).stream == "test"
    layout = dict(nnodes=1, pp_size=2, tp_size=2, dp_size=1)
    assert supports_delta_layout(layout, 4)
    assert not supports_delta_layout(layout, 2)
    for changed in [dict(nnodes=2), dict(dp_size=2), dict(pp_size=0), dict(tp_size=False), dict(tp_size=None)]:
        assert not supports_delta_layout(dict(layout, **changed), 4)
