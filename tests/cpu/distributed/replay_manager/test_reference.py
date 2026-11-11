# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from relax.distributed.replay_manager.config import ReplayConfig, validate_reference_training
from relax.distributed.replay_manager.consumer import ConsumerBatchLayout, ConsumerLayerReader, ConsumerTensorCache
from relax.distributed.replay_manager.core import ReplayInventory, ReplayManager
from relax.distributed.replay_manager.storage import TQReplayStorage, _clear_simple_storage_samples


class MemoryBody:
    """Replace remote body I/O; publication, ownership and reads stay real."""

    def __init__(self):
        self.blocks = {}
        self.reads = 0

    def write_many(self, keys, values):
        self.blocks.update((key, value.copy()) for key, value in zip(keys, values, strict=True))

    def read(self, keys):
        self.reads += len(keys)
        return [self.blocks[key].copy() for key in keys]

    def delete(self, keys):
        for key in keys:
            self.blocks.pop(key, None)

    def close(self):
        pass


@pytest.mark.parametrize("mode", ["colocate", "hybrid", "fully_async", "separate", "fsdp"])
def test_reference_training_rejects_unwired_consumers_before_startup(monkeypatch, mode):
    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "1")
    args = SimpleNamespace(
        train_backend="fsdp" if mode == "fsdp" else "megatron",
        colocate=mode == "colocate",
        hybrid=mode == "hybrid",
        fully_async=mode == "fully_async",
        use_rollout_routing_replay=True,
    )
    if mode == "colocate":
        validate_reference_training(args)
    else:
        with pytest.raises(ValueError, match="colocated Megatron"):
            validate_reference_training(args)
    args.use_rollout_routing_replay = False
    validate_reference_training(args)


@pytest.fixture
def published():
    config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=2, publication_bytes=64)
    body = MemoryBody()
    inventory = ReplayInventory(config, body.delete)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = body
    values = np.arange(4 * 3 * 2, dtype=np.int32).reshape(4, 3, 2)
    publication = store.begin_publication(kind="moe", layers=3, topk=2)
    for start in (0, 2):
        publication.append(2, lambda layer, start=start: values[start : start + 2, layer].copy())
    ref = publication.finish([10, 11, 12, 13, 14])
    yield config, store, body, inventory, ref, values
    store._writer.close()


@pytest.mark.parametrize("capacity", [0, 8, 4096])
def test_reference_repeated_reads_preserve_values_and_retired_reader_ownership(published, capacity):
    _, store, body, inventory, ref, values = published
    store.attach([ref], "train_0")
    store.commit([ref])
    cache = ConsumerTensorCache(capacity)
    # Include padding and a non-contiguous row order, as after rank slicing.
    layout = ConsumerBatchLayout.from_coordinates(np.array([[0, 3], [0, 0], [-1, 0], [0, 2]]))
    with store.reader([ref], partition="train_0") as session:
        sources = [
            ConsumerLayerReader(session, layout, layer=layer, kind="moe", device="cpu", num_experts=128, cache=cache)
            for layer in range(3)
        ]
    try:
        store.retire_partition("train_0")
        assert body.blocks  # Retiring the queue must not invalidate recompute.
        first_reads = None
        for phase in range(3):
            for layer, source in enumerate(sources):
                expected = np.stack((values[3, layer], values[0, layer], [layer * 2, layer * 2 + 1], values[2, layer]))
                actual = source.read()
                assert actual.dtype == torch.int64
                np.testing.assert_array_equal(actual.numpy(), expected)
                actual.fill_(99)  # Returned tensors must not mutate the cache.
            if phase == 0:
                first_reads = body.reads
            elif capacity == 4096:
                assert body.reads == first_reads
        if capacity in (0, 8):
            assert body.reads > first_reads
    finally:
        for source in sources:
            source.release()
    assert not inventory.state["objects"]
    assert not body.blocks


def test_reference_retire_before_put_ack_keeps_reader_until_final_release(published):
    _, store, body, inventory, ref, _ = published
    store.attach([ref], "train_0")
    leases = store.acquire_many([ref], partition="train_0")
    store.retire_partition("train_0")
    store.commit([ref])
    assert body.blocks
    store.release_many([ref], leases)
    assert not body.blocks and not inventory.state["objects"]


def test_reference_prefetches_next_microbatch_without_blocking_cached_layers(published, monkeypatch) -> None:
    _, store, body, inventory, first_ref, first_values = published
    second_values = np.arange(4 * 5 * 2, dtype=np.int32).reshape(4, 5, 2) + 216
    publication = store.begin_publication(kind="moe", layers=5, topk=2)
    for start in (0, 2):
        publication.append(2, lambda layer, start=start: second_values[start : start + 2, layer].copy())
    second_ref = publication.finish([20, 21, 22, 23, 24])
    refs = [first_ref, second_ref]
    store.attach(refs, "train_0")
    store.commit(refs)
    cache = ConsumerTensorCache(4096)
    groups = []
    for ref, rows, layers in ((first_ref, [3, 0], 3), (second_ref, [1, 2], 5)):
        layout = ConsumerBatchLayout.from_coordinates(np.array([[0, row] for row in rows]))
        with store.reader([ref], partition="train_0") as session:
            groups.append(
                [
                    ConsumerLayerReader(
                        session, layout, layer=layer, kind="moe", device="cpu", num_experts=256, cache=cache
                    )
                    for layer in range(layers)
                ]
            )
    started, resume = Event(), Event()
    read_layers = store._read_layers_into

    def gated_read(layers: list) -> None:
        if layers[0][0][0][0].record == second_ref.record:
            started.set()
            assert resume.wait(5), "Next-microbatch read was never released"
        return read_layers(layers)

    monkeypatch.setattr(store, "_read_layers_into", gated_read)
    try:
        groups[0][0].read()
        groups[0][1].read()
        assert started.wait(2), "Next-microbatch prefetch did not start"
        assert len(cache._read_ahead.pending.sources) == 5
        assert cache.bytes + cache.prefetch_bytes <= cache.capacity_bytes
        # A pending read for another microbatch must not stall cached layers.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(groups[0][2].read)
            try:
                np.testing.assert_array_equal(future.result(timeout=1).numpy(), first_values[[3, 0], 2])
            finally:
                resume.set()
        for layer, source in enumerate(groups[1]):
            np.testing.assert_array_equal(source.read().numpy(), second_values[[1, 2], layer])
        first_reads = body.reads
        for sources, values, rows in ((groups[0], first_values, [3, 0]), (groups[1], second_values, [1, 2])):
            for layer, source in enumerate(sources):
                np.testing.assert_array_equal(source.read().numpy(), values[rows, layer])
        assert body.reads == first_reads
        assert cache._read_ahead.cold_reads == 1
        assert cache.prefetch_peak_bytes <= cache.capacity_bytes
    finally:
        resume.set()
        store.retire_partition("train_0")
        for sources in groups:
            for source in sources:
                source.release()
    assert not body.blocks and not inventory.state["objects"]


def test_reference_prefetch_keeps_uncached_current_layers_ahead_of_next_microbatch(published) -> None:
    _, store, body, inventory, ref, values = published
    store.attach([ref], "train_0")
    store.commit([ref])
    cache = ConsumerTensorCache(600)
    groups = []
    for rows, layers in (([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], (0, 1)), ([2], (0,))):
        layout = ConsumerBatchLayout.from_coordinates(np.array([[0, row] for row in rows]))
        with store.reader([ref], partition="train_0") as session:
            groups.append(
                [
                    ConsumerLayerReader(
                        session, layout, layer=layer, kind="moe", device="cpu", num_experts=256, cache=cache
                    )
                    for layer in layers
                ]
            )
    try:
        np.testing.assert_array_equal(groups[0][0].read().numpy(), values[[0, 1, 2, 3, 0, 1, 2, 3, 0, 1], 0])
        # The next current layer cannot fit the conservative reservation.
        # A smaller future microbatch must not jump ahead and make it wait.
        assert cache._read_ahead.pending is None
        groups[0][1].read()
        np.testing.assert_array_equal(groups[1][0].read().numpy(), values[[2], 0])
        assert cache.prefetch_peak_bytes <= cache.capacity_bytes
    finally:
        store.retire_partition("train_0")
        for sources in groups:
            for source in sources:
                source.release()
    assert not body.blocks and not inventory.state["objects"]


def test_reference_wire_round_trip_validates_token_identity_without_body_reads(published):
    config, store, body, _, ref, _ = published
    manager = ReplayManager(config=config, store_factory=lambda: store)
    consumer = manager.register_consumer({"moe"})
    assert consumer.fields(["tokens", "rollout_routed_experts"]) == ["tokens", "rollout_replay_refs"]
    packed = manager.pack("rollout_replay_refs", [{"moe": ref.to_dict()}])
    batch = {"tokens": [[10, 11, 12, 13, 14]], "rollout_replay_refs": list(packed.unbind())}
    consumer.receive_references(batch)
    assert batch["rollout_replay_refs"] == [{"moe": ref.to_dict()}]
    assert body.reads == 0
    batch = {"tokens": [[10, 11, 12, 13, 99]], "rollout_replay_refs": list(packed.unbind())}
    with pytest.raises(ValueError, match="does not match sample tokens"):
        consumer.receive_references(batch)


@pytest.mark.asyncio
async def test_reference_delete_failure_drains_units_and_preserves_retry_metadata(published):
    _, _, _, inventory, ref, _ = published
    metadata = SimpleNamespace(global_indexes=[1, 2])
    remaining = {1, 2}
    events = []
    fail = True

    async def clear(indexes, *, target_storage_unit):
        if target_storage_unit == 1 and fail:
            raise RuntimeError("storage unit unavailable")
        await asyncio.sleep(0)
        remaining.difference_update(indexes)
        events.append(target_storage_unit)

    async def clear_meta(meta):
        assert meta is metadata and not remaining
        events.append("metadata")

    client = SimpleNamespace(
        storage_manager=SimpleNamespace(
            _group_by_hash=lambda _: {i: SimpleNamespace(global_indexes=[i]) for i in (1, 2)},
            _clear_single_storage_unit=clear,
        ),
        _clear_meta_in_controller=clear_meta,
    )
    with pytest.raises(RuntimeError, match="storage unit unavailable"):
        await _clear_simple_storage_samples(client, metadata)
    assert events == [2] and remaining == {1}

    # Exercise the inventory contract: a failed deletion must retain its record.
    def fail_delete(keys):
        raise RuntimeError("storage unit unavailable")

    inventory._delete = fail_delete
    with pytest.raises(RuntimeError, match="storage unit unavailable"):
        inventory.discard([ref])
    assert ref.record in inventory.state["objects"]
    fail = False
    await _clear_simple_storage_samples(client, metadata)
    assert events[-1] == "metadata"
    inventory._delete = lambda keys: None
    inventory.discard([ref])
    assert not inventory.state["objects"]


def test_reference_restart_replaces_registry_and_rejects_previous_generation(published, monkeypatch):
    config, store, _, _, ref, _ = published
    monkeypatch.setenv("RELAX_REPLAY_RUN_ID", config.run_id)
    manager = ReplayManager(config=config)
    manager._store_instance = store
    manager.start_run()
    # A dead consumer retains a lease; normal close would reject this state.
    store.acquire_many([ref])
    runtime_env = {"env_vars": {"EXISTING": "value"}}
    manager.prepare_restart(runtime_env)
    assert manager.config.run_id != config.run_id
    assert runtime_env["env_vars"] == {"EXISTING": "value", "RELAX_REPLAY_RUN_ID": manager.config.run_id}
    assert os.environ["RELAX_REPLAY_RUN_ID"] == manager.config.run_id
    assert not manager._owns_run and manager._store_instance is None
    new_inventory = ReplayInventory(manager.config, lambda keys: None)
    replacement = TQReplayStorage(manager.config, inventory=new_inventory)
    manager._store_instance = replacement
    manager.start_run()
    with pytest.raises(RuntimeError, match="another replay run"):
        replacement.acquire_many([ref])
    assert not new_inventory.state["objects"] and not new_inventory.state["closed_partitions"]
    manager.finish_run()


def test_reference_failed_finalization_is_reported_without_retrying_closed_tq(published):
    config, store, _, _, ref, _ = published
    manager = ReplayManager(config=config)
    manager._store_instance = store
    manager.start_run()
    store.acquire_many([ref])
    with pytest.raises(RuntimeError, match="writers, readers or queue holds"):
        manager.finish_run()
    assert not manager._owns_run and manager._run_stopped
    # The second call is the entrypoint after Controller has destroyed TQ.
    store.close_run = lambda: pytest.fail("must not access the closed TQ")
    with pytest.raises(RuntimeError, match="previously failed"):
        manager.finish_run()


def test_reference_capture_reads_modern_request_kv_slots(published, monkeypatch):
    from relax.distributed.replay_manager.backend import sglang

    config, store, body, _, _, _ = published
    manager = ReplayManager(config=config, store_factory=lambda: store)
    monkeypatch.setattr(sglang, "replay_data_manager", manager)
    buffer = torch.arange(6 * 3 * 2, dtype=torch.int32).reshape(6, 3, 2)
    capturer = SimpleNamespace(host_cache=SimpleNamespace(buffer=buffer))
    pool = SimpleNamespace(req_to_token=torch.tensor([[4, 1, 5, 0]]))
    processor = SimpleNamespace(req_to_token_pool=pool)
    req = SimpleNamespace(
        kv=SimpleNamespace(req_pool_idx=0),
        return_routed_experts=True,
        origin_input_ids=[10, 11],
        output_ids_through_stop=[12, 13, 14],
    )
    refs = sglang._advance_publication(processor, req, 4, {"moe": lambda: capturer}, final=True)
    ref = sglang.ReplayRef.from_dict(refs["moe"])
    with store.reader([ref]) as reader:
        for layer in range(3):
            output = np.empty((4, 2), np.int32)
            reader.read_into(0, layer, np.arange(4), output, np.arange(4))
            np.testing.assert_array_equal(output, buffer[[4, 1, 5, 0], layer].numpy())
    store.discard([ref])
    assert all(not key.startswith(ref.record) for key in body.blocks)


def test_bitmap_publication_preserves_codec_and_accounts_encoded_bytes():
    from relax.distributed.replay_manager.ref_data import decode_refs, encode_refs

    config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=2, publication_bytes=32768)
    body = MemoryBody()
    inventory = ReplayInventory(config, body.delete)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = body
    publication = store.begin_publication(kind="indexer", layers=2, topk=2048, codec="keye-bitmap-v1")
    values = np.zeros((2, 65), dtype=np.int32)
    values[:, 0] = -18  # implicit FULL prefix of 17 tokens
    publication.append(2, lambda layer: values.copy(), encoded_width=65)
    ref = publication.finish([10, 11, 12])
    assert ref.codec == "keye-bitmap-v1"
    assert inventory.state["objects"][ref.record]["bytes"] == values.nbytes * 2
    assert decode_refs(encode_refs({"indexer": ref.to_dict()})) == {"indexer": ref.to_dict()}
    store.discard([ref])
    assert not body.blocks and not inventory.state["objects"]
    store.close_run()


def test_encoded_capture_publication_uses_adapter_shape_and_snapshot(monkeypatch):
    from relax.distributed.replay_manager.backend import sglang
    from relax.distributed.replay_manager.codecs import KEYE_BITMAP

    config = ReplayConfig(delivery="reference", run_id="ada", chunk_rows=2, publication_bytes=32768)
    body = MemoryBody()
    inventory = ReplayInventory(config, body.delete)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = body
    manager = ReplayManager(config=config, store_factory=lambda: store)
    monkeypatch.setattr(sglang, "replay_data_manager", manager)
    # No model or CUDA dependency: exercise the real scheduler publication and
    # inventory using an adapter's header-only encoded CPU capture storage.
    buffer = torch.full((4, 2, 1025), 1234, dtype=torch.int32)
    buffer[:, :, 0] = -2
    capturer = SimpleNamespace(
        host_cache=SimpleNamespace(buffer=buffer),
        publication_codec=KEYE_BITMAP,
        topk_size=2048,
        encoded_width=lambda stop: 1,
        snapshot_layer=lambda slots, layer, stop: buffer[slots, layer, :1].numpy().copy(),
    )
    processor = SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[3, 1, 2, 0]])))
    req = SimpleNamespace(
        kv=SimpleNamespace(req_pool_idx=0),
        return_indexer_topk=True,
        origin_input_ids=[10, 11],
        output_ids_through_stop=[12, 13, 14],
    )
    try:
        refs = sglang._advance_publication(processor, req, 4, {"indexer": lambda: capturer}, final=True)
        ref = sglang.ReplayRef.from_dict(refs["indexer"])
        assert ref.codec == KEYE_BITMAP.name and ref.topk == 2048 and ref.layers == 2
        assert inventory.state["objects"][ref.record]["bytes"] == 4 * 2 * 4
        for value in body.blocks.values():
            np.testing.assert_array_equal(value, np.full((2, 1), -2, dtype=np.int32))
        store.discard([ref])
    finally:
        store._writer.close()
