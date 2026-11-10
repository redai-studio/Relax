# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import os
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
