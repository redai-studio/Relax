# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Media bit preservation and shared ownership use the real replay state
machine."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from relax.distributed.replay_manager.config import ReplayConfig
from relax.distributed.replay_manager.core import ReplayInventory, ReplayManager
from relax.distributed.replay_manager.mm_embedding import MMPublisher, read_mm_embedding, release_media_entries
from relax.distributed.replay_manager.ref_data import (
    ReplayRef,
    decode_refs,
    encode_refs,
    iter_refs,
    token_digest,
    validate_sample,
)
from relax.distributed.replay_manager.storage import TQReplayStorage


class Body:
    def __init__(self) -> None:
        self.values = {}

    def write_many(self, keys, values) -> None:
        self.values.update((key, value.copy()) for key, value in zip(keys, values, strict=True))

    def read(self, keys):
        return [self.values[key].copy() for key in keys]

    def delete(self, keys) -> None:
        for key in keys:
            self.values.pop(key, None)

    def close(self) -> None:
        pass


@pytest.fixture
def media_store():
    body = Body()
    config = ReplayConfig(delivery="reference", run_id="abcd", capacity_bytes=4 * 2560 * 2, chunk_rows=2)
    inventory = ReplayInventory(config, body.delete)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = body
    yield store, inventory, body
    if store._writer is not None:
        store._writer.close()


def bits() -> np.ndarray:
    # Includes BF16 NaNs, infinities, signed zeros and subnormal bit patterns.
    values = np.arange(4 * 2560, dtype=np.uint16).reshape(4, 2560)
    values[0, :8] = [0, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0xFFC1, 1, 0x8001]
    return values.view(np.uint8).reshape(4, 5120)


def sample(entry):
    return SimpleNamespace(tokens=[7, 8, 9], rollout_replay_refs={"mm_embedding": {"version": 1, "media": [entry]}})


def test_mm_embedding_roundtrip_preserves_raw_bf16_bits_and_capacity(media_store):
    store, inventory, _ = media_store
    entry = MMPublisher(store).publish("a" * 64, bits(), [1, 4, 4])
    result = read_mm_embedding(store, entry)
    assert result.dtype == torch.bfloat16 and result.shape == (4, 2560)
    np.testing.assert_array_equal(result.view(torch.uint8).numpy(), bits())
    assert inventory.state["objects"][entry["ref"]["record"]]["bytes"] == bits().nbytes
    wrapped = sample(entry).rollout_replay_refs
    indexer = ReplayRef(2, "abcd", "abcd", "f" * 32, "indexer", 2, 1, 512, 2, token_digest([7, 8, 9]))
    wrapped["indexer"] = indexer.to_dict()
    assert decode_refs(encode_refs(wrapped)) == wrapped
    validate_sample(wrapped, [7, 8, 9])
    assert list(iter_refs(wrapped)) == [ReplayRef.from_dict(entry["ref"]), indexer]
    release_media_entries(store, [entry])
    assert inventory.state["objects"] == {}


def test_mm_embedding_hash_dedup_survives_discard_queue_retirement_and_reader(media_store):
    store, inventory, body = media_store
    publisher = MMPublisher(store)
    first = publisher.publish("b" * 64, bits(), [1, 4, 4])
    second = publisher.publish("b" * 64, np.zeros_like(bits()), [1, 4, 4])
    assert first["ref"] == second["ref"] and first["lease"] != second["lease"]
    # Image hash, not differing output values, selects the first canonical body.
    np.testing.assert_array_equal(read_mm_embedding(store, second).view(torch.uint8).numpy(), bits())
    manager = ReplayManager(config=store.config, store_factory=lambda: store)
    manager.discard([sample(first)])
    assert body.values
    with manager.region("train_0").queue_ownership([sample(second)]):
        pass
    ref = ReplayRef.from_dict(second["ref"])
    reader_lease = store.acquire_many([ref], partition="train_0")[0]
    manager.region("train_0").retire()
    assert body.values
    result = read_mm_embedding(store, second, lease=reader_lease)
    np.testing.assert_array_equal(result.view(torch.uint8).numpy(), bits())
    store.release_many([ref], [reader_lease])
    assert inventory.state["objects"] == {} and not body.values
    # A later cache miss can republish after all ownership was reclaimed.
    third = publisher.publish("b" * 64, bits(), [1, 4, 4])
    assert third["ref"]["record"] != ref.record
    release_media_entries(store, [third, third])
    assert not body.values


def test_mm_embedding_concurrent_publishers_share_first_media_body(media_store):
    store, inventory, _ = media_store
    entered, proceed = Event(), Event()

    def snapshot():
        entered.set()
        assert proceed.wait(5)
        return bits()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(MMPublisher(store).publish, "c" * 64, snapshot, [1, 4, 4])
        assert entered.wait(5)
        second = executor.submit(MMPublisher(store).publish, "c" * 64, np.zeros_like(bits()), [1, 4, 4])
        proceed.set()
        entries = [first.result(timeout=5), second.result(timeout=5)]
    assert entries[0]["ref"] == entries[1]["ref"]
    assert len(inventory.state["objects"]) == 1
    release_media_entries(store, entries)
    assert not inventory.state["objects"]


def test_mm_embedding_geometry_mismatch_and_failed_snapshot_do_not_poison_hash(media_store):
    store, inventory, _ = media_store
    publisher = MMPublisher(store)
    with pytest.raises(ValueError, match="raw bytes"):
        publisher.publish("d" * 64, np.zeros((4, 2560), dtype=np.float32), [1, 4, 4])
    assert not inventory.state["objects"]
    entry = publisher.publish("d" * 64, bits(), [1, 4, 4])
    with pytest.raises(ValueError, match="geometry"):
        publisher.lookup("d" * 64, [1, 2, 8])
    with pytest.raises(RuntimeError, match="active reader lease"):
        read_mm_embedding(store, entry, lease="f" * 32)
    release_media_entries(store, [entry])


def test_mm_embedding_empty_text_envelope_uses_no_tensor_bodies():
    refs = {"mm_embedding": {"version": 1, "media": []}}
    validate_sample(refs, [1, 2])
    assert decode_refs(encode_refs(refs)) == refs
    assert list(iter_refs(refs)) == []


def test_mm_embedding_lost_lookup_ack_rolls_back_its_reader_hold(media_store, monkeypatch):
    store, inventory, _ = media_store
    publisher = MMPublisher(store)
    entry = publisher.publish("e" * 64, bits(), [1, 4, 4])
    invoke = store.invoke
    before = list(inventory.state["objects"][entry["ref"]["record"]]["readers"])

    def lose_reply(method, *args, **kwargs):
        result = invoke(method, *args, **kwargs)
        if method == "lookup_media":
            raise ConnectionError("lost metadata acknowledgement")
        return result

    monkeypatch.setattr(store, "invoke", lose_reply)
    with pytest.raises(ConnectionError, match="acknowledgement"):
        publisher.lookup("e" * 64, [1, 4, 4])
    assert inventory.state["objects"][entry["ref"]["record"]]["readers"] == before
    release_media_entries(store, [entry])
    assert not inventory.state["objects"]


def test_mm_embedding_first_hold_is_installed_before_other_producer_can_release(media_store, monkeypatch):
    store, inventory, _ = media_store
    publisher = MMPublisher(store)
    invoke = store.invoke

    def interleave(method, *args, **kwargs):
        result = invoke(method, *args, **kwargs)
        if method == "finish_publication":
            competing = publisher.lookup("f" * 64, [1, 4, 4])
            release_media_entries(store, [competing])
        return result

    monkeypatch.setattr(store, "invoke", interleave)
    entry = publisher.publish("f" * 64, bits(), [1, 4, 4])
    np.testing.assert_array_equal(read_mm_embedding(store, entry).view(torch.uint8).numpy(), bits())
    release_media_entries(store, [entry])
    assert not inventory.state["objects"]


def test_mm_embedding_row_snapshots_support_media_larger_than_publication_budget():
    body = Body()
    config = ReplayConfig(delivery="reference", run_id="abcd", capacity_bytes=4 * 5120, publication_bytes=5120)
    inventory = ReplayInventory(config, body.delete)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = body
    snapshots = []

    def snapshot(start, stop):
        snapshots.append((start, stop))
        return np.full((stop - start, 5120), start, dtype=np.uint8)

    try:
        entry = MMPublisher(store).publish_chunks("1" * 64, snapshot, [1, 4, 4])
        assert snapshots == [(0, 1), (1, 2), (2, 3), (3, 4)]
        actual = read_mm_embedding(store, entry).view(torch.uint8).numpy()
        np.testing.assert_array_equal(actual[:, 0], np.arange(4, dtype=np.uint8))
        release_media_entries(store, [entry])
        assert not body.values
    finally:
        if store._writer is not None:
            store._writer.close()


def test_mm_embedding_driver_close_reclaims_only_producer_holds(media_store):
    store, inventory, body = media_store
    publisher = MMPublisher(store)
    entry = publisher.publish("2" * 64, bits(), [1, 4, 4])
    cache = publisher.lookup("2" * 64, [1, 4, 4])
    assert entry["lease"] != cache["lease"]
    ref = ReplayRef.from_dict(entry["ref"])
    training_lease = store.acquire_many([ref])[0]
    with pytest.raises(RuntimeError, match="readers"):
        store.close_run()
    assert body.values
    store.release_many([ref], [training_lease])
    # Engine termination can leave cache/response producer leases. The owning
    # driver may release those at run close after all training readers drained.
    store.close_run()
    assert inventory.state["closed"] and not inventory.state["objects"] and not body.values
