# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real raw-indexer reads with controlled external body I/O."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from relax.distributed.replay_manager.config import ReplayConfig
from relax.distributed.replay_manager.consumer import ConsumerBatchLayout, ConsumerLayerReader, ConsumerTensorCache
from relax.distributed.replay_manager.core import ReplayInventory
from relax.distributed.replay_manager.storage import TQReplayStorage
from tests.cpu.distributed.replay_manager.test_reference import MemoryBody


@pytest.fixture
def raw_indexer():
    body = MemoryBody()
    config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=2, publication_bytes=32)
    inventory = ReplayInventory(config, body.delete)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = body
    values = np.arange(40, dtype=np.int32).reshape(4, 5, 2)
    publication = store.begin_publication(kind="indexer", layers=5, topk=2)
    for start in (0, 2):
        publication.append(2, lambda layer, start=start: values[start : start + 2, layer].copy())
    ref = publication.finish([10, 11, 12, 13, 14])
    store.attach([ref], "train_0")
    store.commit([ref])
    yield store, body, ref, values
    store._writer.close()


def sources_for(store, ref, capacity):
    cache = ConsumerTensorCache(capacity)
    layout = ConsumerBatchLayout.from_coordinates(np.array([[0, row] for row in range(4)]))
    with store.reader([ref], partition="train_0") as reader:
        sources = [
            ConsumerLayerReader(reader, layout, layer=layer, kind="indexer", device="cpu", cache=cache)
            for layer in range(5)
        ]
    return cache, sources


@pytest.mark.parametrize("fail_later", [False, True])
def test_indexer_prefetch_publishes_ready_layer_before_window_and_propagates_late_failure(raw_indexer, fail_later):
    store, body, ref, expected = raw_indexer
    cache, sources = sources_for(store, ref, 256)
    blocked, resume = Event(), Event()
    original_read = body.read

    def read(keys):
        if any(key.endswith(":2") for key in keys):
            blocked.set()
            assert resume.wait(5), "test did not release the delayed body read"
            if fail_later:
                raise RuntimeError("injected late-layer read failure")
        return original_read(keys)

    body.read = read
    try:
        sources[0].read()
        assert blocked.wait(5)
        with ThreadPoolExecutor(max_workers=1) as executor:
            ready = executor.submit(sources[1].read)
            try:
                values, mask = ready.result(timeout=2)
                np.testing.assert_array_equal(values.numpy(), expected[:, 1])
                assert mask.all()
                values.fill_(99)
                assert cache.bytes + cache.prefetch_bytes <= cache.capacity_bytes
            finally:
                resume.set()
        if fail_later:
            with pytest.raises(RuntimeError, match="late-layer"):
                cache._read_ahead.pending.future.result(timeout=5)
            # Even a warm hit must report a completed background failure.
            with pytest.raises(RuntimeError, match="late-layer"):
                sources[0].read()
            assert sources[1]._cache_key not in cache.entries
        else:
            for layer, source in enumerate(sources):
                actual, mask = source.read()
                np.testing.assert_array_equal(actual.numpy(), expected[:, layer])
                assert mask.all()
        assert cache.prefetch_peak_bytes <= cache.capacity_bytes
    finally:
        resume.set()
        store.retire_partition("train_0")
        for source in sources:
            source.release()
    assert not body.blocks


def test_indexer_prefetch_follows_reverse_consumption_under_cache_pressure(raw_indexer):
    store, body, ref, expected = raw_indexer
    cache, sources = sources_for(store, ref, 160)
    reads = []
    original_read = body.read

    def read(keys):
        reads.extend(keys)
        return original_read(keys)

    body.read = read
    try:
        for source in sources:
            source.read()
        cache._read_ahead.drain()
        reads.clear()
        sources[3].read()
        cache._read_ahead.drain()
        sources[2].read()
        cache._read_ahead.drain()
        reads.clear()
        sources[1].read()
        assert cache._read_ahead.pending is not None
        cache._read_ahead.pending.future.result(timeout=5)
        assert any(key.endswith(":0") for key in reads)
        actual, mask = sources[0].read()
        np.testing.assert_array_equal(actual.numpy(), expected[:, 0])
        assert mask.all()
        assert cache.evictions > 0
        assert cache.prefetch_peak_bytes <= cache.capacity_bytes
        assert cache._read_ahead.prefetch_ready_hits > 0
    finally:
        store.retire_partition("train_0")
        for source in sources:
            source.release()
    assert not body.blocks


def test_progressive_read_rejects_overlapping_outputs_before_io(raw_indexer):
    store, body, ref, _ = raw_indexer
    with store.reader([ref], partition="train_0") as reader:
        output = np.zeros((4, 2), dtype=np.int32)
        request = [(ref, np.arange(4), np.arange(4), reader._leases[0])]
        reads = body.reads
        with pytest.raises(ValueError, match="non-overlapping"):
            store._read_layers_into([(request, 0, output), (request, 1, output)], on_layer=lambda _: None)
        assert body.reads == reads
    store.retire_partition("train_0")
