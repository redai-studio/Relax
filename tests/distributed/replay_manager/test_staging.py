# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Actual CUDA DMA, consumer streams and bounded pinned-buffer reuse."""

import numpy as np
import pytest
import torch

from relax.distributed.replay_manager import _staging
from relax.distributed.replay_manager.consumer import ConsumerBatchLayout, ConsumerLayerReader, ConsumerTensorCache
from tests.cpu.distributed.replay_manager.test_prefetch import raw_indexer  # noqa: F401


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA pinned-memory DMA")


def test_indexer_staging_reuses_slots_across_consumer_streams_and_oversized_copies(monkeypatch):
    monkeypatch.setattr(_staging, "_MIN_SLOT_BYTES", 256)
    monkeypatch.setattr(_staging, "_MAX_SLOT_BYTES", 4096)
    device = torch.device("cuda", torch.cuda.current_device())
    pool = _staging._PinnedCopies(device)
    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    results = []
    for index, rows in enumerate((8, 16, 32, 64, 128, 256, 512, 16, 128, 32)):
        values = torch.arange(rows * 4, dtype=torch.int32).reshape(rows, 4) + index * 10000
        mask = torch.arange(rows) % 3 != 0
        expected, expected_mask = values.clone(), mask.clone()
        with torch.cuda.stream(streams[index % 2]):
            # Keep consuming work outstanding while later calls reuse the DMA
            # stream and allocator. Outputs may be freed before this work runs.
            torch.cuda._sleep(1000000)
            output, output_mask = pool.copy(values, mask)
            result = output + 7
            result_mask = ~output_mask
            del output, output_mask
        values.fill_(-99)
        mask.fill_(False)
        results.append((result, result_mask, expected + 7, ~expected_mask))
    for stream in streams:
        stream.synchronize()
    for actual, actual_mask, expected, expected_mask in results:
        torch.testing.assert_close(actual.cpu(), expected)
        torch.testing.assert_close(actual_mask.cpu(), expected_mask)
    retained = [slot[0].numel() for slot in pool.slots if slot is not None]
    assert len(retained) == 2 and sum(retained) <= 2 * 4096


@pytest.mark.parametrize("capacity", [0, 256])
def test_raw_indexer_cuda_reads_preserve_padding_recompute_and_caller_ownership(raw_indexer, capacity):  # noqa: F811
    store, body, ref, expected = raw_indexer
    cache = ConsumerTensorCache(capacity)
    layout = ConsumerBatchLayout.from_coordinates(np.array([[0, 3], [0, 0], [-1, 0], [0, 2]]))
    with store.reader([ref], partition="train_0") as reader:
        sources = [
            ConsumerLayerReader(reader, layout, layer=layer, kind="indexer", device="cuda", cache=cache)
            for layer in range(5)
        ]
    store.retire_partition("train_0")
    try:
        for _ in range(3):
            for layer, source in enumerate(sources):
                values, mask = source.read()
                wanted = np.stack((expected[3, layer], expected[0, layer], [-1, -1], expected[2, layer]))
                np.testing.assert_array_equal(values.cpu().numpy(), wanted)
                assert mask.cpu().tolist() == [True, True, False, True]
                values.fill_(99)
                mask.fill_(False)
        for tensors, _ in cache.entries.values():
            assert all(not tensor.is_pinned() for tensor in tensors if tensor is not None)
    finally:
        for source in sources:
            source.release()
    assert not body.blocks
