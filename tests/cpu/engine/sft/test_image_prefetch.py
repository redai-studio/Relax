# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.engine.sft.image import ImageBatchSampler
from relax.engine.sft.image_prefetch import ImagePrefetch


@pytest.mark.parametrize("start", [0, 2, 5])
def test_image_prefetch_order_resume_rng_and_bound(start):
    sampler = ImageBatchSampler(19, 8, 42)
    rng = torch.get_rng_state().clone()
    prefetch = ImagePrefetch(list(range(19)), sampler, 1, 4, start, 9, depth=3)
    try:
        for step in range(start, 9):
            for micro, sample_id in enumerate(sampler.batch(step)[1::4]):
                assert len(prefetch.pending) <= 3
                assert prefetch.get(step, micro, sample_id) == sample_id
        assert not prefetch.pending
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        prefetch.close()
    assert prefetch.closed


def test_image_prefetch_rejects_cursor_mismatch():
    prefetch = ImagePrefetch(list(range(8)), ImageBatchSampler(8, 8, 42), 0, 4, 0, 1)
    try:
        with pytest.raises(ValueError, match="cursor"):
            prefetch.get(1, 0, 0)
    finally:
        prefetch.close()


def test_image_prefetch_propagates_decode_error_and_closes():
    class Dataset:
        def __getitem__(self, index):
            raise OSError("decode failed")

    sampler = ImageBatchSampler(8, 8, 42)
    prefetch = ImagePrefetch(Dataset(), sampler, 0, 4, 0, 1)
    with pytest.raises(OSError, match="decode failed"):
        prefetch.get(0, 0, sampler.batch(0)[0])
    assert prefetch.closed
    assert not prefetch.pending


@pytest.mark.parametrize("depth,workers", [(0, 2), (-1, 2), (4, 0)])
def test_image_prefetch_rejects_invalid_bounds(depth, workers):
    with pytest.raises(ValueError):
        ImagePrefetch([], ImageBatchSampler(8, 8, 42), 0, 4, 0, 1, depth, workers)
