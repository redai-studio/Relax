# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the bucketed LoRA-adapter broadcast in DeviceDirectBackend.

The adapter push is a two-sided protocol: this process broadcasts tensors in
buckets and the SGLang engine (see ``update_lora_from_distributed`` in
``docker/patch/sglang/``) allocates and receives them using the SAME boundaries,
shipped in the payload as ``bucket_sizes``. A mismatch does not raise — it makes
the two sides issue different broadcasts and hang until the NCCL timeout — so
the boundary rule is pinned down here.

Bucketing exists because a MoE adapter is multi-GB (40 layers x 256 experts x
rank 32 is ~3.4 GiB in BF16); staging it whole would spike the sender and every
receiving engine by the full adapter size.
"""

import random

import pytest


# DeviceDirectBackend imports megatron.core at module level; skip on CPU-only checkouts.
pytest.importorskip("megatron.core")

from relax.distributed.checkpoint_service.backends.device_direct import (  # noqa: E402
    bucket_tensor_counts,
)


class TestBucketTensorCounts:
    def test_everything_fits_in_one_bucket(self):
        assert bucket_tensor_counts([10, 20, 30], 100) == [3]

    def test_splits_when_the_next_tensor_would_overflow(self):
        # 40 + 40 = 80 fits; adding the third would be 120, so it starts a new bucket.
        assert bucket_tensor_counts([40, 40, 40], 100) == [2, 1]

    def test_exactly_at_the_cap_still_splits(self):
        assert bucket_tensor_counts([100, 100], 100) == [1, 1]

    def test_oversized_tensor_gets_its_own_bucket(self):
        """The transport is per-tensor, so one tensor is the smallest unit —
        never drop it and never merge it with a neighbour."""
        assert bucket_tensor_counts([500], 100) == [1]
        assert bucket_tensor_counts([10, 500, 10], 100) == [1, 1, 1]

    def test_empty_input(self):
        assert bucket_tensor_counts([], 100) == []

    def test_counts_always_cover_every_tensor(self):
        """``sum(counts) == len(sizes)`` is the invariant the engine asserts
        on; breaking it desyncs the broadcast rather than raising."""
        random.seed(7)
        for _ in range(200):
            sizes = [random.randint(1, 300) for _ in range(random.randint(0, 40))]
            cap = random.randint(50, 400)
            counts = bucket_tensor_counts(sizes, cap)
            assert sum(counts) == len(sizes)
            assert all(count >= 1 for count in counts)

    def test_buckets_respect_the_cap_unless_a_single_tensor_exceeds_it(self):
        random.seed(11)
        for _ in range(200):
            sizes = [random.randint(1, 300) for _ in range(random.randint(1, 40))]
            cap = random.randint(50, 400)
            offset = 0
            for count in bucket_tensor_counts(sizes, cap):
                total = sum(sizes[offset : offset + count])
                assert total <= cap or count == 1
                offset += count


class TestSenderReceiverAgreement:
    """Replay the engine's slicing loop against the sender's counts."""

    def test_receiver_walk_reconstructs_the_exact_tensor_order(self):
        names = [f"t{i}" for i in range(37)]
        sizes = [random.Random(3).randint(1, 200) for _ in names]
        counts = bucket_tensor_counts(sizes, 256)

        # Mirror of the engine loop: names[offset : offset + count] per bucket.
        seen = []
        offset = 0
        for count in counts:
            seen.extend(names[offset : offset + count])
            offset += count

        assert seen == names, "receiver would receive tensors in a different order"
        assert offset == len(names), "receiver would stop before the last tensor"
