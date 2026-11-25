# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.distributed.weight_sync.codec import encode
from relax.distributed.weight_sync.expert_lanes import ExpertLanePlan


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA expert lane compaction")
@pytest.mark.parametrize(
    "dtype,word,nan_word",
    [
        (torch.bfloat16, torch.int16, 0x7F81),
        (torch.float32, torch.int32, 0x7F800001),
        (torch.float64, torch.int64, 0x7FF0000000000001),
    ],
)
def test_gpu_partition_matches_reference_wire_bytes_for_mixed_native_slices(dtype, word, nan_word):
    workers = []
    for expert in range(2):
        for rank in range(2):
            route = dict(
                prefix="layer.experts",
                expert_start=expert,
                expert_stop=expert + 1,
                intermediate_start=48 * rank,
                intermediate_stop=48 * (rank + 1),
                intermediate_size=96,
                hidden_size=37,
                dtype=str(dtype).removeprefix("torch."),
            )
            workers.append(dict(routes=[route]))
    plan = ExpertLanePlan([workers])
    rows = []
    for expert, projection, count in (
        (0, "gate_proj", 513),
        (1, "up_proj", 0),
        (1, "down_proj", 700),
        (0, "down_proj", 444),
        (1, "gate_proj", 257),
    ):
        shape = (37, 96) if projection == "down_proj" else (96, 37)
        positions = torch.arange(count, dtype=torch.int32) * 17 % (shape[0] * shape[1])
        words = torch.arange(count, dtype=word)
        if count:
            positions[:5] = torch.tensor([0, 0, shape[1], shape[0] * shape[1] - 1, shape[1] + 1])
            words[:5] = torch.tensor([0, torch.iinfo(word).min, nan_word, -1, 1], dtype=word)
        rows.append((f"layer.experts.{expert}.{projection}.weight", shape, dtype, positions, words.view(dtype)))
    limit = 1 << 20
    directory, payload = encode(rows, max_bytes=limit)
    expected_meta, expected_slabs = plan.partition(directory, payload, limit, {})
    actual_meta, actual_slabs = plan.partition(directory, payload.cuda(), limit, {})
    assert actual_meta == expected_meta
    assert len(actual_slabs) == len(workers)
    for actual, expected in zip(actual_slabs, expected_slabs, strict=True):
        assert torch.equal(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA expert lane compaction")
def test_signed_coordinates_keep_the_reference_lane_and_local_position():
    from relax.distributed.weight_sync.expert_lanes import _select_lane

    # Generated HF coordinates are nonnegative. A malformed position must follow the
    # same signed arithmetic on both paths before receiver validation.
    indices = torch.tensor([-(1 << 31), -15, -8, -7, -1, 0, 1, 7, 15], dtype=torch.int32).repeat(2)
    values = torch.arange(indices.numel(), dtype=torch.float32)
    lengths = [9, 9]
    slices = [(0, 0, 4, 7), (1, 3, 7, 7)]
    expected = _select_lane(indices, values, lengths, slices)
    actual = _select_lane(indices.cuda(), values.cuda(), lengths, slices)
    assert actual[2] == expected[2]
    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[1].cpu(), expected[1])
