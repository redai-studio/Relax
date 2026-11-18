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
