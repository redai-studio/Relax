# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Owner packing freezes reusable canonical values and preserves TP slices."""

import json
from contextlib import closing

import pytest
import torch

from relax.backends.megatron.weight_update.expert_layout import DirectExpertLayout, expert_slice_chunk
from relax.backends.megatron.weight_update.owner_expert_blocks import iter_owner_expert_blocks
from relax.distributed.weight_sync.streaming_delta import (
    SnapshotBudget,
    StreamingDeltaEncoder,
    steady_state_snapshot_bytes,
)


def canonical(*, incomplete=False):
    shared = torch.empty(2, 3, dtype=torch.bfloat16)
    for projection in ("gate_proj", "up_proj"):
        # Numeric expert order is deliberately reversed: placement uses IDs.
        for expert in (1, 0):
            if incomplete and expert == 0 and projection == "up_proj":
                continue
            shared.fill_(expert + (10 if projection == "up_proj" else 1))
            yield 0, expert, f"model.language_model.layers.0.mlp.experts.{expert}.{projection}.weight", shared
    for expert in (1, 0):
        shared.fill_(expert + 20)
        yield 0, expert, f"model.language_model.layers.0.mlp.experts.{expert}.down_proj.weight", shared.transpose(0, 1)
    shared.fill_(-99)


def test_owner_blocks_freeze_reused_converter_buffer_and_match_each_tp_native_slice(monkeypatch):
    import weakref

    from relax.backends.megatron.weight_update import owner_expert_blocks

    layout = DirectExpertLayout(4, 3, 2, 2)
    budget = SnapshotBudget(1 << 20)
    results = []
    allocated, original_empty = [], torch.empty

    def allocate(*args, **kwargs):
        slab = len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == 3
        if slab:
            assert all(reference() is None for reference in allocated), "Previous slab view outlived its budget lease"
        value = original_empty(*args, **kwargs)
        if slab:
            allocated.append(weakref.ref(value))
        return value

    monkeypatch.setattr(owner_expert_blocks.torch, "empty", allocate)
    with closing(iter_owner_expert_blocks(canonical(), layout, 0, 2, budget)) as blocks:
        for name, slab in blocks:
            results.append(
                (
                    name,
                    [
                        layout.shard(
                            slab,
                            "down_proj" if "down_proj" in name else "gate_up_proj",
                            tp,
                            expert_start=0,
                            expert_end=2,
                        ).clone()
                        for tp in range(2)
                    ],
                )
            )
            assert 0 < budget.used_bytes <= budget.limit_bytes
            del slab
    assert budget.used_bytes == 0 and len(results) == 2
    for _, shards in results:
        for tensor in shards:
            for expert in range(2):
                if tensor.shape[1:] == (2, 3):
                    assert torch.equal(tensor[expert, 0], torch.full((3,), expert + 1, dtype=torch.bfloat16))
                    assert torch.equal(tensor[expert, 1], torch.full((3,), expert + 10, dtype=torch.bfloat16))
                else:
                    assert torch.equal(tensor[expert], torch.full((3, 1), expert + 20, dtype=torch.bfloat16))


def test_owner_blocks_refuse_incomplete_projection_and_release_conversion_budget():
    layout, budget = DirectExpertLayout(4, 3, 2, 2), SnapshotBudget(1 << 20)
    with pytest.raises(ValueError, match="missing or duplicate"):
        list(iter_owner_expert_blocks(canonical(incomplete=True), layout, 0, 2, budget))
    assert budget.used_bytes == 0


def test_owner_block_lease_is_released_when_encoder_rejects_a_yield():
    layout, budget = DirectExpertLayout(4, 3, 2, 2), SnapshotBudget(1 << 20)
    with pytest.raises(RuntimeError, match="downstream"):
        with closing(iter_owner_expert_blocks(canonical(), layout, 0, 2, budget)) as blocks:
            next(blocks)
            raise RuntimeError("downstream encoder failed")
    assert budget.used_bytes == 0


@pytest.mark.parametrize("experts,hidden,intermediate,tp", [(6, 3, 4, 1), (8, 5, 6, 2), (12, 3, 8, 4)])
def test_bounded_owner_blocks_use_actual_geometry_and_global_owner_rank(experts, hidden, intermediate, tp):
    layout = DirectExpertLayout(experts, hidden, intermediate, tp, supports_expert_slices=True)
    local_count = experts // 2
    # At most two gate/up experts, including one live TP slice. Down may fit
    # more experts; each projection's stable chunking still covers its owner.
    capacity = 2 * (2 * intermediate * hidden * 2) * (tp + 1) // tp
    budget = SnapshotBudget(1 << 20)
    encoders = [StreamingDeltaEncoder(budget, bucket_bytes=capacity) for _ in range(tp)]

    def converted():
        shared = torch.empty(intermediate, hidden, dtype=torch.bfloat16)
        for projections in (("gate_proj", "up_proj"), ("down_proj",)):
            for expert in range(local_count, experts):
                for projection in projections:
                    shared.fill_(expert + {"gate_proj": 1, "up_proj": 20, "down_proj": 40}[projection])
                    tensor = shared.t() if projection == "down_proj" else shared
                    yield 17, expert, f"model.layers.0.mlp.experts.{expert}.{projection}.weight", tensor
        shared.fill_(-99)

    try:
        for encoder in encoders:
            encoder.begin()
        with closing(
            iter_owner_expert_blocks(
                converted(),
                layout,
                1,
                2,
                budget,
                bucket_bytes=capacity,
                source_rank=17,
            )
        ) as blocks:
            for name, slab in blocks:
                assert slab.numel() * slab.element_size() * (tp + 1) // tp <= capacity
                for rank, encoder in enumerate(encoders):
                    chunk = expert_slice_chunk([(name, slab)], layout, rank)
                    for descriptor, shard in chunk:
                        encoder.add(descriptor, shard)
                    del chunk, shard
                del slab
        for encoder in encoders:
            coverage = {}
            for bucket in encoder.finish():
                flat = torch.frombuffer(bytearray(bucket.payload), dtype=torch.bfloat16)
                offset = 0
                for descriptor, shape in bucket.names:
                    name, start, _ = json.loads(descriptor.split(":", 1)[1])
                    count = torch.Size(shape).numel()
                    value = flat[offset : offset + count].reshape(shape)
                    offset += count
                    coverage.setdefault(name, []).extend(range(start, start + shape[0]))
                    for index, expert in enumerate(range(start, start + shape[0])):
                        if name.endswith("w13_weight"):
                            gate, up = value[index].chunk(2, dim=0)
                            assert torch.equal(gate, torch.full_like(gate, expert + 1))
                            assert torch.equal(up, torch.full_like(up, expert + 20))
                        else:
                            assert torch.equal(value[index], torch.full_like(value[index], expert + 40))
            assert len(coverage) == 2
            assert all(indices == list(range(local_count, experts)) for indices in coverage.values())
    finally:
        for encoder in encoders:
            encoder.close()
    assert budget.used_bytes == 0


@pytest.mark.parametrize("capacity,budget_bytes", [(35, 1024), (72, 35), (72, 50)])
def test_owner_packing_rejects_single_expert_capacity_and_shared_budget(capacity, budget_bytes):
    budget = SnapshotBudget(budget_bytes)
    with pytest.raises(ValueError, match="single expert|shared rank budget"):
        list(
            iter_owner_expert_blocks(canonical(), DirectExpertLayout(4, 3, 2, 2), 0, 2, budget, bucket_bytes=capacity)
        )
    assert budget.used_bytes == 0


@pytest.mark.parametrize("limit", [1000, 1500, 2000, 3000])
def test_steady_admission_rejects_first_full_that_cannot_fit_its_next_delta(limit):
    layout = DirectExpertLayout(8, 3, 8, 2, supports_expert_slices=True)
    budget = SnapshotBudget(limit)
    encoders = [StreamingDeltaEncoder(budget, bucket_bytes=144) for _ in range(layout.tp_size)]

    def converted():
        for projections in (("gate_proj", "up_proj"), ("down_proj",)):
            for expert in range(4):
                for projection in projections:
                    shape = (3, 8) if projection == "down_proj" else (8, 3)
                    yield (
                        0,
                        expert,
                        f"model.layers.0.mlp.experts.{expert}.{projection}.weight",
                        torch.zeros(shape, dtype=torch.bfloat16),
                    )

    def capture():
        packing_peak = 0
        for encoder in encoders:
            encoder.begin()
        with closing(iter_owner_expert_blocks(converted(), layout, 0, 2, budget, bucket_bytes=144)) as blocks:
            for name, slab in blocks:
                for rank, encoder in enumerate(encoders):
                    chunk = expert_slice_chunk([(name, slab)], layout, rank)
                    descriptor, shard = chunk[0]
                    packing_peak = max(packing_peak, slab.nbytes + shard.nbytes)
                    encoder.add(descriptor, shard)
                    del chunk, shard
                del slab
        return [bucket for encoder in encoders for bucket in encoder.finish()], packing_peak

    try:
        buckets, packing_peak = capture()
        # Measure actual first-capture snapshots, not the helper's formula.
        assert budget.used_bytes == sum(len(bucket.payload) for bucket in buckets) == 576
        estimate = steady_state_snapshot_bytes(buckets, packing_bytes=packing_peak)
        assert (estimate <= budget.limit_bytes) == (limit == 3000)
        # Deliberately bypass admission here to reproduce the production
        # failure it prevents: a valid FULL baseline, then predictable OOM.
        for encoder in encoders:
            encoder.prepare_commit({"worker"}, {"worker"})
            encoder.commit()
        if limit < estimate:
            with pytest.raises(ValueError, match="shared rank budget"):
                capture()
        else:
            next_buckets, next_packing_peak = capture()
            assert all(bucket.changed == 0 for bucket in next_buckets)
            assert steady_state_snapshot_bytes(next_buckets, packing_bytes=next_packing_peak) == estimate
            assert budget.peak_bytes <= estimate
    finally:
        for encoder in encoders:
            encoder.close()
    assert budget.used_bytes == 0
