# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Owner packing freezes reusable canonical values and preserves TP slices."""

from contextlib import closing

import pytest
import torch

from relax.backends.megatron.weight_update.expert_layout import DirectExpertLayout
from relax.backends.megatron.weight_update.owner_expert_blocks import iter_owner_expert_blocks
from relax.distributed.weight_sync.streaming_delta import SnapshotBudget


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
            assert 0 < budget.used_bytes <= 192 << 20
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
