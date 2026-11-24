# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections import Counter

import pytest
import torch

from relax.engine.sft.image import ImageBatchSampler, assign_image_batch, image_tier_budget
from relax.engine.sft.image_prefetch import ImagePrefetch


@pytest.mark.parametrize("tiers", [None, [4], [9, 4], [4, 9, 16]])
def test_tier_budget_matches_existing_policy_without_global_rng(tiers):
    config = {"fixed_pixels_tiers": tiers, "seed": 42}
    state = torch.get_rng_state().clone()
    for index in range(32):
        expected = None
        if tiers:
            generator = torch.Generator().manual_seed(42 + index)
            expected = (
                tiers[torch.randint(len(tiers), (1,), generator=generator).item()] if len(tiers) > 1 else tiers[0]
            )
        assert image_tier_budget(config, index) == expected
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("size,world,batch", [(19, 4, 8), (7, 8, 16), (40, 16, 32), (7, 32, 32), (40, 32, 32)])
def test_tier_assignment_preserves_membership_and_reduces_micro_cost(size, world, batch):
    config = {"fixed_pixels_tiers": [4, 9, 16], "seed": 42, "dp_tier_micro_balance": True}
    sampler = ImageBatchSampler(size, batch, 42)
    resumed = ImageBatchSampler(size, batch, 42)
    for step in range(8):
        ids = sampler.batch(step)
        ordered = assign_image_batch(ids, config)
        assert Counter(ordered) == Counter(ids)
        assert ordered == assign_image_batch(resumed.batch(step), config)
        assert all(len(ordered[rank::world]) == batch // world for rank in range(world))
        costs = [image_tier_budget(config, index) for index in ordered]
        assert costs == sorted(costs, reverse=True)
        old_costs = [image_tier_budget(config, index) for index in ids]
        proxy = lambda values: sum(max(values[start : start + world]) for start in range(0, batch, world))
        assert proxy(costs) <= proxy(old_costs)
        for tier in config["fixed_pixels_tiers"]:
            assert [index for index in ordered if image_tier_budget(config, index) == tier] == [
                index for index in ids if image_tier_budget(config, index) == tier
            ]


def test_disabled_tier_assignment_keeps_canonical_order():
    ids = [4, 3, 1, 4]
    assert assign_image_batch(ids, {}) == ids


@pytest.mark.parametrize("world,rank", [(4, rank) for rank in range(4)] + [(32, rank) for rank in (0, 15, 16, 31)])
@pytest.mark.parametrize("rotation", [False, True])
def test_balanced_prefetch_matches_training_assignment_across_resume(world, rank, rotation):
    class Dataset:
        config = {
            "fixed_pixels_tiers": [4, 9],
            "seed": 42,
            "dp_tier_micro_balance": True,
            "dp_tier_rank_rotation": rotation,
        }

        def __getitem__(self, index):
            return index

    dataset = Dataset()
    sampler = ImageBatchSampler(19, 8 if world == 4 else 32, 42)
    prefetch = ImagePrefetch(dataset, sampler, rank, world, 2, 9, depth=3)
    try:
        for step in range(2, 9):
            ids = assign_image_batch(sampler.batch(step), dataset.config, step=step, world=world)[rank::world]
            for micro, index in enumerate(ids):
                assert prefetch.get(step, micro, index) == dataset[index]
                assert len(prefetch.pending) <= 3
    finally:
        prefetch.close()


@pytest.mark.parametrize("size,world,batch", [(19, 4, 8), (7, 8, 16), (40, 16, 32), (7, 32, 32), (40, 32, 32)])
def test_rank_rotation_preserves_each_micro_block_across_epochs_and_resume(size, world, batch):
    config = {"fixed_pixels_tiers": [4, 9, 16], "seed": 42, "dp_tier_micro_balance": True}
    rotated_config = {**config, "dp_tier_rank_rotation": True}
    sampler, resumed = ImageBatchSampler(size, batch, 42), ImageBatchSampler(size, batch, 42)
    rng = torch.get_rng_state().clone()
    for step in range(32):
        ids = sampler.batch(step)
        ordered = assign_image_batch(ids, config)
        rotated = assign_image_batch(ids, rotated_config, step=step, world=world)
        assert rotated == assign_image_batch(resumed.batch(step), rotated_config, step=step, world=world)
        for start in range(0, batch, world):
            assert Counter(rotated[start : start + world]) == Counter(ordered[start : start + world])
        assert Counter(rotated) == Counter(ids)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("world", [16, 32])
def test_rank_rotation_visits_every_rank_over_one_period(world):
    config = {"fixed_pixels_tiers": [4, 9], "seed": 42, "dp_tier_micro_balance": True, "dp_tier_rank_rotation": True}
    ids = list(range(32))
    batches = [assign_image_batch(ids, config, step=step, world=world) for step in range(world)]
    for index in ids:
        assert {batch.index(index) % world for batch in batches} == set(range(world))
    assert assign_image_batch(ids, config, step=world, world=world) == batches[0]


@pytest.mark.parametrize("step,world", [(None, 4), (-1, 4), (0, None), (0, 0), (True, 4), (0, 3)])
def test_rank_rotation_rejects_invalid_micro_bounds(step, world):
    config = {"fixed_pixels_tiers": [4, 9], "dp_tier_micro_balance": True, "dp_tier_rank_rotation": True}
    with pytest.raises(ValueError, match="world-sized"):
        assign_image_batch(list(range(8)), config, step=step, world=world)


def test_rank_rotation_requires_balancing():
    with pytest.raises(ValueError, match="requires micro balancing"):
        assign_image_batch([0, 1], {"dp_tier_rank_rotation": True}, step=0, world=2)
