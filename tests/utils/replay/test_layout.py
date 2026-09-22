# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the PR C distributed layout primitives (pure CPU)."""

from __future__ import annotations

import pytest
import torch

from relax.utils.replay.layout import (
    LayoutError,
    RankTopology,
    ShardSpec,
    TensorLayout,
    compute_offsets,
    cp_capability,
    reconstruct_shards,
    shard_order,
    validate_dp_completeness,
)
from relax.utils.replay.schema import StageCapability, StageId


def test_layout_parse_topology():
    topology = RankTopology.from_rank_dict({"dp": 2, "tp": 1, "pp": 2, "cp": 4})
    assert topology == RankTopology(dp=2, tp=1, pp=2, cp=4)
    assert topology.world_size == 16
    assert RankTopology.from_rank_dict({"cp": 1}).world_size == 1


def test_layout_shard_order():
    assert shard_order([3, 1, 2]) == [1, 2, 3]


def test_layout_compute_offsets():
    layout = compute_offsets("log_probs", lengths=[3, 5], dims=[8])
    assert layout.dims == (8,)
    assert layout.shards[0].offsets == (0,) and layout.shards[0].shape == (3,)
    assert layout.shards[1].offsets == (3,) and layout.shards[1].shape == (5,)


def test_layout_compute_offsets_padding():
    layout = compute_offsets("log_probs", lengths=[3, 5], dims=[8], pad_to=4)
    assert layout.shards[0].shape == (4,) and layout.shards[0].unpadded_length == 3
    assert layout.shards[1].shape == (8,) and layout.shards[1].unpadded_length == 5
    # Offsets are canonical (unpadded) coordinates: rank 1 starts after rank 0's
    # 3 real elements, regardless of rank 0's padding.
    assert layout.shards[1].offsets == (3,)


def test_layout_dp_completeness_ok():
    validate_dp_completeness(StageId.REWARD_POST_PROCESS, required_ranks=[0, 1], present_ranks=[1, 0])


def test_layout_dp_completeness_missing():
    with pytest.raises(LayoutError, match="missing DP rank shard"):
        validate_dp_completeness(StageId.ADVANTAGE_ESTIMATE, required_ranks=[0, 1, 2], present_ranks=[0, 2])


def test_layout_reconstruct_contiguous():
    canonical = torch.arange(8).float()
    layout = compute_offsets("advantages", lengths=[3, 5], dims=[8])
    shards = {0: canonical[:3], 1: canonical[3:]}
    assert torch.equal(reconstruct_shards(layout, shards), canonical)


def test_layout_reconstruct_with_padding():
    canonical = torch.arange(8).float()
    layout = compute_offsets("advantages", lengths=[3, 5], dims=[8], pad_to=4)
    # Rank 0 holds 3 real elements padded to 4; rank 1 holds 5 padded to 8.
    shard0 = torch.zeros(4)
    shard0[:3] = canonical[:3]
    shard1 = torch.zeros(8)
    shard1[:5] = canonical[3:]
    assert torch.equal(reconstruct_shards(layout, {0: shard0, 1: shard1}), canonical)


def test_layout_reconstruct_missing_shard():
    canonical = torch.arange(8).float()
    layout = compute_offsets("advantages", lengths=[3, 5], dims=[8])
    with pytest.raises(LayoutError, match="missing shard"):
        reconstruct_shards(layout, {0: canonical[:3]})


def test_layout_reconstruct_offsets_not_recorded():
    layout = TensorLayout(name="x", dims=(8,), shards={0: ShardSpec(0, None, (8,))})
    with pytest.raises(LayoutError, match="offsets not recorded"):
        reconstruct_shards(layout, {0: torch.arange(8).float()})


def test_layout_reconstruct_overlap():
    canonical = torch.arange(8).float()
    bad = TensorLayout(
        name="advantages",
        dims=(8,),
        shards={0: ShardSpec(0, (0,), (3,), 3), 1: ShardSpec(1, (2,), (5,), 5)},
    )
    with pytest.raises(LayoutError, match="gap or overlap"):
        reconstruct_shards(bad, {0: canonical[:3], 1: canonical[2:7]})


def test_layout_cp_capability_frozen():
    assert cp_capability(StageId.ADVANTAGE_ESTIMATE, cp=1, reconstructable=True) == StageCapability.RECOMPUTE
    assert cp_capability(StageId.ADVANTAGE_ESTIMATE, cp=1, reconstructable=False) == StageCapability.UNSUPPORTED
    # V1 freezes CP>1 as unsupported even when the layout is reconstructable.
    assert cp_capability(StageId.ADVANTAGE_ESTIMATE, cp=2, reconstructable=True) == StageCapability.UNSUPPORTED


def test_layout_json_roundtrip():
    layout = compute_offsets("log_probs", lengths=[3, 5], dims=[8], pad_to=4)
    restored = TensorLayout.from_dict(layout.to_dict())
    assert restored == layout
