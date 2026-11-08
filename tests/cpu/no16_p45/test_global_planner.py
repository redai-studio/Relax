# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import pytest

from relax.utils.data.global_flops_pack_planner import ExecutionContract, build_global_plan
from relax.utils.data.global_flops_pack_protocol import cost_identity

from .support import setup


def plan(lengths, costs, *, dp=2, capacity=3, **geometry):
    contract = ExecutionContract(dp=dp, tp=1, pp=1, cp=1, vpp=1, mb_group=1, max_tokens=capacity, **geometry)
    return build_global_plan(
        list(range(len(lengths))),
        lengths,
        contract.physical(lengths),
        costs,
        contract,
        identity=("p", 0, "sft_train", 0),
        logical_count=len(lengths),
        model_signature="model",
        cost_signature="cost",
        checkpoint_signature="checkpoint",
    )


def test_independent_p2_hand_oracle_freezes_mbs_before_dp(tmp_path):
    _, ctx, _, _, _ = setup(tmp_path)
    model, cost = cost_identity(ctx.estimator)
    # P2 tiny hybrid: 8136*L + 48*L^2. Whole MB-to-DP assigns 3 short rows together.
    result = plan([1, 1, 1, 2], [8184, 8184, 8184, 16464])
    assert tuple(mb.rows for mb in result.microbatches) == ((0, 1, 2), (3,))
    assert result.rank_rows(0) == (0, 1, 2) and result.rank_rows(1) == (3,)
    assert result.K == 1 and result.fallback_reason is None
    assert len(model) == len(cost) == 64


def test_global_fallback_uses_feasible_bin_count_not_total_token_lower_bound():
    result = plan([6, 6, 6], [1, 100, 1], capacity=10)
    assert len(result.microbatches) == 3  # ceil(18/10)=2 is infeasible.
    assert sorted(row for mb in result.microbatches for row in mb.rows) == [0, 1, 2]
    assert all(mb.physical_tokens <= 10 for mb in result.microbatches)
    assert sorted(map(len, result.rank_mbs)) == [1, 2]
    assert result.K == 2 and sum(dummy for rank in range(2) for _, dummy in result.rank_slots(rank)) == 1


def test_unsafe_weighted_candidate_falls_back_globally_before_mb_assignment():
    result = plan([5, 5, 5, 5], [100, 1, 1, 1], capacity=10)
    assert result.fallback_reason == "weighted_kk_physical_capacity"
    assert all(mb.physical_tokens <= 10 for mb in result.microbatches)
    assert all(len(mb.rows) == 2 for mb in result.microbatches)


def test_determinism_and_cost_scaling_do_not_change_frozen_membership():
    first = plan([1, 1, 1, 2], [8184, 8184, 8184, 16464])
    repeated = plan([1, 1, 1, 2], [8184, 8184, 8184, 16464])
    assert first.plan_digest == repeated.plan_digest
    assert tuple(mb.rows for mb in first.microbatches) == tuple(mb.rows for mb in repeated.microbatches)
    scaled = plan([1, 1, 1, 2], [8184 * 3, 8184 * 3, 8184 * 3, 16464 * 3])
    assert tuple(mb.rows for mb in first.microbatches) == tuple(mb.rows for mb in scaled.microbatches)
    assert first.rank_mbs == scaled.rank_mbs


def test_static_cp_and_vpp_contract_capacity_and_dummy():
    contract = ExecutionContract(2, 2, 2, 2, 2, 4, 10, pad_multiplier=2, unsplit=True, allgather_cp=True)
    assert contract.capacity == 16 and contract.physical([5, 9]) == (8, 16)
    result = build_global_plan(
        [10, 20, 30],
        [5, 9, 5],
        [8, 16, 8],
        [1, 9, 1],
        contract,
        identity=("p", 0, "train", 0),
        logical_count=2,
        identities=[0, 0, 1],
        model_signature="m",
        cost_signature="c",
        checkpoint_signature="k",
    )
    assert result.K == 4
    for rank in range(2):
        assert len(result.rank_slots(rank)) == 4
        real_ids = {i for i, dummy in result.rank_slots(rank) if not dummy}
        assert all(i in real_ids for i, dummy in result.rank_slots(rank) if dummy)


def test_oversized_singleton_preserves_exception_without_claiming_memory_safety():
    result = plan([20, 1], [20, 1], capacity=5)
    assert sum(mb.oversized_singleton for mb in result.microbatches) == 1
    assert all(len(mb.rows) == 1 or mb.physical_tokens <= 5 for mb in result.microbatches)


@pytest.mark.parametrize("bad", [0, -1, True, float("inf"), float("nan"), "1"])
def test_bad_costs_reject_before_frozen_plan(bad):
    with pytest.raises(ValueError):
        plan([1, 1], [bad, 1])


def test_empty_real_rank_and_identity_row_miscount_are_rejected():
    with pytest.raises(ValueError, match="insufficient"):
        plan([1], [1])
    contract = ExecutionContract(2, 1, 1, 1, 1, 1, 10)
    with pytest.raises(ValueError, match="identities"):
        build_global_plan(
            [1, 2, 3],
            [1] * 3,
            [1] * 3,
            [1] * 3,
            contract,
            identity=("p", 0, "train", 0),
            logical_count=3,
            identities=[0, 0, 1],
            model_signature="m",
            cost_signature="c",
            checkpoint_signature="k",
        )


def test_sft_partial_logical_batch_is_rejected_before_plan_publication():
    contract = ExecutionContract(2, 1, 1, 1, 1, 1, 10)
    with pytest.raises(ValueError, match="complete logical batch"):
        build_global_plan(
            [1, 2],
            [1, 1],
            [1, 1],
            [1, 1],
            contract,
            identity=("p", 0, "sft_train", 0),
            logical_count=4,
            model_signature="m",
            cost_signature="c",
            checkpoint_signature="k",
        )
