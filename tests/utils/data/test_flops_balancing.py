# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only scheduling invariants, including capacity and synchronized
padding."""

import ast
import itertools
import json
import random
from functools import lru_cache
from pathlib import Path
from unittest.mock import Mock

import pytest

from relax.utils.data.flops_balancing import (
    balance_workloads,
    expand_partitions,
    guarded_dp_assignment,
    partition_by_flops,
    partition_objective,
    summarize_partitions,
)
from relax.utils.data.packing import get_microbatch_count
from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions


@lru_cache(maxsize=1)
def baseline_helper():
    path = Path(__file__).resolve().parents[3] / "relax/backends/megatron/data.py"
    names = {
        "_get_first_fit_partitions",
        "_partitions_fit_capacity_or_singleton_oversize",
        "_get_capacity_safe_balanced_partitions",
        "_get_seqlen_partitions_with_dummy_padding",
    }
    nodes = [node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) in names]
    namespace = dict(logger=Mock(), get_seqlen_balanced_partitions=get_seqlen_balanced_partitions)
    unit = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], *nodes], type_ignores=[])
    exec(compile(unit, str(path), "exec"), namespace)
    return namespace["_get_seqlen_partitions_with_dummy_padding"]


def baseline_plan(lengths, capacity, count=None):
    if not lengths:
        return []
    count = get_microbatch_count(list(lengths), capacity) if count is None else count
    return baseline_helper()(list(lengths), count, capacity)[0]


def assert_plan(lengths, costs, capacity, partitions, baseline):
    assert len(partitions) == len(baseline)
    assert sorted(i for part in partitions for i in part) == list(range(len(lengths)))
    for part in partitions:
        assert part
        total = sum(lengths[i] for i in part)
        assert total <= capacity or (len(part) == 1 and lengths[part[0]] > capacity)
    objective = lambda plan: partition_objective([sum(costs[i] for i in part) for part in plan])
    assert objective(partitions) <= objective(baseline)
    if objective(partitions) == objective(baseline):
        assert partitions == baseline
    assert partitions == partition_by_flops(lengths, costs, capacity, baseline)


@pytest.mark.parametrize(
    "lengths,capacity",
    [
        ([], 10),
        ([3], 10),
        ([10], 10),
        ([11], 10),
        ([6, 4], 10),
        ([6, 5], 10),
        ([11, 2, 8, 25], 10),
        ([0] * 9, 10),
        ([5] * 16, 10),
        ([1, 1, 1, 9, 9], 10),
    ],
)
def test_partition_edges_preserve_capacity_and_coverage(lengths, capacity):
    costs = [length**2 for length in lengths]
    baseline = baseline_plan(lengths, capacity)
    assert_plan(lengths, costs, capacity, partition_by_flops(lengths, costs, capacity, baseline), baseline)


def test_equal_work_ties_keep_baseline():
    lengths = [2, 4, 4, 2]
    baseline = [[3, 2], [1, 0]]
    assert partition_by_flops(lengths, [1] * 4, 6, baseline) == baseline


def test_old_greedy_inflation_regression_keeps_baseline_K():
    # V1 unrestricted greedy needed K=4 although existing first-fit fits K=3.
    lengths, costs = [10, 5, 1, 4, 10], [32, 2, 19, 77, 85]
    baseline = baseline_plan(lengths, 10)
    assert len(baseline) == 3
    result = partition_by_flops(lengths, costs, 10, baseline)
    assert_plan(lengths, costs, 10, result, baseline)


def test_profitable_move_is_executed():
    baseline = [[0, 1], [2, 3]]
    stats = {}
    result = partition_by_flops([1] * 4, [8, 4, 3, 1], 3, baseline, stats=stats)
    assert [sum([8, 4, 3, 1][i] for i in part) for part in result] == [8, 8]
    assert stats["moves"] == 1 and stats["swaps"] == 0
    assert baseline == [[0, 1], [2, 3]]


def test_profitable_swap_when_no_move_fits():
    stats = {}
    result = partition_by_flops([1] * 4, [8, 7, 2, 1], 2, [[0, 1], [2, 3]], stats=stats)
    assert [sum([8, 7, 2, 1][i] for i in part) for part in result] == [9, 9]
    assert stats["moves"] == 0 and stats["swaps"] == 1


def test_no_profitable_operation_preserves_exact_baseline_order():
    baseline = [[1, 0], [3, 2]]
    stats = {}
    assert partition_by_flops([1] * 4, [1] * 4, 2, baseline, stats=stats) == baseline
    assert stats["improved"] == stats["moves"] == stats["swaps"] == 0


def test_oversized_singleton_is_frozen():
    baseline = [[0], [1, 2], [3]]
    result = partition_by_flops([20, 2, 2, 2], [1, 80, 10, 10], 5, baseline)
    assert result[0] == [0]
    assert_plan([20, 2, 2, 2], [1, 80, 10, 10], 5, result, baseline)


def test_explicit_deterministic_search_budget():
    lengths = [1] * 128
    costs = list(range(128))
    baseline = [list(range(i, i + 8)) for i in range(0, 128, 8)]
    stats = {}
    result = partition_by_flops(lengths, costs, 12, baseline, stats=stats)
    assert stats["iterations"] <= 32
    assert stats["moves"] + stats["swaps"] <= 32
    assert stats["candidate_checks"] <= 32 * (4 * 8 + 4 * 8 * 8)
    assert_plan(lengths, costs, 12, result, baseline)


def test_exhaustive_small_partitions():
    for n in range(5):
        for lengths in itertools.product([0, 2, 4, 7], repeat=n):
            costs = [length**2 for length in lengths]
            baseline = baseline_plan(lengths, 5)
            assert_plan(lengths, costs, 5, partition_by_flops(lengths, costs, 5, baseline), baseline)


def test_randomized_heterogeneous_and_zero_cost_plans():
    rng = random.Random(16)
    for _ in range(120):
        lengths = [rng.randrange(31) for _ in range(rng.randrange(1, 50))]
        costs = [rng.choice([0, length, length**2, length * 1000]) for length in lengths]
        baseline = baseline_plan(lengths, 20)
        assert_plan(lengths, costs, 20, partition_by_flops(lengths, costs, 20, baseline), baseline)


def test_shared_count_matches_existing_helper():
    # Import only the actual function AST: data.py imports optional dataset and
    # torch dependencies which are irrelevant to the CPU count contract.
    path = Path(__file__).resolve().parents[3] / "relax/utils/data/data.py"
    node = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == "get_first_fit_partitions")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    rng = random.Random(24)
    for _ in range(100):
        lengths = [rng.randrange(30) for _ in range(rng.randrange(40))]
        assert get_microbatch_count(lengths, 10) == len(namespace["get_first_fit_partitions"](lengths, 10))


@pytest.mark.parametrize("target", [2, 3, 4, 5, 8])
def test_expand_preserves_feasible_seed_and_marks_only_dummy(target):
    lengths, costs = [6, 4, 6, 4], [36, 16, 36, 16]
    original = [[0, 1], [2, 3]]
    result, dummy = expand_partitions(original, lengths, costs, target)
    assert original == [[0, 1], [2, 3]]
    assert len(result) == target
    assert dummy == set(range(4, target))
    real = [part for b, part in enumerate(result) if b not in dummy]
    assert sorted(i for part in real for i in part) == list(range(4))
    assert all(sum(lengths[i] for i in part) <= 10 for part in result)
    for part in real:
        assert any(set(part) <= set(seed) for seed in original)
    assert (result, dummy) == expand_partitions(original, lengths, costs, target)


def test_dummy_uses_smallest_real_token_partition():
    bins, dummy = expand_partitions([[0], [1], [2]], [9, 1, 3], [81, 1, 9], 5)
    assert bins == [[0], [1], [2], [1], [1]]
    assert dummy == {3, 4}


def test_empty_expand():
    assert expand_partitions([], [], [], 0) == ([], set())
    with pytest.raises(ValueError, match="without a real sample"):
        expand_partitions([], [], [], 1)


@pytest.mark.parametrize("bins,target", [([[0, 1], [2]], 1), ([[0, 0], [2]], 3), ([[0], []], 3)])
def test_expand_rejects_shrink_missing_duplicate_or_empty(bins, target):
    with pytest.raises(ValueError):
        expand_partitions(bins, [2, 3, 4], [4, 9, 16], target)


@pytest.mark.parametrize("equal_size", [True, False])
def test_dp_workload_assignment_coverage_and_determinism(equal_size):
    costs = [50, 10, 100, 8, 2, 0, 9, 15, 12, 13, 40, 10]
    result = balance_workloads(costs, 3, equal_size=equal_size)
    assert sorted(i for part in result for i in part) == list(range(len(costs)))
    if equal_size:
        assert [len(part) for part in result] == [4, 4, 4]
    assert result == balance_workloads(costs, 3, equal_size=equal_size)


def test_dp_empty_and_invalid_counts():
    assert balance_workloads([], 2) == [[], []]
    for costs, ranks in [([1, 2], 3), ([1, 2, 3], 2), ([1], 0)]:
        with pytest.raises(ValueError):
            balance_workloads(costs, ranks)


def test_synthetic_workload_balance_improves_over_token_baseline():
    # Quadratic work is a mechanism example, not a measured training speedup.
    lengths = [20, 21, 23, 10, 8, 10, 12, 9, 14, 3, 12, 16]
    costs = [length**2 for length in lengths]
    baseline = summarize_partitions(lengths, costs, get_seqlen_balanced_partitions(lengths, 3, True))
    balanced = summarize_partitions(lengths, costs, balance_workloads(costs, 3))
    assert baseline["capacity_tokens"] == [54, 52, 52]
    assert baseline["estimated_flops"] == [804, 766, 894]
    assert balanced["estimated_flops"] == [826, 820, 818]
    assert balanced["max_flops"] - balanced["min_flops"] < (baseline["max_flops"] - baseline["min_flops"]) / 10
    assert balanced["imbalance_ratio"] < baseline["imbalance_ratio"]
    json.dumps({"baseline": baseline, "balanced": balanced}, allow_nan=False)


def test_summary_zero_work():
    summary = summarize_partitions([0], [0], [[0]])
    assert summary["imbalance_ratio"] == 1.0
    assert summary["total_flops"] == 0
    assert summarize_partitions([], [], [])["estimated_flops"] == []


@pytest.mark.parametrize("costs", [[-1], [float("nan")], [float("inf")]])
def test_nonfinite_or_negative_work_rejected(costs):
    with pytest.raises(ValueError, match="finite and non-negative"):
        partition_by_flops([1], costs, 10, [[0]])


@pytest.mark.parametrize("lengths,costs,capacity", [([1], [], 10), ([-1], [1], 10), ([1], [1], 0)])
def test_invalid_partition_inputs(lengths, costs, capacity):
    with pytest.raises(ValueError):
        partition_by_flops(lengths, costs, capacity, [[0]])


def test_microbatch_workload_objective_improves_over_token_partition():
    lengths = [18, 3, 19, 14, 2, 1, 3, 7, 8, 17, 20, 1]
    costs = [length**2 for length in lengths]
    baseline = summarize_partitions(lengths, costs, get_seqlen_balanced_partitions(lengths, 3, False))
    seed = get_seqlen_balanced_partitions(lengths, 3, False)
    bins = partition_by_flops(lengths, costs, 100, seed)
    result = summarize_partitions(lengths, costs, bins)
    assert result["max_flops"] - result["min_flops"] < (baseline["max_flops"] - baseline["min_flops"]) / 3
    assert all(sum(lengths[i] for i in part) <= 100 for part in bins)
    assert sorted(i for part in bins for i in part) == list(range(len(lengths)))


def test_uniform_forward_backward_workload_scale_keeps_plan():
    lengths = [18, 3, 19, 14, 2, 1, 3, 7, 8, 17, 20, 1]
    costs = [length**2 + 10 * length for length in lengths]
    seed = baseline_plan(lengths, 40)
    assert partition_by_flops(lengths, costs, 40, seed) == partition_by_flops(
        lengths, [3 * cost for cost in costs], 40, seed
    )


@pytest.mark.parametrize(
    "lengths,decision",
    [
        ([2, 6, 8, 7, 9, 3, 5, 1], "accepted"),
        ([1, 1, 4, 7, 5, 9, 7, 2], "rejected_by_k"),
        ([6, 8, 8, 5, 7, 4, 8, 1], "rejected_no_improvement"),
    ],
)
def test_dp_guard_acceptance_and_fallback(lengths, decision):
    costs = [length**2 for length in lengths]
    baseline = get_seqlen_balanced_partitions(lengths, 2, True)
    original = [list(part) for part in baseline]
    result, diagnostics = guarded_dp_assignment(lengths, costs, baseline, 10)
    assert baseline == original
    assert diagnostics["decision"] == decision
    counts = diagnostics["counts"]["raw"]
    if decision == "accepted":
        assert max(counts["candidate"]) <= max(counts["baseline"])
        assert diagnostics["after"]["max_flops"] < diagnostics["before"]["max_flops"]
    else:
        assert result == baseline
        assert diagnostics["after"] == diagnostics["before"]
    if decision == "rejected_by_k":
        assert max(counts["candidate"]) > max(counts["baseline"])
        assert diagnostics["candidate"]["max_flops"] < diagnostics["before"]["max_flops"]
    assert (result, diagnostics) == guarded_dp_assignment(lengths, costs, baseline, 10)


def test_identical_dp_candidate_keeps_exact_baseline():
    baseline = balance_workloads([1] * 4, 2)
    result, diagnostics = guarded_dp_assignment([1] * 4, [1] * 4, baseline, 10)
    assert result == baseline
    assert diagnostics["decision"] == "rejected_no_improvement"


def test_cp_guard_checks_raw_and_worker_padded_layout():
    lengths = [9, 13, 9, 3, 7, 10, 6, 12, 5, 23, 22, 21]
    costs = [length**2 for length in lengths]
    baseline = get_seqlen_balanced_partitions(lengths, 3, True)
    _, raw = guarded_dp_assignment(lengths, costs, baseline, 32)
    assert raw["decision"] == "accepted"
    result, padded = guarded_dp_assignment(lengths, costs, baseline, 32, capacity_alignment=8)
    assert padded["decision"] == "rejected_by_k"
    assert result == baseline
    assert padded["counts"]["raw"] == raw["counts"]["raw"]
    assert padded["counts"]["padded"] == {"baseline": [2, 2, 2], "candidate": [3, 2, 2]}


@pytest.mark.parametrize("strategy", ["balanced", "binpack"])
def test_dp_guard_uses_actual_strategy_count(strategy):
    lengths = [4, 3, 5, 5, 11, 14, 12, 14, 14, 15, 6, 2]
    costs = [length**2 for length in lengths]
    baseline = get_seqlen_balanced_partitions(lengths, 3, True)
    _, diagnostics = guarded_dp_assignment(lengths, costs, baseline, 20, packing_strategy=strategy)
    candidate = balance_workloads(costs, 3)
    for label, groups in (("baseline", baseline), ("candidate", candidate)):
        assert diagnostics["counts"]["raw"][label] == [
            get_microbatch_count([lengths[i] for i in rank], 20, strategy) for rank in groups
        ]


@pytest.mark.parametrize("baseline", [[[0, 1], [1, 2]], [[0], []], [[0, 1, 2]]])
def test_repair_rejects_invalid_or_infeasible_seed(baseline):
    with pytest.raises(ValueError, match="baseline partitions"):
        partition_by_flops([3, 3, 3], [9, 9, 9], 5, baseline)
