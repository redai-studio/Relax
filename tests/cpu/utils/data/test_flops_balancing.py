# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import pickle
from types import SimpleNamespace

import pytest

from relax.utils.data.flops_balancing import (
    FlopsWorkload,
    _refine_capacity_packing,
    _refine_rank_assignment,
    build_flops_schedule,
    load_flops_workload,
    validate_flops_balancing,
)
from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions
from tests.cpu.utils.test_flops_counter import FLOPS_TEST_CONFIGS, FLOPS_TEST_VL_CONFIGS, Config


def _assert_schedule_contract(
    schedule: list[list[list[int]]],
    dummy: list[set[int]],
    lengths: list[int],
    capacity: int,
    num_microbatches: int,
) -> None:
    assert len(schedule) == 4
    assert {len(rank) for rank in schedule} == {num_microbatches}
    consumed = []
    for rank, dummy_indices in zip(schedule, dummy, strict=True):
        assert dummy_indices <= set(range(len(rank)))
        real = [pack for offset, pack in enumerate(rank) if offset not in dummy_indices]
        assert real
        assert all(rank[offset] in real for offset in dummy_indices)
        for pack in rank:
            assert pack
            assert sum(lengths[index] for index in pack) <= capacity or len(pack) == 1
        consumed.extend(index for pack in real for index in pack)
    assert sorted(consumed) == list(range(len(lengths)))


@pytest.mark.parametrize(
    "num_samples,microbatch_group_size,num_microbatches,expected_load",
    [(20, 1, 2, 5.0), (36, 4, 4, 9.0)],
    ids=["dp-alignment", "vpp-alignment"],
)
def test_flops_schedule_alignment_balances_equal_cost_samples(
    num_samples: int, microbatch_group_size: int, num_microbatches: int, expected_load: float
) -> None:
    lengths = [1024] * num_samples
    costs = [1.0] * num_samples
    schedule, dummy = build_flops_schedule(lengths, costs, 4096, 4, microbatch_group_size)

    _assert_schedule_contract(schedule, dummy, lengths, 4096, num_microbatches)
    assert not any(dummy)
    loads = [sum(costs[index] for pack in rank for index in pack) for rank in schedule]
    assert loads == [expected_load] * 4


@pytest.mark.parametrize(
    "lengths,capacity,microbatch_group_size,num_microbatches,num_dummy",
    [
        ([1, 4, 4, 6, 3, 6, 6], 10, 1, 1, 0),
        ([9] + [4] * 8, 8, 1, 2, 0),
        ([8, 4, 2, 1], 8, 2, 2, 4),
    ],
    ids=["capacity-fallback", "oversized-singleton", "dummy-padding"],
)
def test_flops_schedule_alignment_preserves_capacity_and_padding(
    lengths: list[int], capacity: int, microbatch_group_size: int, num_microbatches: int, num_dummy: int
) -> None:
    costs = [float(length * length + length) for length in lengths]
    schedule, dummy = build_flops_schedule(lengths, costs, capacity, 4, microbatch_group_size)

    _assert_schedule_contract(schedule, dummy, lengths, capacity, num_microbatches)
    assert sum(len(indices) for indices in dummy) == num_dummy


def test_flops_schedule_rebalances_aligned_binpack_fallback() -> None:
    # Three balanced packs overflow the capacity, and bin-packing already
    # returns a DP-aligned count of four: [8], [8], [3, 6], [3]. Keeping those
    # packs gives the ranks 16 and 12; rebalancing at four packs gives 14 each.
    lengths = [3, 8, 3, 8, 6]
    costs = [float(length) for length in lengths]
    schedule, dummy = build_flops_schedule(lengths, costs, 10, 2)

    assert not any(dummy)
    assert all(sum(lengths[index] for index in pack) <= 10 for rank in schedule for pack in rank)
    assert [sum(costs[index] for pack in rank for index in pack) for rank in schedule] == [14.0, 14.0]


def test_flops_schedule_refines_capacity_fallback_without_more_microbatches() -> None:
    lengths = [1, 2, 7, 8, 2, 3, 3, 8]
    costs = [float(length * length + length) for length in lengths]
    schedule, dummy = build_flops_schedule(lengths, costs, 10, 2)

    assert not any(dummy)
    assert [len(rank) for rank in schedule] == [2, 2]
    assert sorted(i for rank in schedule for group in rank for i in group) == list(range(len(lengths)))
    assert all(sum(lengths[i] for i in group) <= 10 for rank in schedule for group in rank)
    # The token-only fallback gives 140/98. Two length-8 samples and one
    # length-7 sample force one rank to carry at least 72 + 56 = 128.
    assert max(sum(costs[i] for group in rank for i in group) for rank in schedule) == 128
    assert build_flops_schedule(lengths, costs, 10, 2) == (schedule, dummy)


def test_flops_capacity_refinement_moves_samples_without_emptying_packs() -> None:
    lengths = [6, 3, 2, 4, 1, 1]
    costs = [float(length * length + length) for length in lengths]
    bins = [[0, 1], [2], [3], [4, 5]]
    owners = [[0, 1], [2, 3]]

    _refine_capacity_packing(bins, owners, lengths, costs, 10)

    assert len(bins) == 4 and all(bins)
    assert sorted(i for group in bins for i in group) == list(range(len(lengths)))
    assert all(sum(lengths[i] for i in group) <= 10 for group in bins)
    # Cost 42 for the longest sample plus at least 2 for its rank's other pack.
    assert max(sum(costs[i] for index in rank for i in bins[index]) for rank in owners) == 44


def test_flops_rank_refinement_improves_equal_count_kk_assignment() -> None:
    weights = [16.0, 44.0, 83.0, 29.0, 38.0, 94.0, 81.0, 50.0, 30.0, 28.0, 64.0, 84.0]
    owners = get_seqlen_balanced_partitions(weights, 4, equal_size=True)
    assert max(sum(weights[i] for i in rank) for rank in owners) == 162

    _refine_rank_assignment(weights, owners)

    assert [len(rank) for rank in owners] == [3] * 4
    assert sorted(i for rank in owners for i in rank) == list(range(len(weights)))
    # Integral costs summing to 641 across four ranks require max load >= 161.
    assert max(sum(weights[i] for i in rank) for rank in owners) == 161


def test_flops_rank_refinement_accounts_for_local_dummy_execution() -> None:
    lengths = [8, 7, 5]
    costs = [8.0, 7.0, 5.0]
    schedule, dummy = build_flops_schedule(lengths, costs, 8, 2, microbatch_group_size=2)

    assert [len(rank) for rank in schedule] == [2, 2]
    assert sum(map(len, dummy)) == 1
    real = [
        i
        for rank, offsets in zip(schedule, dummy, strict=True)
        for j, pack in enumerate(rank)
        if j not in offsets
        for i in pack
    ]
    assert sorted(real) == [0, 1, 2]
    assert all(
        rank[j] in [pack for k, pack in enumerate(rank) if k not in offsets]
        for rank, offsets in zip(schedule, dummy, strict=True)
        for j in offsets
    )
    # KK's useful loads 8/12 execute as 16/12 after duplicating the local 8.
    # Swapping 8 and 7 gives executed loads 14/13, despite useful loads 7/13.
    assert sorted(sum(costs[i] for pack in rank for i in pack) for rank in schedule) == [13, 14]
    assert build_flops_schedule(lengths, costs, 8, 2, 2) == (schedule, dummy)


@pytest.mark.parametrize("name", ["qwen3_dense", "qwen3_moe"])
def test_flops_workload_matches_existing_model_fixtures_without_device_probe(name, monkeypatch):
    from relax.utils.training import flops_counter

    monkeypatch.setattr(flops_counter, "get_device_peak_flops", lambda **_: pytest.fail("GPU peak probe"))
    fixture = FLOPS_TEST_CONFIGS[name]
    workload = FlopsWorkload.from_config(Config(fixture["config"]))
    for lengths, expected in zip(fixture["batch_seqlens_list"], fixture["expected_tflops_list"], strict=True):
        assert sum(workload(lengths)) / 1e12 == pytest.approx(expected)
    assert pickle.loads(pickle.dumps(workload))([7, 13]) == workload([7, 13])


def test_flops_workload_dense_and_active_expert_cost():
    config = dict(
        model_type="qwen3",
        hidden_size=16,
        vocab_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=48,
    )
    lengths = [1, 4, 19]
    assert FlopsWorkload.from_config(config)(lengths) == pytest.approx([43008 * n + 192 * n * n for n in lengths])
    config.update(model_type="qwen3_moe", num_experts=4, num_experts_per_tok=2, moe_intermediate_size=8)
    assert FlopsWorkload.from_config(config)(lengths) == pytest.approx([25344 * n + 192 * n * n for n in lengths])
    config.update(model_type="qwen2_moe", shared_expert_intermediate_size=4)
    assert FlopsWorkload.from_config(config)(lengths) == pytest.approx([27840 * n + 192 * n * n for n in lengths])


def test_flops_workload_hybrid_quadratic_term_only_counts_full_attention():
    config = copy.deepcopy(FLOPS_TEST_VL_CONFIGS["qwen3_5_moe"]["config"])
    text = config["text_config"]
    text["layer_types"] = ["linear_attention"] * text["num_hidden_layers"]
    workload = FlopsWorkload.from_config(config)
    assert workload([2048])[0] == pytest.approx(2 * workload([1024])[0])
    text["layer_types"][0] = "full_attention"
    workload = FlopsWorkload.from_config(config)
    assert workload([2048])[0] > 2 * workload([1024])[0]
    previous = workload([1024])[0]
    text["shared_expert_intermediate_size"] *= 2
    assert FlopsWorkload.from_config(config)([1024])[0] > previous


@pytest.mark.parametrize("length", [0, -1, 1.5, float("nan"), True])
def test_flops_workload_rejects_invalid_lengths(length):
    workload = FlopsWorkload.from_config(FLOPS_TEST_CONFIGS["qwen3_dense"]["config"])
    with pytest.raises(ValueError, match="positive integer"):
        workload([length])


def test_flops_workload_rejects_unknown_or_incomplete_config():
    with pytest.raises(ValueError, match="model_type"):
        FlopsWorkload.from_config({"model_type": "unknown_moe"})
    with pytest.raises(ValueError, match="Invalid qwen3"):
        FlopsWorkload.from_config({"model_type": "qwen3"})


def test_flops_workload_reads_s3_config_without_loading_weights(monkeypatch):
    from relax.utils import s3_model_loader

    config = FLOPS_TEST_VL_CONFIGS["qwen3_5"]["config"]
    monkeypatch.setattr(s3_model_loader, "read_s3_model_config", lambda args: config)
    args = SimpleNamespace(model_source=SimpleNamespace(uri="s3://fixture/model"))
    assert load_flops_workload(args)([13]) == FlopsWorkload.from_config(config)([13])


def test_flops_validation_off_is_noop_and_on_enables_dp_balance():
    args = SimpleNamespace(balance_by_flops=False)
    validate_flops_balancing(args)
    assert vars(args) == {"balance_by_flops": False}
    args = _args()
    validate_flops_balancing(args)
    assert args.balance_data


def _args(**overrides):
    values = dict(
        balance_by_flops=True,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=8192,
        fully_async=False,
        hybrid=False,
        sft_packing_strategy="balanced",
        loss_type="sft",
        n_samples_per_prompt=1,
        balance_data=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "override",
    [
        {"use_dynamic_batch_size": False},
        {"max_tokens_per_gpu": 0},
        {"fully_async": True},
        {"hybrid": True},
        {"sft_wholepack": True},
        {"sft_async_prepack": True},
        {"sft_packing_strategy": "binpack"},
        {"loss_type": "dpo"},
        {"n_samples_per_prompt": 2},
        {"mtp_only_training": True},
        {"num_experts": 4, "moe_aux_loss_coeff": [0.0, 0.01]},
        {"num_experts": 4, "moe_expert_capacity_factor": 1.0},
        {"num_experts": 4, "moe_router_enable_expert_bias": True},
        {"num_experts": None, "moe_aux_loss_coeff": 0.01},
        {"moe_z_loss_coeff": 0.01},
        {"moe_router_load_balancing_type": "sinkhorn", "moe_aux_loss_coeff": 0.0},
        {"moe_router_load_balancing_type": ["sinkhorn"], "moe_aux_loss_coeff": [0.0]},
    ],
)
def test_flops_validation_rejects_unsupported_paths(override):
    with pytest.raises(ValueError, match="--balance-by-flops"):
        validate_flops_balancing(_args(**override))
