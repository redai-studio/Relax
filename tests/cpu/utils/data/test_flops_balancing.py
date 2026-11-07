# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import pickle
import random
from types import SimpleNamespace

import pytest

from relax.utils.data.flops_balancing import (
    FlopsWorkload,
    align_flops_microbatches,
    load_flops_workload,
    plan_flops_microbatches,
    validate_flops_balancing,
)
from relax.utils.data.packing import get_first_fit_partitions
from tests.cpu.utils.test_flops_counter import FLOPS_TEST_CONFIGS, FLOPS_TEST_VL_CONFIGS, Config


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


@pytest.mark.parametrize("lengths", [[], [40, 30], [40, 3, 6], [4, 3, 5, 5, 11, 14, 12, 14, 14, 15, 6, 2]])
def test_flops_packing_capacity_coverage_and_dummy(lengths):
    workloads = [20 * length + length**2 for length in lengths]
    plan = plan_flops_microbatches(lengths, workloads, 20)
    assert len(plan.partitions) <= len(get_first_fit_partitions(lengths, 20))
    for count in {len(plan.partitions), len(lengths), len(lengths) + 2 if lengths else 0}:
        partitions, dummy = align_flops_microbatches(plan, count)
        assert len(partitions) == count
        assert sorted(i for j, group in enumerate(partitions) if j not in dummy for i in group) == list(
            range(len(lengths))
        )
        assert all(sum(lengths[i] for i in group) <= 20 or len(group) == 1 for group in partitions)
        assert all(partitions[j] in [g for i, g in enumerate(partitions) if i not in dummy] for j in dummy)


def test_flops_packing_randomized_capacity_and_alignment():
    rng = random.Random(16)
    for _ in range(50):
        lengths = [rng.randint(1, 130) for _ in range(rng.randint(1, 40))]
        workloads = [32 * length + length**2 for length in lengths]
        plan = plan_flops_microbatches(lengths, workloads, 100)
        count = (len(plan.partitions) + 3) // 4 * 4
        partitions, dummy = align_flops_microbatches(plan, count)
        assert (partitions, dummy) == align_flops_microbatches(plan, count)
        assert sorted(i for j, group in enumerate(partitions) if j not in dummy for i in group) == list(
            range(len(lengths))
        )
        assert all(sum(lengths[i] for i in group) <= 100 or len(group) == 1 for group in partitions)


def test_flops_packing_falls_back_to_feasible_count_before_alignment():
    # The token lower bound is two, but three six-token samples cannot fit
    # into two ten-token bins. Synchronizing the lower bound would deadlock.
    plan = plan_flops_microbatches([6, 6, 6], [42, 42, 42], 10)
    assert plan.partitions == [[0], [1], [2]]
    with pytest.raises(ValueError, match="Aligned"):
        align_flops_microbatches(plan, 2)
    groups, dummy = align_flops_microbatches(plan, 4)
    assert sorted(groups[:3]) == [[0], [1], [2]]
    assert dummy == {3}
    assert groups[3] == groups[0]


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
