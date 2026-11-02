# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Keep balanced packing capacity protection and dummy padding compatible."""

import ast
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def partitioner():
    root = Path(__file__).resolve().parents[3]
    from relax.utils.training.microbatch_cost import sample_costs, validate_microbatch_balance

    namespace = {
        "logger": Mock(),
        "sample_costs": sample_costs,
        "validate_microbatch_balance": validate_microbatch_balance,
    }
    names = {
        "get_first_fit_partitions",
        "get_binpack_partitions",
        "get_flops_packing_seed",
        "get_minimum_num_micro_batch_size",
        "_get_first_fit_partitions",
        "_partitions_fit_capacity_or_singleton_oversize",
        "_get_capacity_safe_balanced_partitions",
        "_get_seqlen_partitions_with_dummy_padding",
        "_get_synchronised_sample_costs",
    }
    for path in (
        "relax/utils/data/seqlen_balancing.py",
        "relax/utils/data/data.py",
        "relax/utils/data/packing.py",
        "relax/backends/megatron/data.py",
    ):
        nodes = ast.parse((root / path).read_text()).body
        if not path.endswith("seqlen_balancing.py"):
            nodes = [node for node in nodes if getattr(node, "name", None) in names]
        module = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], *nodes], type_ignores=[])
        exec(compile(module, path, "exec"), namespace)
    return namespace["_get_seqlen_partitions_with_dummy_padding"]


@pytest.fixture
def iterator_data(monkeypatch):
    import torch

    from tests.backends.megatron.test_data_vpp import _load_data_module

    data = _load_data_module(monkeypatch)
    for name, value in {
        "get_data_parallel_world_size": 1,
        "get_data_parallel_group": "dp",
        "get_model_parallel_group": "model",
        "get_virtual_pipeline_model_parallel_world_size": None,
        "get_context_parallel_world_size": 1,
        "get_tensor_model_parallel_world_size": 1,
    }.items():
        monkeypatch.setattr(data.mpu, name, lambda value=value, **_: value, raising=False)
    monkeypatch.setattr(data.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data.dist, "all_reduce", lambda *args, **kwargs: None)
    return data


@pytest.mark.parametrize("lengths,count", [([8, 8, 2, 2], 2), ([20, 4, 6], 3), ([4, 6], 4)])
def test_capacity_and_exact_sample_coverage_with_dummy_padding(partitioner, lengths, count):
    partitions, dummy = partitioner(lengths, count, 10)
    assert len(partitions) == count
    real = [part for i, part in enumerate(partitions) if i not in dummy]
    assert sorted(index for part in real for index in part) == list(range(len(lengths)))
    for part in real:
        assert sum(lengths[index] for index in part) <= 10 or len(part) == 1


def test_archived_binpack_count_membership_and_dummy_padding(partitioner):
    lengths = [4, 3, 5, 5, 11, 14, 12, 14, 14, 15, 6, 2]
    expected = [[9, 11], [3, 5], [0, 7], [1, 8], [2, 6], [4, 10]]
    real, dummy = partitioner(lengths, 6, 20, packing_strategy="binpack")
    assert real == expected
    assert dummy == set()
    padded, dummy = partitioner(lengths, 8, 20, packing_strategy="binpack")
    assert padded[:6] == expected
    assert padded[6:] == [expected[0], expected[0]]
    assert dummy == {6, 7}
    assert sorted(index for part in padded[:6] for index in part) == list(range(len(lengths)))
    assert all(sum(lengths[index] for index in part) <= 20 for part in padded)


def test_explicit_balanced_matches_default(partitioner):
    lengths = [4, 3, 5, 5, 11, 14, 12, 14, 14, 15, 6, 2]
    assert partitioner(lengths, 7, 20) == partitioner(lengths, 7, 20, packing_strategy="balanced")


@pytest.mark.parametrize("strategy", ["balanced", "binpack"])
@pytest.mark.parametrize("unsplit", ["is_vl_model", "uses_unsplit_forward", "multimodal_train_inputs", None])
@pytest.mark.parametrize("cp_size", [1, 4])
def test_iterator_plans_physical_cp_lengths_without_changing_samples(
    iterator_data, monkeypatch, strategy, unsplit, cp_size
):
    from types import SimpleNamespace

    monkeypatch.setattr(iterator_data.mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(iterator_data.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    # With CP4, physical lengths are 32+32 despite the raw lengths fitting one pack.
    args = SimpleNamespace(
        global_batch_size=2,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=12 if cp_size > 1 else 48,
        sft_packing_strategy=strategy,
        qkv_format="thd",
        dynamic_context_parallel=True,
    )
    lengths = [17, 17]
    rollout = {"total_lengths": lengths}
    if unsplit == "multimodal_train_inputs":
        rollout[unsplit] = [{}, {}]
    elif unsplit is not None:
        setattr(args, unsplit, True)
    iterators, counts = iterator_data.get_data_iterator(args, [], rollout)
    groups = iterators[0].micro_batch_indices
    padded = cp_size > 1 and unsplit is not None
    assert counts == [2 if padded else 1]
    assert sorted(i for group in groups for i in group) == [0, 1]
    if padded:
        assert all(len(group) * 32 <= args.max_tokens_per_gpu * cp_size for group in groups)
    assert rollout["total_lengths"] is lengths and lengths == [17, 17]


@pytest.mark.parametrize("strategy", ["balanced", "binpack"])
def test_iterator_uses_selected_packing_and_dp_aligned_count(iterator_data, monkeypatch, strategy):
    from types import SimpleNamespace

    import torch

    reductions = []

    def all_reduce(counts, *, op, group):
        assert group == "dp"
        reductions.append(counts.tolist())
        counts.fill_(8)  # Another DP rank needs eight microbatches.

    monkeypatch.setattr(iterator_data.dist, "all_reduce", all_reduce)
    lengths = [4, 3, 5, 5, 11, 14, 12, 14, 14, 15, 6, 2]
    args = SimpleNamespace(
        global_batch_size=12,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=20,
        sft_packing_strategy=strategy,
        qkv_format="thd",
        sft_whole_pack=False,
        pad_multiplier=1,
        is_vl_model=False,
        uses_unsplit_forward=False,
    )
    rollout = {"total_lengths": lengths, "tokens": [torch.zeros(n) for n in lengths]}
    iterators, counts = iterator_data.get_data_iterator(args, [], rollout)
    expected, dummy = iterator_data._get_seqlen_partitions_with_dummy_padding(
        lengths, 8, 20, packing_strategy=strategy
    )
    assert counts == [8]
    assert iterators[0].micro_batch_indices == expected
    assert iterators[0].dummy_micro_batch_offsets == dummy
    assert reductions == [
        [6 if strategy == "binpack" else iterator_data.get_minimum_num_micro_batch_size(lengths, 20)]
    ]
    if strategy == "binpack":
        assert expected[:6] == [[9, 11], [3, 5], [0, 7], [1, 8], [2, 6], [4, 10]]
        assert dummy == {6, 7}


def test_flops_objective_improves_equal_token_work_and_keeps_capacity(partitioner):
    lengths = [8] * 6
    costs = [100, 74, 99, 75, 80, 79]
    reference, _ = partitioner(lengths, 3, 16)
    actual, dummy = partitioner(lengths, 3, 16, costs=costs)
    worst = lambda groups: max(sum(costs[i] for i in group) for group in groups)
    assert worst(actual) < worst(reference)
    assert dummy == set()
    assert all(sum(lengths[i] for i in group) <= 16 for group in actual)
    assert sorted(i for group in actual for i in group) == list(range(6))


def test_flops_capacity_fallback_and_cheapest_dummy(partitioner):
    lengths = [3, 9, 7, 6, 2, 5, 2, 5, 9]
    actual, _ = partitioner(lengths, 5, 10, costs=[x * x for x in lengths])
    assert all(sum(lengths[i] for i in group) <= 10 for group in actual)
    groups, dummy = partitioner([4, 2], 3, 8, costs=[1, 100])
    assert dummy == {2}
    assert groups[2] == [0]  # Less compute even though this sample has more tokens.


def test_flops_seed_reduces_first_fit_count_and_survives_synced_count(partitioner):
    seed = partitioner.__globals__["get_flops_packing_seed"]
    lengths, costs = [4, 4, 6, 6], [16, 16, 36, 36]
    old_count = partitioner.__globals__["get_minimum_num_micro_batch_size"](lengths, 10)
    assert old_count == 3
    assert len(seed(lengths, costs, 10)) == 2
    for synced_count in (2, 3, 4, 6):
        groups, dummy = partitioner(lengths, synced_count, 10, costs=costs)
        assert len(groups) == synced_count
        assert sorted(i for j, group in enumerate(groups) if j not in dummy for i in group) == list(range(4))
        assert all(sum(lengths[i] for i in group) <= 10 for group in groups)


def test_flops_seed_preserves_oversized_singletons_and_input(partitioner):
    seed = partitioner.__globals__["get_flops_packing_seed"]
    lengths, costs = [17, 4, 6, 3, 7], [289, 16, 36, 9, 49]
    before = lengths.copy(), costs.copy()
    groups = seed(lengths, costs, 10)
    assert [0] in groups
    assert sorted(i for group in groups for i in group) == list(range(5))
    assert all(sum(lengths[i] for i in group) <= 10 or group == [0] for group in groups)
    assert (lengths, costs) == before


def test_flops_seed_randomized_capacity_coverage_and_determinism(partitioner):
    import random

    seed = partitioner.__globals__["get_flops_packing_seed"]
    first_fit = partitioner.__globals__["get_minimum_num_micro_batch_size"]
    generator = random.Random(2026)
    for _ in range(100):
        lengths = [generator.randint(1, 24) for _ in range(generator.randint(1, 32))]
        costs = [length * generator.randint(1, 500) for length in lengths]
        groups = seed(lengths, costs, 20)
        assert groups == seed(lengths, costs, 20)
        assert len(groups) <= first_fit(lengths, 20)
        assert sorted(i for group in groups for i in group) == list(range(len(lengths)))
        assert all(sum(lengths[i] for i in group) <= 20 or len(group) == 1 for group in groups)


def test_iterator_slices_costs_at_each_optimizer_step(iterator_data, monkeypatch):
    from types import SimpleNamespace

    microbatch = "flops"

    costs = [100, 74, 99, 75, 80, 79, 1, 90, 2, 91, 3, 92]
    cost_calls, sync_groups = [], []

    def estimator(args, data):
        cost_calls.append(1)
        return costs

    monkeypatch.setattr(iterator_data, "sample_costs", estimator)
    monkeypatch.setattr(iterator_data.dist, "all_reduce", lambda *args, **kw: sync_groups.append(kw["group"]))
    args = SimpleNamespace(
        microbatch_balance=microbatch,
        global_batch_size=6,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=16,
    )
    rollout = {"total_lengths": [8] * 12, "flops_cost": costs}
    iterators, counts = iterator_data.get_data_iterator(args, [], rollout)
    partition = iterator_data._get_seqlen_partitions_with_dummy_padding
    first, _ = partition([8] * 6, 3, 16, costs=costs[:6] if microbatch == "flops" else None)
    second, _ = partition([8] * 6, 3, 16, costs=costs[6:] if microbatch == "flops" else None)
    assert counts == [3, 3]
    assert iterators[0].micro_batch_indices == [*first, *[[i + 6 for i in group] for group in second]]
    assert len(cost_calls) == int(microbatch == "flops")
    assert sync_groups == (["model", "dp"] if microbatch == "flops" else []) + ["dp"]


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("reduce_k", [False, True])
def test_real_qwen35_repartition_preserves_loss_gradient_and_update(partitioner, moe, reduce_k, monkeypatch):
    import copy
    import importlib
    import inspect
    from argparse import Namespace

    import torch
    import transformers

    from relax.utils.training.qwen35_cost import estimate_sample_costs

    config_class = transformers.Qwen3_5MoeTextConfig if moe else transformers.Qwen3_5TextConfig
    model_class = transformers.Qwen3_5MoeForCausalLM if moe else transformers.Qwen3_5ForCausalLM
    # Keep this CPU test on the upstream PyTorch reference when optional
    # kernel dispatchers are installed in the training image.
    modeling = importlib.import_module(model_class.__module__)
    monkeypatch.setattr(
        modeling, "torch_chunk_gated_delta_rule", inspect.unwrap(modeling.torch_chunk_gated_delta_rule)
    )
    for name in ("chunk_gated_delta_rule", "FusedRMSNormGated", "causal_conv1d_fn"):
        monkeypatch.setattr(modeling, name, None)
    config = config_class(
        hidden_size=16,
        vocab_size=32,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=1,
        linear_key_head_dim=4,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        **(
            {
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8,
                "shared_expert_intermediate_size": 8,
            }
            if moe
            else {"intermediate_size": 32}
        ),
    )
    config._attn_implementation = "eager"
    lengths = [4, 4, 6, 6] if reduce_k else [33, 43, 27, 21, 12, 15, 59, 47]
    capacity = 10 if reduce_k else 80
    costs = estimate_sample_costs(config, {"total_lengths": lengths}, Namespace())
    baseline_k = partitioner.__globals__["get_minimum_num_micro_batch_size"](lengths, capacity) if reduce_k else 4
    optimized_k = len(partitioner.__globals__["get_flops_packing_seed"](lengths, costs, capacity)) if reduce_k else 4
    by_tokens, _ = partitioner(lengths, baseline_k, capacity)
    by_flops, _ = partitioner(lengths, optimized_k, capacity, costs=costs)
    if reduce_k:
        assert len(by_flops) < len(by_tokens)
    assert sorted(by_tokens) != sorted(by_flops)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(47)
        reference = model_class(config).train()
        actual = copy.deepcopy(reference)
        initial = [parameter.detach().clone() for parameter in reference.parameters()]
        samples = [torch.randint(0, 32, (1, length)) for length in lengths]
        losses = []
        for model, groups in ((reference, by_tokens), (actual, by_flops)):
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            total = 0.0
            for group in groups:
                for index in group:
                    loss = model(input_ids=samples[index], labels=samples[index], use_cache=False).loss
                    weighted = loss * (lengths[index] - 1) / sum(n - 1 for n in lengths)
                    weighted.backward()
                    total += float(weighted.detach())
            losses.append(total)
            optimizer.step()
        assert losses[0] == pytest.approx(losses[1], abs=1e-6)
        for left, right, before in zip(reference.parameters(), actual.parameters(), initial, strict=True):
            assert (left.grad is None) == (right.grad is None)
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(left.detach() - before, right.detach() - before, atol=2e-6, rtol=2e-5)
    finally:
        torch.set_num_threads(threads)
