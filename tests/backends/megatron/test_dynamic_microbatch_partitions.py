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
def test_iterator_plans_physical_cp_lengths_without_changing_samples(partitioner, strategy, unsplit, cp_size):
    from types import SimpleNamespace

    import numpy as np
    import torch

    root = Path(__file__).resolve().parents[3]
    namespace = partitioner.__globals__
    names = {"get_data_iterator", "_get_micro_batch_token_capacity"}
    nodes = [
        node
        for node in ast.parse((root / "relax/backends/megatron/data.py").read_text()).body
        if getattr(node, "name", None) in names
    ]
    mpu = SimpleNamespace(
        get_data_parallel_world_size=lambda **kwargs: 1,
        get_data_parallel_group=lambda: "dp",
        get_virtual_pipeline_model_parallel_world_size=lambda: None,
        get_context_parallel_world_size=lambda: cp_size,
        get_tensor_model_parallel_world_size=lambda: 2,
    )
    namespace.update(
        torch=torch,
        np=np,
        mpu=mpu,
        dist=SimpleNamespace(all_reduce=lambda *a, **kw: None, ReduceOp=torch.distributed.ReduceOp),
        device_utils=SimpleNamespace(make_current_torch_device=lambda: torch.device("cpu")),
        is_preference_mode=lambda args: False,
        ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY="global_counts",
        ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY="local_counts",
        DataIterator=lambda data, micro_batch_size, indices, **kw: SimpleNamespace(indices=indices, **kw),
    )
    module = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], *nodes], type_ignores=[])
    exec(compile(module, "data.py", "exec"), namespace)
    # With CP4, raw lengths fit one 48-token pack, but physical lengths are 32+32.
    args = SimpleNamespace(
        global_batch_size=2,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=12 if cp_size > 1 else 48,
        sft_packing_strategy=strategy,
        qkv_format="thd",
        dynamic_context_parallel=True,
    )
    lengths = [17, 17]
    data = {"total_lengths": lengths}
    if unsplit == "multimodal_train_inputs":
        data[unsplit] = [{}, {}]
    elif unsplit is not None:
        setattr(args, unsplit, True)
    iterators, counts = namespace["get_data_iterator"](args, [], data)
    padded = cp_size > 1 and unsplit is not None
    assert counts == [2 if padded else 1]
    assert sorted(i for pack in iterators[0].indices for i in pack) == [0, 1]
    if padded:
        assert all(len(pack) * 32 <= args.max_tokens_per_gpu * cp_size for pack in iterators[0].indices)
    assert data["total_lengths"] is lengths
    assert lengths == [17, 17]


@pytest.mark.parametrize("strategy", ["balanced", "binpack"])
def test_iterator_uses_selected_packing_and_dp_aligned_count(partitioner, strategy):
    from types import SimpleNamespace

    import numpy as np
    import torch

    root = Path(__file__).resolve().parents[3]
    namespace = partitioner.__globals__
    names = {"get_data_iterator", "_get_micro_batch_token_capacity"}
    nodes = [
        n
        for n in ast.parse((root / "relax/backends/megatron/data.py").read_text()).body
        if getattr(n, "name", None) in names
    ]
    reductions = []

    def all_reduce(counts, *, op, group):
        assert group == "dp"
        reductions.append(counts.tolist())
        counts.fill_(8)  # Another DP rank needs eight microbatches.

    namespace.update(
        torch=torch,
        np=np,
        mpu=SimpleNamespace(
            get_data_parallel_world_size=lambda **kw: 1,
            get_data_parallel_group=lambda: "dp",
            get_virtual_pipeline_model_parallel_world_size=lambda: None,
            get_context_parallel_world_size=lambda: 1,
            get_tensor_model_parallel_world_size=lambda: 1,
        ),
        dist=SimpleNamespace(all_reduce=all_reduce, ReduceOp=torch.distributed.ReduceOp),
        device_utils=SimpleNamespace(make_current_torch_device=lambda: torch.device("cpu")),
        is_preference_mode=lambda args: False,
        ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY="global_counts",
        ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY="local_counts",
        DataIterator=lambda data, micro_batch_size, indices, **kw: SimpleNamespace(indices=indices, **kw),
    )
    exec(
        compile(
            ast.Module(body=[ast.parse("from __future__ import annotations").body[0], *nodes], type_ignores=[]),
            "data.py",
            "exec",
        ),
        namespace,
    )
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
    iterators, counts = namespace["get_data_iterator"](args, [], rollout)
    expected, dummy = partitioner(lengths, 8, 20, packing_strategy=strategy)
    assert counts == [8]
    assert iterators[0].indices == expected
    assert iterators[0].dummy_micro_batch_offsets == dummy
    assert reductions == [[6 if strategy == "binpack" else namespace["get_minimum_num_micro_batch_size"](lengths, 20)]]
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


def test_iterator_slices_costs_at_each_optimizer_step(partitioner):
    from types import SimpleNamespace

    import numpy as np
    import torch

    root = Path(__file__).resolve().parents[3]
    namespace = partitioner.__globals__
    nodes = [
        n
        for n in ast.parse((root / "relax/backends/megatron/data.py").read_text()).body
        if getattr(n, "name", None) in ("get_data_iterator", "_get_micro_batch_token_capacity")
    ]
    costs = [100, 74, 99, 75, 80, 79, 1, 90, 2, 91, 3, 92]
    namespace.update(
        torch=torch,
        np=np,
        mpu=SimpleNamespace(
            get_data_parallel_world_size=lambda **kw: 1,
            get_data_parallel_group=lambda **kw: "dp",
            get_model_parallel_group=lambda: "model",
            get_virtual_pipeline_model_parallel_world_size=lambda: None,
            get_context_parallel_world_size=lambda: 1,
            get_tensor_model_parallel_world_size=lambda: 1,
        ),
        dist=SimpleNamespace(all_reduce=lambda *a, **kw: None, ReduceOp=torch.distributed.ReduceOp),
        device_utils=SimpleNamespace(make_current_torch_device=lambda: torch.device("cpu")),
        is_preference_mode=lambda args: False,
        ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY="global_counts",
        ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY="local_counts",
        DataIterator=lambda data, micro_batch_size, indices, **kw: SimpleNamespace(indices=indices, **kw),
        sample_costs=lambda args, data: costs,
    )
    exec(
        compile(
            ast.Module(body=[ast.parse("from __future__ import annotations").body[0], *nodes], type_ignores=[]),
            "data.py",
            "exec",
        ),
        namespace,
    )
    args = SimpleNamespace(
        microbatch_balance="flops", global_batch_size=6, use_dynamic_batch_size=True, max_tokens_per_gpu=16
    )
    rollout = {"total_lengths": [8] * 12}
    iterators, counts = namespace["get_data_iterator"](args, [], rollout)
    first, _ = partitioner([8] * 6, 3, 16, costs=costs[:6])
    second, _ = partitioner([8] * 6, 3, 16, costs=costs[6:])
    assert counts == [3, 3]
    assert iterators[0].indices == [*first, *[[i + 6 for i in group] for group in second]]


@pytest.mark.parametrize("moe", [False, True])
def test_real_qwen35_repartition_preserves_loss_gradient_and_update(partitioner, moe, monkeypatch):
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
    lengths = [33, 43, 27, 21, 12, 15, 59, 47]
    costs = estimate_sample_costs(config, {"total_lengths": lengths}, Namespace())
    by_tokens, _ = partitioner(lengths, 4, 80)
    by_flops, _ = partitioner(lengths, 4, 80, costs=costs)
    assert sorted(by_tokens) != sorted(by_flops)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(47)
        reference = model_class(config).train()
        actual = copy.deepcopy(reference)
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
        for left, right in zip(reference.parameters(), actual.parameters(), strict=True):
            assert (left.grad is None) == (right.grad is None)
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    finally:
        torch.set_num_threads(threads)
