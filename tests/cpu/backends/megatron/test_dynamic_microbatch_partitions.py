# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Keep balanced packing capacity protection and dummy padding compatible."""

import pytest


@pytest.fixture
def partitioner(iterator_data):
    return iterator_data._get_seqlen_partitions_with_dummy_padding


@pytest.fixture
def iterator_data(monkeypatch):
    import torch

    from tests.cpu.backends.megatron.test_data_vpp import _load_data_module

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


@pytest.mark.parametrize("rank", [0, 1])
def test_iterator_executes_central_packs_without_local_repacking(iterator_data, monkeypatch, rank):
    from types import SimpleNamespace

    from relax.utils.data.flops_balancing import encode_schedule, restore_schedule

    monkeypatch.setattr(iterator_data.mpu, "get_data_parallel_world_size", lambda **_: 2)
    monkeypatch.setattr(iterator_data.mpu, "get_data_parallel_rank", lambda **_: rank, raising=False)
    metadata = {i: dict(total_lengths=1, flops_cost=9 if i == 5 else 1) for i in range(6)}
    rows, updates = encode_schedule(
        list(range(6)),
        metadata,
        dict(version=4, capacity=5, microbatch_group_size=1, context_parallel_size=1, pipeline_stages=2),
        2,
        6,
        ("p", "sft_train", 0),
    )
    local = rows[rank]
    batch = dict(total_lengths=[1] * len(local), ids=local)
    restore_schedule(
        SimpleNamespace(global_indexes=local, get_all_custom_meta=lambda: [updates[i] for i in local]), batch
    )
    args = SimpleNamespace(
        balance_by_flops=True,
        use_dynamic_batch_size=True,
        global_batch_size=6,
        max_tokens_per_gpu=5,
        pipeline_model_parallel_size=2,
    )
    monkeypatch.setattr(
        iterator_data, "_get_capacity_safe_balanced_partitions", lambda *_: pytest.fail("central packs must survive")
    )
    iterators, counts = iterator_data.get_data_iterator(args, [], batch)
    assert counts == [1]
    assert iterators[0].get_next(["ids"])["ids"] == local
    assert batch["rollout_mini_local_sample_counts"] == [len(local)]
    assert batch["rollout_mini_global_sample_counts"] == [6]


def test_iterator_rejects_missing_central_plan_instead_of_silent_local_fallback(iterator_data):
    from types import SimpleNamespace

    args = SimpleNamespace(
        balance_by_flops=True, use_dynamic_batch_size=True, global_batch_size=2, max_tokens_per_gpu=5
    )
    with pytest.raises(ValueError, match="Invalid FLOPs schedule"):
        iterator_data.get_data_iterator(args, [], dict(total_lengths=[1, 1]))


def test_forward_explicit_budget_needs_no_training_plan(iterator_data):
    from types import SimpleNamespace

    args = SimpleNamespace(
        balance_by_flops=True, use_dynamic_batch_size=True, global_batch_size=2, max_tokens_per_gpu=5
    )
    iterators, counts = iterator_data.get_data_iterator(args, [], dict(total_lengths=[2, 2]), max_tokens_per_gpu=5)
    assert counts == [1]
    assert iterators[0].micro_batch_indices == [[0, 1]]


def test_multi_mini_restore_errors_reach_synchronised_validation(iterator_data, monkeypatch):
    from types import SimpleNamespace

    batch = iterator_data.concat_rollout_batches(
        [
            dict(total_lengths=[1], _flops_schedule_error=["Partial schedule"]),
            dict(total_lengths=[1], _flops_schedule_error=["Mixed schedules"]),
        ]
    )
    assert batch["_flops_schedule_error"] == ["Partial schedule", "Mixed schedules"]
    reductions = []
    monkeypatch.setattr(iterator_data.dist, "all_reduce", lambda status, **_: reductions.append(status.tolist()))
    args = SimpleNamespace(
        balance_by_flops=True, use_dynamic_batch_size=True, global_batch_size=2, max_tokens_per_gpu=5
    )
    with pytest.raises(ValueError, match="Invalid FLOPs schedule"):
        iterator_data.get_data_iterator(args, [], batch)
    assert len(reductions) == 2 and reductions[0][0] == 1
