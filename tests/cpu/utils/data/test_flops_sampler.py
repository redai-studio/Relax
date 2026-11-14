# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import inspect
from types import SimpleNamespace

import pytest

from relax.utils.data.flops_balancing import (
    PLAN_METADATA_FIELD,
    FlopsWorkload,
    get_dp_packing_geometry,
    local_schedule,
    restore_schedule,
)
from tests.cpu.utils.test_flops_counter import FLOPS_TEST_CONFIGS


pytest.importorskip("transfer_queue")

from transfer_queue import SeqlenBalancedSampler

from relax.utils.data.flops_sampler import FlopsBalancedSampler
from relax.utils.data.identity_window_sampler import IdentityWindowSampler


class Partition:
    def __init__(self, lengths: list[int], rows_per_identity: int = 1):
        self.global_indexes = set(range(len(lengths)))
        self.metadata = {
            i: {"sample_index": i // rows_per_identity, "total_lengths": length, "producer": i}
            for i, length in enumerate(lengths)
        }

    def get_custom_meta(self, indexes: list[int]) -> dict:
        return {i: self.metadata[i] for i in indexes}

    def set_custom_meta(self, updates: dict) -> None:
        self.metadata.update(updates)


@pytest.fixture
def workload():
    return FlopsWorkload.from_config(FLOPS_TEST_CONFIGS["qwen3_dense"]["config"])


@pytest.fixture
def geometry():
    return get_dp_packing_geometry(SimpleNamespace(balance_by_flops=True, max_tokens_per_gpu=24576), "train")


def _assert_restored_plan(partition: Partition, rows: list[int], global_count: int) -> None:
    rollout = {"total_lengths": [partition.metadata[i]["total_lengths"] for i in rows]}
    batch_meta = SimpleNamespace(
        global_indexes=rows, get_all_custom_meta=lambda: [partition.metadata[i] for i in rows]
    )
    restore_schedule(batch_meta, rollout)
    indices, counts, dummy = local_schedule(rollout)
    real = [i for j, pack in enumerate(indices) if j not in dummy for i in pack]
    assert sorted(real) == list(range(len(rows)))
    assert counts == [len(indices)]
    assert rollout["rollout_mini_global_sample_counts"] == [global_count]


def _assert_original_metadata(partition: Partition, original: dict) -> None:
    assert {
        i: {k: v for k, v in meta.items() if k != PLAN_METADATA_FIELD} for i, meta in partition.metadata.items()
    } == original


def test_flops_sampler_requires_assignment_api(monkeypatch, workload):
    monkeypatch.setattr(SeqlenBalancedSampler, "sample", lambda self, *args, **kwargs: ([], []))
    with pytest.raises(RuntimeError, match="assignment_transform"):
        FlopsBalancedSampler(dp_size=2, workload=workload)


@pytest.mark.parametrize("first_rank", [0, 1])
def test_flops_sft_sampler_preserves_selection_metadata_and_checkpoint(first_rank, workload, geometry):
    if "assignment_transform" not in inspect.signature(SeqlenBalancedSampler.sample).parameters:
        pytest.skip("Requires the existing Docker TransferQueue assignment_transform patch")
    sampler = FlopsBalancedSampler(dp_size=2, workload=workload)
    partition = Partition([11000] + [2000] * 7)
    original_metadata = copy.deepcopy(partition.metadata)
    common = dict(
        batch_size=4,
        task_name="sft_train",
        partition_id="sft_0",
        partition=partition,
        batch_index=0,
        dp_packing_geometry=geometry,
    )
    assert sampler.sample([0, 1], dp_rank=first_rank, **common) == ([], [])
    first, consumed = sampler.sample(list(range(8)), dp_rank=first_rank, **common)
    assert first == consumed
    state = copy.deepcopy(sampler.save_checkpoint())
    restored = FlopsBalancedSampler(dp_size=2, workload=workload)
    restored.load_checkpoint(state)
    assert restored.has_cached_result("sft_0", "sft_train", {"dp_rank": 1 - first_rank, "batch_index": 0})
    assert restored.sample([], dp_rank=first_rank, **common)[0] == first
    second = restored.sample([], dp_rank=1 - first_rank, **common)[0]
    assert len(first) != len(second)
    assert sorted(first + second) == list(range(8))
    baseline = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=2)
    baseline_groups = [baseline.sample(list(range(8)), dp_rank=rank, **common)[0] for rank in (0, 1)]
    costs = workload([partition.metadata[i]["total_lengths"] for i in range(8)])
    assert max(sum(costs[i] for i in group) for group in (first, second)) < max(
        sum(costs[i] for i in group) for group in baseline_groups
    )
    _assert_restored_plan(partition, first, 8)
    _assert_restored_plan(partition, second, 8)
    _assert_original_metadata(partition, original_metadata)
    with pytest.raises(ValueError, match="geometry changed"):
        restored.sample([], dp_rank=first_rank, **{**common, "dp_packing_geometry": {**geometry, "capacity": 1}})
    for rank in (0, 1):
        eval_kwargs = {
            **common,
            "task_name": "sft_eval",
            "partition_id": "eval_0",
            "dp_rank": rank,
            "dp_packing_geometry": None,
        }
        baseline = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=2)
        assert restored.sample(list(range(8)), **eval_kwargs) == baseline.sample(list(range(8)), **eval_kwargs)
    restored.clear_cache("sft_0")
    assert not restored.has_cached_result("sft_0", "sft_train", {"dp_rank": 0, "batch_index": 0})


@pytest.mark.parametrize("first_rank", [0, 1])
def test_flops_rl_sampler_preserves_identity_windows_and_context_rows(first_rank, workload, geometry):
    sampler = IdentityWindowSampler(dp_size=2, placement="balanced", workload=workload)
    baseline = IdentityWindowSampler(dp_size=2, placement="balanced")
    partition = Partition([512, 2048, 3072, 4096, 8192, 16384, 24576, 32768] * 2, rows_per_identity=2)
    original_metadata = copy.deepcopy(partition.metadata)
    common = dict(
        batch_size=2, task_name="train", partition_id="train_0", partition=partition, dp_packing_geometry=geometry
    )
    all_rows = []
    for window in range(2):
        ready = list(range(window * 8, (window + 1) * 8))
        assert sampler.sample(ready[:6], dp_rank=first_rank, batch_index=window, **common) == ([], [])
        first = sampler.sample(ready, dp_rank=first_rank, batch_index=window, **common)[0]
        restored = IdentityWindowSampler(dp_size=2, placement="balanced", workload=workload)
        restored.load_checkpoint(copy.deepcopy(sampler.save_checkpoint()))
        assert restored.sample([], dp_rank=first_rank, batch_index=window, **common)[0] == first
        second = restored.sample([], dp_rank=1 - first_rank, batch_index=window, **common)[0]
        baseline_rows = [
            row
            for rank in (0, 1)
            for row in baseline.sample(
                ready, dp_rank=rank, batch_index=window, **{**common, "dp_packing_geometry": None}
            )[0]
        ]
        assert sorted(first + second) == sorted(baseline_rows) == ready
        _assert_restored_plan(partition, first, 4)
        _assert_restored_plan(partition, second, 4)
        assert restored.is_window_drained("train_0", "train", window, 4, False, False)
        all_rows.extend(first + second)
        sampler = restored
    assert sorted(all_rows) == list(range(16))
    _assert_original_metadata(partition, original_metadata)
