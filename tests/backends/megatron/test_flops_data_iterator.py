# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise production iterator planning on CPU with mocked DP collectives.

The autograd checks below are a CPU reference for effective-batch invariance;
these are not a Megatron model, distributed training, or GPU smoke test.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from relax.utils.data.flops_balancing import partition_by_flops
from relax.utils.training.flops_counter import SampleFlopsCost


@pytest.fixture
def iterator_harness(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    names = {
        "get_first_fit_partitions",
        "get_minimum_num_micro_batch_size",
        "get_binpack_partitions",
        "_get_first_fit_partitions",
        "_partitions_fit_capacity_or_singleton_oversize",
        "_get_capacity_safe_balanced_partitions",
        "_get_seqlen_partitions_with_dummy_padding",
        "_get_micro_batch_token_capacity",
        "_round_up_to_microbatch_group",
        "DataIterator",
        "get_data_iterator",
    }
    namespace = {"logger": Mock(), "np": np, "torch": torch}
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

    config = SimpleNamespace(
        model_type="qwen3",
        hidden_size=8,
        vocab_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
    )
    cost = SampleFlopsCost.from_config(config)
    utils = ModuleType("megatron.core.utils")
    utils.get_model_config = lambda model: SimpleNamespace(microbatch_group_size_per_vp_stage=8)
    monkeypatch.setitem(sys.modules, "megatron.core.utils", utils)

    def run(lengths, *, flag=True, target=None, cp=1, tp=1, vpp=None, fail_at=None, metadata=None, **overrides):
        reductions = []
        namespace["last_reductions"] = reductions
        count_call = (3 if overrides.get("balance_data", False) else 2) if flag else 0

        def all_reduce(tensor, *, op, group):
            assert group == "dp"
            assert op == torch.distributed.ReduceOp.MAX
            ordinal = len(reductions)
            reductions.append(tensor.tolist())
            if ordinal == fail_at:
                tensor.fill_(1)
            elif ordinal == count_call and target is not None:
                required = torch.tensor(target, dtype=tensor.dtype)
                assert required.shape == tensor.shape
                assert torch.all(required >= tensor)
                tensor.copy_(required)

        namespace.update(
            mpu=SimpleNamespace(
                get_data_parallel_world_size=lambda **kw: 2,
                get_data_parallel_group=lambda: "dp",
                get_virtual_pipeline_model_parallel_world_size=lambda: vpp,
                get_context_parallel_world_size=lambda: cp,
                get_tensor_model_parallel_world_size=lambda: tp,
            ),
            dist=SimpleNamespace(all_reduce=all_reduce, ReduceOp=torch.distributed.ReduceOp),
            device_utils=SimpleNamespace(make_current_torch_device=lambda: torch.device("cpu")),
            is_preference_mode=lambda args: False,
            ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY="global_counts",
            ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY="local_counts",
        )
        args = SimpleNamespace(
            balance_by_flops=flag,
            loss_type="sft",
            train_backend="megatron",
            n_samples_per_prompt=1,
            global_batch_size=max(2, len(lengths) * 2),
            use_dynamic_batch_size=True,
            max_tokens_per_gpu=10,
            sft_packing_strategy="balanced",
            qkv_format="thd",
            _sample_flops_cost=cost,
        )
        args.__dict__.update(overrides)
        data = {"total_lengths": lengths, "sample_ids": list(range(len(lengths)))}
        data.update(metadata or {})
        iterators, counts = namespace["get_data_iterator"](args, [object()], data)
        return SimpleNamespace(
            iterators=iterators,
            counts=counts,
            reductions=reductions,
            args=args,
            data=data,
        )

    return SimpleNamespace(run=run, namespace=namespace, cost=cost)


def real_partitions(result):
    iterator = result.iterators[0]
    return [
        part
        for offset, part in enumerate(iterator.micro_batch_indices)
        if offset not in iterator.dummy_micro_batch_offsets
    ]


def assert_coverage_and_capacity(result, lengths, capacity):
    partitions = real_partitions(result)
    assert sorted(index for part in partitions for index in part) == list(range(len(lengths)))
    assert sum(result.counts) == len(result.iterators[0].micro_batch_indices)
    for part in partitions:
        assert sum(lengths[i] for i in part) <= capacity or (len(part) == 1 and lengths[part[0]] > capacity)


@pytest.mark.parametrize("strategy", ["balanced", "binpack"])
def test_flops_iterator_flag_off_preserves_existing_schedule(iterator_harness, monkeypatch, strategy):
    import relax.utils.data.flops_config as flops_config

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled FLOPs path must not parse model configuration")

    monkeypatch.setattr(flops_config, "get_sample_flops_cost", forbidden)
    lengths = [4, 3, 5, 5, 11, 14, 12, 14, 14, 15, 6, 2]
    result = iterator_harness.run(
        lengths, flag=False, target=[8], max_tokens_per_gpu=20, sft_packing_strategy=strategy
    )
    expected, dummy = iterator_harness.namespace["_get_seqlen_partitions_with_dummy_padding"](
        lengths,
        8,
        20,
        packing_strategy=strategy,
    )
    assert result.counts == [8]
    assert result.iterators[0].micro_batch_indices == expected
    assert result.iterators[0].dummy_micro_batch_offsets == dummy
    assert len(result.reductions) == 1
    assert_coverage_and_capacity(result, lengths, 20)


@pytest.mark.parametrize("target", [3, 8])
def test_flops_iterator_dp_max_expands_feasible_plan_and_pads_dummy(iterator_harness, target):
    lengths = [8, 8, 2, 2]
    costs = iterator_harness.cost.estimate(lengths)
    result = iterator_harness.run(lengths, target=[target])
    expected, dummy = iterator_harness.namespace["_get_seqlen_partitions_with_dummy_padding"](lengths, target, 10)
    real_count = len(expected) - len(dummy)
    expected[:real_count] = partition_by_flops(lengths, costs, 10, expected[:real_count])
    assert result.reductions == [[0], [0], [2]]
    assert result.counts == [target]
    assert result.iterators[0].micro_batch_indices == expected
    assert result.iterators[0].dummy_micro_batch_offsets == dummy
    assert len(dummy) == max(0, target - len(lengths))
    assert_coverage_and_capacity(result, lengths, 10)
    batches = [result.iterators[0].get_next(["sample_ids"]) for _ in range(target)]
    assert sorted(i for batch in batches if not batch.get("__is_dummy__") for i in batch["sample_ids"]) == [0, 1, 2, 3]
    assert sum(bool(batch.get("__is_dummy__")) for batch in batches) == len(dummy)


def test_flops_iterator_vpp_group_exceeds_real_sample_count(iterator_harness):
    result = iterator_harness.run([6, 4], vpp=2)
    assert result.counts == [8]
    assert len(result.iterators) == 2
    assert all(len(iterator.micro_batch_indices) == 8 for iterator in result.iterators)
    assert all(iterator.dummy_micro_batch_offsets == set(range(2, 8)) for iterator in result.iterators)
    assert result.iterators[0].micro_batch_indices == result.iterators[1].micro_batch_indices
    assert_coverage_and_capacity(result, [6, 4], 10)


@pytest.mark.parametrize("unsplit", [False, True])
def test_flops_iterator_uses_existing_cp_physical_length_capacity(iterator_harness, unsplit):
    lengths = [17, 17]
    result = iterator_harness.run(
        lengths,
        cp=4,
        tp=2,
        max_tokens_per_gpu=12,
        uses_unsplit_forward=unsplit,
    )
    physical = [32, 32] if unsplit else lengths
    assert result.counts == [2 if unsplit else 1]
    assert result.data["total_lengths"] is lengths
    assert lengths == [17, 17]
    assert_coverage_and_capacity(result, physical, 48)


def test_flops_iterator_optimizer_step_boundaries_and_synchronized_counts(iterator_harness):
    lengths = [8, 2, 6, 4]
    result = iterator_harness.run(
        lengths,
        target=[3, 4],
        balance_data=True,
        metadata={"local_counts": [2, 2], "global_counts": [4, 4]},
    )
    assert result.reductions == [[0], [2], [0], [1, 1]]
    assert result.counts == [3, 4]
    offset = 0
    real_ids = []
    for step, count in enumerate(result.counts):
        iterator = result.iterators[0]
        ids = [
            i
            for ordinal in range(offset, offset + count)
            if ordinal not in iterator.dummy_micro_batch_offsets
            for i in iterator.micro_batch_indices[ordinal]
        ]
        assert sorted(ids) == [2 * step, 2 * step + 1]
        real_ids.extend(ids)
        offset += count
    assert sorted(real_ids) == [0, 1, 2, 3]
    assert result.data["global_counts"] == [4, 4]
    assert_coverage_and_capacity(result, lengths, 10)


@pytest.mark.parametrize("lengths,fail_at", [([], None), ([4, 6], 0)])
def test_flops_iterator_empty_local_or_remote_rank_fails_before_count_sync(iterator_harness, lengths, fail_at):
    with pytest.raises(ValueError, match="non-empty SFT shard"):
        iterator_harness.run(lengths, fail_at=fail_at)
    assert len(iterator_harness.namespace["last_reductions"]) == 1


@pytest.mark.parametrize("lengths,fail_at", [([-1, 6], None), ([4, 6], 1)])
def test_flops_iterator_local_or_remote_planning_error_is_collectively_rejected(iterator_harness, lengths, fail_at):
    with pytest.raises(ValueError, match="planning failed on a DP rank"):
        iterator_harness.run(lengths, fail_at=fail_at)
    reductions = iterator_harness.namespace["last_reductions"]
    assert reductions == [[0], [int(lengths[0] < 0)]]


def test_flops_iterator_config_load_failure_is_collectively_rejected(iterator_harness, monkeypatch):
    from relax.utils.data import flops_config

    def unavailable_config(args):
        raise OSError("model config unavailable on this worker")

    monkeypatch.setattr(flops_config, "get_sample_flops_cost", unavailable_config)
    with pytest.raises(ValueError, match="planning failed on a DP rank") as caught:
        iterator_harness.run([4, 6])
    assert isinstance(caught.value.__cause__, OSError)
    assert iterator_harness.namespace["last_reductions"] == [[0], [1]]


@pytest.mark.parametrize("normalization", ["sample_mean", "token_mean"])
def test_flops_iterator_cpu_reference_loss_gradient_and_fp32_update(iterator_harness, normalization):
    lengths = [8, 3, 5, 2, 7, 4]
    data = {"features": [], "targets": [], "masks": []}
    for index, length in enumerate(lengths):
        positions = torch.arange(length, dtype=torch.float32)
        data["features"].append(torch.stack([positions / length + index / 10, torch.ones(length)], dim=1))
        data["targets"].append(positions / 20 - index / 7)
        mask = torch.ones(length)
        mask[0] = 0
        data["masks"].append(mask)
    valid_tokens = sum(mask.sum().item() for mask in data["masks"])
    results = []
    for flag in (False, True):
        result = iterator_harness.run(lengths, flag=flag, target=[8], metadata=data)
        parameter = torch.nn.Parameter(torch.tensor([0.3, -0.2], dtype=torch.float32))
        optimizer = torch.optim.SGD([parameter], lr=0.01)
        initial = parameter.detach().clone()
        loss_total = torch.zeros(())
        effective_ids = []
        effective_tokens = 0
        dummy_count = 0
        for _ in range(sum(result.counts)):
            batch = result.iterators[0].get_next(["features", "targets", "masks", "sample_ids"])
            dummy = bool(batch.get("__is_dummy__"))
            dummy_count += dummy
            micro_loss = torch.zeros(())
            for features, targets, mask in zip(batch["features"], batch["targets"], batch["masks"], strict=True):
                error = ((features @ parameter - targets) ** 2 * mask).sum()
                denominator = mask.sum() * len(lengths) if normalization == "sample_mean" else valid_tokens
                micro_loss = micro_loss + error / denominator
            if dummy:
                micro_loss = micro_loss * 0
            else:
                effective_ids.extend(batch["sample_ids"])
                effective_tokens += sum(int(mask.sum()) for mask in batch["masks"])
            gradient_before = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.clone()
            micro_loss.backward()
            if dummy:
                torch.testing.assert_close(parameter.grad, gradient_before, rtol=0, atol=0)
            loss_total += micro_loss.detach()
        gradient = parameter.grad.detach().clone()
        optimizer.step()
        delta = parameter.detach() - initial
        assert torch.isfinite(loss_total)
        assert sorted(effective_ids) == list(range(len(lengths)))
        assert effective_tokens == valid_tokens
        assert dummy_count == 2
        results.append((loss_total, gradient, delta))
    # Independent full-effective-batch reference, without any partitions.
    reference_parameter = torch.nn.Parameter(torch.tensor([0.3, -0.2], dtype=torch.float32))
    initial = reference_parameter.detach().clone()
    if normalization == "token_mean":
        errors = (torch.cat(data["features"]) @ reference_parameter - torch.cat(data["targets"])) ** 2
        reference_loss = (errors * torch.cat(data["masks"])).sum() / valid_tokens
    else:
        reference_loss = torch.stack(
            [
                (((features @ reference_parameter - targets) ** 2) * mask).sum() / mask.sum()
                for features, targets, mask in zip(data["features"], data["targets"], data["masks"], strict=True)
            ]
        ).mean()
    reference_loss.backward()
    reference_gradient = reference_parameter.grad.detach().clone()
    torch.optim.SGD([reference_parameter], lr=0.01).step()
    reference_delta = reference_parameter.detach() - initial
    for result in results:
        for actual, expected in zip(result, (reference_loss, reference_gradient, reference_delta), strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    for baseline, optimized in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(baseline, optimized, rtol=1e-5, atol=1e-7)


def test_flops_iterator_preserves_oversized_singleton_exception(iterator_harness):
    lengths = [20, 4, 6]
    result = iterator_harness.run(lengths)
    assert [0] in real_partitions(result)
    assert result.counts == [2]
    assert_coverage_and_capacity(result, lengths, 10)


@pytest.mark.parametrize("strategy", ["balanced", "binpack"])
def test_flops_iterator_fixed_K_matches_baseline_after_dp_max(iterator_harness, strategy):
    lengths = [9, 3, 8, 2, 9, 1, 5, 5, 4, 6]
    baseline = iterator_harness.run(lengths, flag=False, sft_packing_strategy=strategy)
    optimized = iterator_harness.run(lengths, sft_packing_strategy=strategy)
    assert optimized.counts == baseline.counts
    assert_coverage_and_capacity(optimized, lengths, 10)
