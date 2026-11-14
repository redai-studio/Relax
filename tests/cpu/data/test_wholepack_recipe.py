# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Whole-pack replay and exact-once data contracts; CPU only."""

import copy
import inspect
from types import SimpleNamespace

import pytest

from relax.utils.data import wholepack as wp


def test_wholepack_binpack_capacity_and_stable_assignment():
    lengths = [9, 1, 7, 2, 12, 3, 4]
    result = wp._binpack_partitions(lengths, 10)
    assert result == [[4], [0], [2, 3], [1, 5, 6]]
    assert sorted(i for pack in result for i in pack) == list(range(len(lengths)))
    assert all(sum(lengths[i] for i in pack) <= 10 or len(pack) == 1 for pack in result)


@pytest.mark.parametrize("ready", [True, False])
def test_wholepack_mixed_readiness_retains_acquired_batch(ready):
    batch = ({"tokens": [1]}, object())
    states = iter([[1, 1, 0], [1, 0, 0]])
    calls = []

    def fetch():
        calls.append(1)
        return batch

    actual = wp.await_batch(
        batch if ready else (None, None), fetch, lambda _: next(states), timeout_seconds=5, poll_seconds=0
    )
    assert actual is batch
    assert len(calls) == (0 if ready else 1)


def test_wholepack_all_empty_returns_to_outer_poll():
    assert wp.await_batch(
        (None, None), lambda: pytest.fail("Unexpected fetch"), lambda _: [0, 1, 0], timeout_seconds=5, poll_seconds=0
    ) == (None, None)


def test_wholepack_peer_timeout_is_collective():
    with pytest.raises(TimeoutError):
        wp.await_batch(({}, None), lambda: None, lambda _: [1, 1, 1], timeout_seconds=5, poll_seconds=0)


def test_wholepack_scope_requires_explicit_recipe():
    assert not wp.enabled(SimpleNamespace())
    assert not wp.enabled(
        SimpleNamespace(custom_megatron_before_train_step_hook_path="relax.models.qwen3_5.recipe.before_step")
    )
    assert wp.enabled(SimpleNamespace(sft_wholepack=True))
    assert wp.enabled(
        SimpleNamespace(sft_wholepack=True, custom_megatron_before_train_step_hook_path="custom.wrapper")
    )


@pytest.mark.parametrize("oversize", [False, True])
@pytest.mark.parametrize(
    "dp,gbs,tp,capacity,pad",
    [
        (4, 400, 4, 262144, None),
        (2, 20, 4, 262144, None),
        (4, 40, 2, 131072, None),
        (4, 4, 2, 131072, None),
        (4, 4, 4, 262144, None),
        (8, 800, 4, 262144, None),
        (8, 1600, 4, 262144, None),
        (14, 1400, 4, 262144, None),
        (16, 1600, 4, 262144, None),
        (16, 2048, 4, 262144, None),
        (16, 16, 4, 262144, None),
        (32, 1600, 2, 262144, None),
        (16, 800, 2, 131072, None),
        (32, 3200, 4, 262144, None),
        (4, 40, 4, 1024, 1024),
        (32, 1600, None, 131072, None),
    ],
)
def test_wholepack_real_sampler_checkpoint_replay(oversize, dp, gbs, tp, capacity, pad):
    import transfer_queue

    if vars(transfer_queue).get("__relax_ci_stub__", False) or not hasattr(transfer_queue, "SeqlenBalancedSampler"):
        pytest.skip("Requires Docker-pinned TransferQueue 9784ad0")
    if "assignment_transform" not in inspect.signature(transfer_queue.SeqlenBalancedSampler.sample).parameters:
        pytest.skip("Requires the image-built TransferQueue assignment_transform patch")

    class Partition:
        def __init__(self):
            self.metadata = {i: {"total_lengths": 1800 + (i * 997) % 6000, "producer": i} for i in range(gbs)}
            if oversize:
                self.metadata[0]["total_lengths"] = 300000

        def get_custom_meta(self, ids):
            return {i: self.metadata[i] for i in ids}

        def set_custom_meta(self, values):
            self.metadata.update(values)

    partition = Partition()
    sampler = wp.WholepackSampler(dp_size=dp, capacity=capacity, pad_size=pad, tp_size=tp, global_batch_size=gbs)
    assert sampler.tp_size == (2 if tp is None else tp)
    original = []
    hashes, waves = set(), set()
    for rank in range(dp):
        ids, consumed = sampler.sample(
            list(range(gbs)), gbs // dp, "sft_train", "rollout_0", dp_rank=rank, batch_index=0, partition=partition
        )
        assert ids == consumed
        original.append(ids)
        data = {"total_lengths": [partition.metadata[i]["total_lengths"] for i in ids]}
        meta = SimpleNamespace(global_indexes=ids, get_all_custom_meta=lambda: [partition.metadata[i] for i in ids])
        wp.restore_plan(meta, data, gbs)
        hashes.add(data[wp.PLAN_KEY]["sha256"])
        waves.add(data[wp.PLAN_KEY]["K"])
        assert data["rollout_mini_global_sample_counts"] == [gbs]
        assert all(partition.metadata[i]["producer"] == i for i in ids)
    assert sorted(i for ids in original for i in ids) == list(range(gbs))
    assert len(hashes) == len(waves) == 1
    recovered = wp.WholepackSampler(dp_size=dp, capacity=capacity, pad_size=pad, tp_size=tp, global_batch_size=gbs)
    if gbs == 2048:
        incompatible = wp.WholepackSampler(
            dp_size=dp, capacity=capacity, pad_size=pad, tp_size=tp, global_batch_size=1600
        )
        with pytest.raises(ValueError, match="GBS mismatch"):
            incompatible.load_checkpoint(copy.deepcopy(sampler.save_checkpoint()))
        corrupt = copy.deepcopy(sampler.save_checkpoint())
        for updates in corrupt["wholepack_metadata"].values():
            for meta in updates.values():
                if "slots" in meta.get(wp.PLAN_KEY, {}):
                    meta[wp.PLAN_KEY]["global_count"] = 1600
        with pytest.raises(ValueError, match="padding geometry mismatch"):
            recovered.load_checkpoint(corrupt)
    recovered.load_checkpoint(copy.deepcopy(sampler.save_checkpoint()))
    partition = Partition()
    for rank in reversed(range(dp)):
        ids, _ = recovered.sample(
            [], gbs // dp, "sft_train", "rollout_0", dp_rank=rank, batch_index=0, partition=partition
        )
        assert ids == original[rank]
        assert wp.PLAN_KEY in partition.metadata[ids[0]]
    # Eval still uses the existing balanced sampler, even with a smaller batch.
    eval_per_rank = min(10, gbs // dp)
    ids, _ = recovered.sample(
        list(range(dp * eval_per_rank)),
        eval_per_rank,
        "sft_eval",
        "eval_0",
        dp_rank=0,
        batch_index=0,
        partition=partition,
    )
    assert len(ids) == eval_per_rank
    assert ("eval_0", "sft_eval", 0) not in recovered.plan_metadata


def _wholepack_gloo_worker(rank, rendezvous):
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    group = dist.group.WORLD
    try:

        def agree(values):
            tensor = torch.tensor(values, dtype=torch.int64)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
            return tensor.tolist()

        data = ({"rank": rank}, rank)
        fetches = []

        def fetch():
            fetches.append(1)
            return data

        assert (
            wp.await_batch(data if rank == 0 else (None, None), fetch, agree, timeout_seconds=10, poll_seconds=0)
            == data
        )
        assert len(fetches) == rank
        # A local invalid plan must fail on both ranks, without a divergent next collective.
        args = SimpleNamespace(
            sft_wholepack=True,
            custom_megatron_before_train_step_hook_path="relax.models.qwen3_5.recipe.before_step",
            tensor_model_parallel_size=4,
            context_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=8,
            expert_tensor_parallel_size=1,
            global_batch_size=1600,
            n_samples_per_prompt=1,
            data_pad_size_multiplier=1024,
            max_tokens_per_gpu=262144,
            qkv_format="thd",
            use_dynamic_batch_size=True,
        )
        valid = {
            "total_lengths": [1],
            wp.PLAN_KEY: {
                "indices": [[0]],
                "dummy": [],
                "capacity": 262144,
                "pad_size": 4096,
                "K": 1,
                "sha256": "a" * 64,
            },
        }
        # With no serialized metadata, restore_plan leaves this already-restored
        # valid plan unchanged. The other rank alone has a missing plan.
        meta = SimpleNamespace(get_all_custom_meta=lambda: [])
        with pytest.raises(RuntimeError, match="validation failed"):
            wp.restore_checked(args, meta, valid if rank == 0 else {}, agree)
    finally:
        dist.destroy_process_group(group)


def test_wholepack_gloo_readiness_and_error_propagation(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_wholepack_gloo_worker, args=(f"file://{tmp_path / 'gloo'}",), nprocs=2, join=True)


@pytest.mark.parametrize("invalid", [False, True])
def test_real_assignment_api_owns_cache_and_checkpoint_replay(invalid):
    import transfer_queue

    cls = getattr(transfer_queue, "SeqlenBalancedSampler", None)
    if cls is None or "assignment_transform" not in inspect.signature(cls.sample).parameters:
        pytest.skip("Requires the image-built TransferQueue assignment_transform patch")
    sampler = cls(n_samples_per_prompt=1, dp_size=2)
    calls = []

    def transform(groups):
        calls.append(groups)
        return [[0, 1, 2], [3]]

    kwargs = dict(task_name="train", partition_id="p", dp_rank=0, batch_index=0)
    if invalid:
        with pytest.raises(ValueError, match="preserve all identities"):
            sampler.sample(list(range(4)), 2, assignment_transform=lambda _: [[0], [0]], **kwargs)
        assert not sampler.has_cached_result("p", "train", {"dp_rank": 0, "batch_index": 0})
    assert sampler.sample(list(range(4)), 2, assignment_transform=transform, **kwargs) == ([0, 1, 2], [0, 1, 2])
    recovered = cls(n_samples_per_prompt=1, dp_size=2)
    recovered.load_checkpoint(copy.deepcopy(sampler.save_checkpoint()))
    kwargs["dp_rank"] = 1
    assert recovered.has_cached_result("p", "train", {"dp_rank": 1, "batch_index": 0})
    assert recovered.sample([], 2, assignment_transform=transform, **kwargs) == ([3], [3])
    assert len(calls) == 1
    with pytest.raises(ValueError, match="DP geometry"):
        cls(n_samples_per_prompt=1, dp_size=4).load_checkpoint(sampler.save_checkpoint())


@pytest.mark.parametrize("dp,gbs,tp", [(0, 400, 4), (3, 400, 4), (8, 801, 4), (8, 0, 4), (8, 800, 3), (16, 2047, 4)])
def test_wholepack_rejects_incomplete_geometry(dp, gbs, tp):
    with pytest.raises(ValueError):
        wp.WholepackSampler(dp_size=dp, global_batch_size=gbs, tp_size=tp)


@pytest.mark.parametrize("wholepack", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_model_recipe_and_wholepack_are_independent(wholepack, optimized):
    from relax.models.qwen3_5 import recipe

    args = SimpleNamespace(
        sft_wholepack=wholepack, custom_megatron_before_train_step_hook_path=recipe.ENTRY if optimized else None
    )
    assert wp.enabled(args) is wholepack
    assert recipe.enabled(args) is optimized


@pytest.mark.parametrize("wholepack", [False, True])
@pytest.mark.parametrize("gbs", [16, 1600, 2048])
@pytest.mark.parametrize("multimodal", [False, True])
def test_optimization_contract_preserves_topology_with_wholepack_disabled(wholepack, gbs, multimodal):
    from relax.models.qwen3_5 import recipe

    args = SimpleNamespace(
        custom_megatron_before_train_step_hook_path=recipe.ENTRY,
        sft_wholepack=wholepack,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        n_samples_per_prompt=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        qkv_format="thd",
        use_dynamic_batch_size=True,
        global_batch_size=gbs,
        loss_type="sft",
        train_backend="megatron",
        max_staleness=1,
        distributed_backend="nccl",
        multimodal_keys={"image": "images"} if multimodal else None,
        data_pad_size_multiplier=1024,
        max_tokens_per_gpu=262144,
        hidden_size=2048,
        num_experts=256,
        bf16=True,
        use_distributed_optimizer=True,
        sequence_parallel=True,
        use_precision_aware_optimizer=True,
        optimizer_cpu_offload=True,
        optimizer_offload_fraction=0.0,
    )
    from relax.engine.sft.data_pipeline_config import configure_sft_data_pipeline

    configure_sft_data_pipeline(args)
    assert not args.sft_async_prepack
    assert args.sft_train_data_prefetch
    recipe._check_args(args)
    for key, value in (
        ("pipeline_model_parallel_size", 2),
        ("context_parallel_size", 2),
        ("expert_model_parallel_size", 4),
        ("expert_tensor_parallel_size", 2),
        ("sft_async_prepack", True),
        ("use_dynamic_batch_size", False),
        ("data_pad_size_multiplier", 0),
        ("rollout_batch_size", gbs + 1),
    ):
        invalid = copy.copy(args)
        setattr(invalid, key, value)
        with pytest.raises((AssertionError, ValueError)):
            recipe._check_args(invalid)
