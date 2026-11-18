# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import multiprocessing
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from relax.utils.data.flops_balancing import (
    STEP_PLANS_FIELD,
    FlopsWorkload,
    build_flops_schedule,
    encode_schedule,
    get_dp_packing_geometry,
    restore_schedule,
)
from tests.cpu.backends.megatron.test_data_vpp import _load_data_module
from tests.cpu.utils.test_flops_counter import FLOPS_TEST_CONFIGS


def _setup(monkeypatch, *, cp=1, vpp=1, rank=0, peer_error=False, real_collectives=False):
    data = _load_data_module(monkeypatch)
    group = data.dist.group.WORLD if real_collectives else object()
    data.mpu = SimpleNamespace(
        get_data_parallel_world_size=lambda **_: 2,
        get_data_parallel_rank=lambda **_: rank,
        get_data_parallel_group=lambda **_: group,
        get_model_parallel_group=lambda: group,
        get_context_parallel_world_size=lambda: cp,
        get_tensor_model_parallel_world_size=lambda: 2,
        get_pipeline_model_parallel_world_size=lambda: 2 if vpp > 1 else 1,
        get_virtual_pipeline_model_parallel_world_size=lambda: vpp,
    )
    if vpp > 1:
        utils = ModuleType("megatron.core.utils")
        utils.get_model_config = lambda _: SimpleNamespace(microbatch_group_size_per_vp_stage=4)
        monkeypatch.setitem(sys.modules, "megatron.core.utils", utils)
    monkeypatch.setattr(data.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    reductions = []

    def agree(values, op, group):
        reductions.append(values.tolist())
        if peer_error and len(values) == 3:
            values[0] = 1

    if not real_collectives:
        monkeypatch.setattr(data.dist, "all_reduce", agree)
    args = SimpleNamespace(
        global_batch_size=8,
        balance_data=True,
        balance_by_flops=True,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=20,
        sft_packing_strategy="balanced",
        qkv_format="thd",
        dynamic_context_parallel=False,
        loss_type="sft",
        is_vl_model=True,
        context_parallel_size=cp,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2 if vpp > 1 else 1,
        flops_microbatch_group_size=4 if vpp > 1 else 1,
        flops_workload=FlopsWorkload.from_config(FLOPS_TEST_CONFIGS["qwen3_dense"]["config"]),
    )
    return data, args, reductions


def _planned_rollout(args, batches, rank=0):
    combined = {
        "total_lengths": [],
        STEP_PLANS_FIELD: [],
        "rollout_mini_local_sample_counts": [],
        "rollout_mini_global_sample_counts": [],
    }
    geometry = get_dp_packing_geometry(args, "train")
    for step, lengths in enumerate(batches):
        metadata = {i: {"total_lengths": length} for i, length in enumerate(lengths)}
        assignments, updates = encode_schedule(
            list(metadata), metadata, args.flops_workload, geometry, 2, len(lengths), ("test", "train", step)
        )
        rows = assignments[rank]
        batch_meta = SimpleNamespace(global_indexes=rows, get_all_custom_meta=lambda: [updates[i] for i in rows])
        rollout = {"total_lengths": [lengths[i] for i in rows]}
        restore_schedule(batch_meta, rollout)
        for key in combined:
            combined[key].extend(rollout[key])
    return combined


def test_flops_iterator_keeps_mini_boundaries_and_vpp_dummy_alignment(monkeypatch):
    data, args, reductions = _setup(monkeypatch, vpp=2)
    rollout = _planned_rollout(args, [[6, 6, 6, 5, 10, 11], [4, 2, 8, 2, 3, 12]])
    lengths = rollout["total_lengths"]
    boundary = rollout["rollout_mini_local_sample_counts"][0]
    iterators, counts = data.get_data_iterator(args, [None, None], rollout)
    assert counts == [4, 4]
    assert reductions[:2] == [[0, 2, -2], [0, 2, -2]]
    assert len(iterators) == 2
    for iterator in iterators:
        real = []
        for offset, indexes in enumerate(iterator.micro_batch_indices):
            assert all((index < boundary) == (offset < 4) for index in indexes)
            assert sum(lengths[index] for index in indexes) <= 20
            if offset not in iterator.dummy_micro_batch_offsets:
                real.extend(indexes)
        assert sorted(real) == list(range(len(lengths)))
        assert iterator.dummy_micro_batch_offsets


def test_flops_iterator_cp_uses_physical_capacity_and_logical_cost(monkeypatch):
    data, args, _ = _setup(monkeypatch, cp=4)
    args.global_batch_size = 4
    args.max_tokens_per_gpu = 12
    args.dynamic_context_parallel = True
    rollout = _planned_rollout(args, [[17, 17, 17, 17]])
    iterators, counts = data.get_data_iterator(args, [], rollout)
    # TP2/CP4 pads each document to 32: two documents exceed 12*4.
    assert counts == [2]
    assert sorted(iterators[0].micro_batch_indices) == [[0], [1]]
    assert rollout["total_lengths"] == [17, 17]


@pytest.mark.parametrize("peer_error", [False, True])
def test_flops_iterator_rejects_invalid_plan_collectively(monkeypatch, peer_error):
    data, args, reductions = _setup(monkeypatch, peer_error=peer_error)
    rollout = _planned_rollout(args, [[2, 3, 4, 5]])
    if not peer_error:
        rollout[STEP_PLANS_FIELD][0]["geometry"]["capacity"] += 1
    with pytest.raises(ValueError, match="Invalid FLOPs schedule on a training rank"):
        data.get_data_iterator(args, [], rollout)
    assert len(reductions) == 2


def test_flops_forward_and_train_budgets_reset_without_losing_rows(monkeypatch):
    data, args, _ = _setup(monkeypatch)
    lengths = [2, 3, 5, 6]
    rollout = {"total_lengths": lengths, "row_ids": list(range(4))}
    training, train_count = data.get_data_iterator(args, [], copy.deepcopy(rollout), max_tokens_per_gpu=6)
    forward, forward_count = data.get_data_iterator(args, [], copy.deepcopy(rollout), max_tokens_per_gpu=20)
    assert train_count[0] > forward_count[0]
    for iterator, count in ((forward[0], forward_count[0]), (training[0], train_count[0])):
        for _ in range(2):
            iterator.reset()
            restored = [None] * len(lengths)
            for offset in range(count):
                batch = iterator.get_next(["row_ids"])
                if not batch.get("__is_dummy__", False):
                    for index, value in zip(iterator.micro_batch_indices[offset], batch["row_ids"], strict=True):
                        restored[index] = value
            assert restored == list(range(4))


@pytest.mark.parametrize("microbatch_group_size", [1, 8])
@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
def test_flops_full_batch_loss_gradients_and_fp32_update(monkeypatch, moe, per_token, microbatch_group_size):
    from relax.utils.data.data import get_first_fit_partitions
    from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions
    from relax.utils.sft_utils import align_loss_mask_for_sft
    from tests.cpu.backends.megatron.test_score_centering_loss import _backend

    data, _, _ = _setup(monkeypatch)
    loss_module, _ = _backend(monkeypatch)
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_world_size", lambda **_: 2, raising=False)

    def log_probs(logits, tokens, *_, **__):
        return -torch.nn.functional.cross_entropy(logits, tokens, reduction="none").unsqueeze(-1), None

    monkeypatch.setattr(loss_module, "calculate_log_probs_and_entropy", log_probs)
    monkeypatch.setattr(loss_module, "validate_opd_topk_gather", lambda *_: None)
    rng = torch.Generator().manual_seed(16)
    # Nine physical rows represent eight logical samples: the first two
    # are context rows of one identity, as in synchronous agentic RL.
    lengths = [2, 3, 4, 5, 6, 7, 8, 9, 60]
    tokens = [torch.randint(0, 9, (n,), generator=rng) for n in lengths]
    features = [torch.randn(n, 4, generator=rng) for n in lengths]
    masks = [align_loss_mask_for_sft(torch.ones(n, dtype=torch.int64)) for n in lengths]
    masks[0].zero_()
    masks[3][1] = 0
    sample_denoms = [mask.sum().item() for mask in masks]
    sample_denoms[0] = sample_denoms[1] = sample_denoms[0] + sample_denoms[1]
    initial = torch.randn(2 if moe else 1, 4, 9, generator=rng)
    workload = FlopsWorkload.from_config(FLOPS_TEST_CONFIGS["qwen3_moe" if moe else "qwen3_dense"]["config"])
    costs = workload(lengths)
    total_tokens = sum(mask.sum().item() for mask in masks)
    args = SimpleNamespace(
        loss_type="sft",
        calculate_per_token_loss=per_token,
        global_batch_size=len(lengths) - 1,
        qkv_format="thd",
        allgather_cp=False,
        rollout_temperature=1.0,
        log_probs_chunk_size=-1,
        recompute_loss_function=False,
    )

    def run(enabled):
        weight = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.SGD([weight], lr=0.1, momentum=0.9)
        if enabled:
            schedule, dummy_offsets = build_flops_schedule(lengths, costs, 10, 2, microbatch_group_size)
            ranks = [
                [i for j, group in enumerate(groups) if j not in dummy for i in group]
                for groups, dummy in zip(schedule, dummy_offsets, strict=True)
            ]
            count = len(schedule[0])
        else:
            ranks = get_seqlen_balanced_partitions(lengths, 2, equal_size=False)
            count = max(len(get_first_fit_partitions([lengths[i] for i in rank], 10)) for rank in ranks)
            count = (count + microbatch_group_size - 1) // microbatch_group_size * microbatch_group_size
        objective = torch.zeros(())
        saw_dummy = saw_unequal_microbatches = False
        for dp_rank, rank in enumerate(ranks):
            local_lengths = [lengths[i] for i in rank]
            if enabled:
                positions = {i: j for j, i in enumerate(rank)}
                partitions = [[positions[i] for i in group] for group in schedule[dp_rank]]
                dummy = dummy_offsets[dp_rank]
            else:
                partitions, dummy = data._get_seqlen_partitions_with_dummy_padding(local_lengths, count, 10)
            saw_dummy |= bool(dummy)
            saw_unequal_microbatches |= len({len(group) for group in partitions}) > 1
            rollout = {
                "total_lengths": local_lengths,
                "response_lengths": local_lengths,
                "unconcat_tokens": [tokens[i] for i in rank],
                "loss_masks": [masks[i] for i in rank],
                "features": [features[i] for i in rank],
                "sample_index_mask_sums": [sample_denoms[i] for i in rank],
            }
            iterator = data.DataIterator(rollout, micro_batch_indices=partitions, dummy_micro_batch_offsets=dummy)
            for _ in partitions:
                batch = iterator.get_next(list(rollout))
                inputs = torch.cat(batch.pop("features"))
                routes = (inputs[:, 0] > 0).long() if moe else torch.zeros(len(inputs), dtype=torch.long)
                logits = torch.bmm(inputs.unsqueeze(1), weight[routes]).squeeze(1).unsqueeze(0)
                value, _, metrics = loss_module.loss_function(args, batch, count, logits)
                if batch.get("__is_dummy__", False):
                    assert metrics["num_tokens"].item() == 0
                    assert torch.count_nonzero(metrics["values"]).item() == 0
                # Emulate Megatron's schedule scaling and the final DP/token
                # normalization after accumulating the entire effective batch.
                value = value / (total_tokens if per_token else count * 2)
                value.backward()
                objective += value.detach()
        assert saw_dummy if microbatch_group_size == 8 else saw_unequal_microbatches
        gradient = weight.grad.clone()
        optimizer.step()
        return objective, gradient, weight.detach() - initial

    baseline = run(False)
    repeated = run(False)
    candidate = run(True)
    for before, again, after in zip(baseline, repeated, candidate, strict=True):
        assert torch.isfinite(before).all() and torch.isfinite(after).all()
        torch.testing.assert_close(before, again, rtol=0, atol=0)
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-7)


def _gloo_bad_plan_worker(rank: int, rendezvous: str) -> None:
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=20)
    )
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            data, args, _ = _setup(monkeypatch, rank=rank, real_collectives=True)
            rollout = _planned_rollout(args, [[2, 3, 4, 5]], rank=rank)
            if rank == 0:
                rollout.pop(STEP_PLANS_FIELD)
            with pytest.raises(ValueError, match="Invalid FLOPs schedule on a training rank"):
                data.get_data_iterator(args, [], rollout)
    finally:
        torch.distributed.destroy_process_group()


def test_flops_bad_plan_exits_on_every_gloo_rank(tmp_path: Path) -> None:
    # A mock collective cannot show that a healthy peer exits instead of hanging.
    context = multiprocessing.get_context("spawn")
    rendezvous = (tmp_path / "rendezvous").as_uri()
    processes = [context.Process(target=_gloo_bad_plan_worker, args=(rank, rendezvous)) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
