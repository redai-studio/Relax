# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Production iterator logic with stubbed Megatron metadata, not CUDA proof."""

import sys
import types
from argparse import Namespace

import pytest
import torch

from relax.utils.data.flops_schedule import build_workload_microbatch_plan
from relax.utils.training.flops_counter import SequenceFlopsEstimator
from tests.backends.megatron.test_data_vpp import _load_data_module


def _setup(monkeypatch, *, vpp=False, peer_error=False, dp_only=False):
    module = _load_data_module(monkeypatch)
    for name, value in (
        ("get_data_parallel_world_size", 2),
        ("get_data_parallel_group", object()),
        ("get_virtual_pipeline_model_parallel_world_size", 2 if vpp else None),
        ("get_context_parallel_world_size", 1),
        ("get_tensor_model_parallel_world_size", 1),
    ):
        monkeypatch.setattr(module.mpu, name, lambda *a, _value=value, **k: _value, raising=False)
    monkeypatch.setattr(module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))

    def agree_counts(tensor, **kwargs):
        if tensor.numel() > 1:
            tensor[1:] = 4
            if peer_error:
                tensor[0] = 1

    monkeypatch.setattr(module.dist, "all_reduce", agree_counts)
    if vpp:
        utils = types.ModuleType("megatron.core.utils")
        utils.get_model_config = lambda _: Namespace(microbatch_group_size_per_vp_stage=4)
        monkeypatch.setitem(sys.modules, "megatron.core.utils", utils)
    args = Namespace(
        global_batch_size=12,
        micro_batch_size=1,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=128,
        data_pad_size_multiplier=128,
        balance_data=True,
        flops_microbatch_packing=not dp_only,
        flops_dp_balancing=dp_only,
        _flops_workload_estimator=SequenceFlopsEstimator(10, 1),
    )
    data = {
        "total_lengths": [128, 1, 1, 128, 1, 1],
        "sample_ids": list(range(6)),
        "tokens": [[i] for i in range(6)],
        "loss_masks": [[i + 10] for i in range(6)],
        module.ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY: [3, 3],
    }
    return module, args, data


def test_flops_iterator_keeps_step_boundaries_fields_and_dummy_marks(monkeypatch):
    module, args, data = _setup(monkeypatch)
    iterators, counts = module.get_data_iterator(args, object(), data)
    assert counts == [4, 4]
    for step in range(2):
        real_ids, dummy_count = [], 0
        for _ in range(4):
            batch = iterators[0].get_next(["sample_ids", "tokens", "loss_masks", "total_lengths"])
            assert batch["tokens"] == [[i] for i in batch["sample_ids"]]
            assert batch["loss_masks"] == [[i + 10] for i in batch["sample_ids"]]
            assert sum(batch["total_lengths"]) <= 128
            if batch.get("__is_dummy__", False):
                dummy_count += 1
            else:
                real_ids.extend(batch["sample_ids"])
        assert sorted(real_ids) == list(range(step * 3, step * 3 + 3))
        assert dummy_count == 1


def test_flops_iterator_vpp_iterators_share_plan_and_independent_offsets(monkeypatch):
    module, args, data = _setup(monkeypatch, vpp=True)
    iterators, counts = module.get_data_iterator(args, [object(), object()], data)
    assert counts == [4, 4] and len(iterators) == 2
    assert iterators[0].get_next(["tokens"]) == iterators[1].get_next(["tokens"])
    assert iterators[0].offset == iterators[1].offset == 1


@pytest.mark.parametrize("peer_error", [False, True])
def test_flops_iterator_coordinates_local_or_peer_planning_error(monkeypatch, peer_error):
    module, args, data = _setup(monkeypatch, peer_error=peer_error)
    if not peer_error:
        data["total_lengths"][0] = 129
    with pytest.raises(RuntimeError, match="planning failed"):
        module.get_data_iterator(args, object(), data)


def test_dp_only_coordinates_unmodeled_media_error(monkeypatch):
    module, args, data = _setup(monkeypatch, dp_only=True)
    data["multimodal_train_inputs"] = [{"pixel_values": object()}] + [None] * 5
    with pytest.raises(RuntimeError, match="planning failed"):
        module.get_data_iterator(args, object(), data)


@pytest.mark.parametrize(
    "tp,cp,unsplit,allgather",
    [(1, 1, False, False), (2, 2, False, False), (2, 2, True, False), (2, 2, True, True), (2, 2, False, True)],
)
def test_flops_plan_bounds_real_get_batch_padding_and_preserves_dummy(monkeypatch, tp, cp, unsplit, allgather):
    # Exercise actual CPU get_batch packing; only Megatron metadata and pinning
    # are replaced. This does not execute distributed CP/TP kernels.
    module = _load_data_module(monkeypatch)
    monkeypatch.setattr(module.mpu, "get_tensor_model_parallel_world_size", lambda: tp, raising=False)
    monkeypatch.setattr(module.mpu, "get_context_parallel_world_size", lambda: cp, raising=False)
    monkeypatch.setattr(module.mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(module, "get_args", lambda: Namespace(uses_unsplit_forward=unsplit))
    monkeypatch.setattr(module, "PackedSeqParams", Namespace)
    monkeypatch.setattr(module, "pin_tensors", lambda batch: batch)
    lengths = [129, 130, 7]
    plan = build_workload_microbatch_plan(
        lengths,
        SequenceFlopsEstimator(10, 1),
        256,
        tp_size=tp,
        cp_size=cp,
        is_vl_model=unsplit,
        allgather_cp=allgather,
        num_microbatches=4,
    )
    data = dict(
        total_lengths=lengths,
        response_lengths=lengths,
        tokens=[torch.arange(length) for length in lengths],
        loss_masks=[torch.ones(length, dtype=torch.int) for length in lengths],
    )
    iterator = module.DataIterator(
        data, micro_batch_indices=plan.partitions, dummy_micro_batch_offsets=set(plan.dummy_offsets)
    )
    keys = ["tokens", "loss_masks", "total_lengths", "response_lengths"]
    for offset, group in enumerate(plan.partitions):
        batch = module.get_batch(iterator, keys, allgather_cp=allgather, pack_device=torch.device("cpu"))
        assert batch["tokens"].shape[-1] <= 256
        assert bool(batch.get("__is_dummy__", False)) == (offset in plan.dummy_offsets)
        for raw, index in zip(batch["unconcat_tokens"], group, strict=True):
            assert torch.equal(raw, data["tokens"][index])
        if unsplit and cp > 1:
            assert int(batch["vlm_packed_seq_params"].cu_seqlens_q[-1]) // cp <= 256
        # PrepackedBatch fast path must carry the exact same dummy marker.
        assert module.get_batch(iter([(batch, None)]), keys) is batch
