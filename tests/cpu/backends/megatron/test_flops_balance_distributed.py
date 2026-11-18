# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real CPU/Gloo collectives for exact-plan execution and peer failures."""

import copy
import importlib.util
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous, native):
    from relax.utils.data.flops_balancing import STEP_PLANS_FIELD, encode_schedule, restore_schedule

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=45))
    patches = pytest.MonkeyPatch()
    try:
        if native:
            from megatron.core import parallel_state as mpu

            from relax.backends.megatron import data

            mpu.initialize_model_parallel(tensor_model_parallel_size=2)
        else:
            from tests.cpu.backends.megatron.test_data_vpp import _load_data_module

            data = _load_data_module(patches)
            mpu = data.mpu
            tp_groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
            dp_groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
            mpu.get_model_parallel_group = lambda: tp_groups[rank // 2]
            mpu.get_data_parallel_group = lambda **_: dp_groups[rank % 2]
            mpu.get_data_parallel_world_size = lambda **_: 2
            mpu.get_data_parallel_rank = lambda **_: rank // 2
            mpu.get_virtual_pipeline_model_parallel_world_size = lambda: None
            mpu.get_context_parallel_world_size = lambda: 1
        patches.setattr(data.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
        args = SimpleNamespace(
            balance_by_flops=True, global_batch_size=6, use_dynamic_batch_size=True, max_tokens_per_gpu=12
        )
        dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
        lengths = [9, 6, 6, 3, 3, 8]
        metadata = {i: dict(total_lengths=n, flops_cost=n + n * n) for i, n in enumerate(lengths)}
        geometry = dict(version=4, capacity=12, microbatch_group_size=1, context_parallel_size=1, pipeline_stages=1)
        ranks, updates = encode_schedule(list(range(6)), metadata, geometry, 2, 6, ("p", "actor_train", 0))
        ids = ranks[dp_rank]
        rollout = dict(total_lengths=[lengths[i] for i in ids], ids=ids)
        meta = SimpleNamespace(global_indexes=ids, get_all_custom_meta=lambda: [updates[i] for i in ids])
        restore_schedule(meta, rollout)
        iterators, counts = data.get_data_iterator(args, [], rollout)
        assert counts == [2]
        expected = [[[1, 3], [4, 5]], [[2], [0]]][dp_rank]
        assert [iterators[0].get_next(["ids"])["ids"] for _ in range(2)] == expected
        assert rollout[data.ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY] == [6]
        assert len(ids) == (4 if dp_rank == 0 else 2)
        bad = copy.deepcopy(rollout)
        if rank == 0:
            bad[STEP_PLANS_FIELD][0]["indices"] = [[0, 0]]
        with pytest.raises(ValueError, match="Invalid FLOPs schedule"):
            data._get_synchronised_flops_schedule(args, bad)
        bad = copy.deepcopy(rollout)
        if rank == 0:
            bad[STEP_PLANS_FIELD][0]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="differs across ranks"):
            data._get_synchronised_flops_schedule(args, bad)
    finally:
        if native:
            mpu.destroy_model_parallel()
        patches.undo()
        dist.destroy_process_group()


def test_global_plan_real_gloo_groups_and_peer_failures(tmp_path):
    # Stub only the Megatron group accessors; collectives run in four OS
    # processes with orthogonal TP2/DP2 groups and the production iterator.
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}", False), nprocs=4, join=True)


@pytest.mark.skipif(importlib.util.find_spec("megatron") is None, reason="requires Megatron Core")
def test_native_megatron_iterator_and_orthogonal_groups(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}", True), nprocs=4, join=True)
