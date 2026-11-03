# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Native Megatron iterator and orthogonal process-group checks on CPU/Gloo."""

import importlib.util
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


pytestmark = pytest.mark.skipif(importlib.util.find_spec("megatron") is None, reason="requires Megatron Core")


def _worker(rank, rendezvous, output):
    from megatron.core import parallel_state as mpu

    from relax.backends.megatron import data
    from relax.utils import misc
    from relax.utils.training.microbatch_cost import register_sample_cost_estimator

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=45))
    try:
        mpu.initialize_model_parallel(tensor_model_parallel_size=2)
        data.device_utils.make_current_torch_device = lambda: torch.device("cpu")
        misc.get_hf_config = lambda _: SimpleNamespace(model_type="distributed_cost_test")
        register_sample_cost_estimator("distributed_cost_test", lambda config, batch, args: batch["test_costs"])
        args = SimpleNamespace(
            microbatch_balance="flops",
            global_batch_size=8,
            use_dynamic_batch_size=True,
            max_tokens_per_gpu=16,
            hf_checkpoint="unused",
            qkv_format="thd",
        )
        dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
        lengths = [8, 8] if dp_rank == 0 else [16] * 6
        costs = [100, 1] if dp_rank == 0 else [10, 20, 30, 40, 50, 60]
        rollout = {
            "total_lengths": lengths,
            "ids": list(range(len(lengths))),
            "test_costs": costs,
            data.ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY: [len(lengths)],
        }
        iterators, counts = data.get_data_iterator(args, [], rollout)
        iterator = iterators[0]
        seen = []
        for _ in range(counts[0]):
            batch = iterator.get_next(["ids", "total_lengths"])
            assert sum(batch["total_lengths"]) <= 16
            if not batch.get("__is_dummy__", False):
                seen.extend(batch["ids"])
        assert sorted(seen) == list(range(len(lengths)))
        assert counts == [6]
        assert len(iterator.dummy_micro_batch_offsets) == (4 if dp_rank == 0 else 0)

        # Only rank 0 fails. TP then DP propagation must make all four ranks
        # exit the same call, rather than allowing peers to enter training.
        bad = {"total_lengths": [8], "test_costs": [-1 if rank == 0 else 1]}
        with pytest.raises(ValueError, match="FLOPs sample cost estimation failed"):
            data._get_synchronised_sample_costs(args, bad)
        Path(output, f"rank-{rank}.json").write_text(json.dumps({"counts": counts, "peer_error": True}))
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def test_native_iterator_dp_alignment_dummy_and_model_parallel_failure(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=4, join=True)
    assert len(list(tmp_path.glob("rank-*.json"))) == 4
