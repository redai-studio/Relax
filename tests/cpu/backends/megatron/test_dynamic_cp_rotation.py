# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _exercise_rank(rank, init_url):
    from relax.backends.megatron import cp_utils
    from relax.backends.megatron.data import DataIterator
    from relax.utils import device

    dist.init_process_group("gloo", rank=rank, world_size=6, init_method=init_url)
    try:
        dp, cp = divmod(rank, 2)
        cp_groups = [dist.new_group([2 * i, 2 * i + 1]) for i in range(3)]
        dp_groups = [dist.new_group(list(range(i, 6, 2))) for i in range(2)]
        groups = SimpleNamespace(
            get_context_parallel_world_size=lambda: 2,
            get_context_parallel_rank=lambda: cp,
            get_context_parallel_group=lambda: cp_groups[dp],
            get_data_parallel_world_size=lambda **kwargs: 3,
            get_data_parallel_group=lambda **kwargs: dp_groups[cp],
        )
        # Only DP2 needs the second microbatch to run at CP2; all DP peers must agree.
        lengths = [6, 6, 6, 6, 17 if dp == 2 else 6, 6]
        ids = list(range(dp * 6, dp * 6 + 6))
        masks = [torch.tensor([i % 2, 1]) for i in ids]
        rows = dict(total_lengths=lengths, ids=ids, loss_masks=masks)
        plan = [[0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 0, 1]]
        it = DataIterator(rows, micro_batch_indices=plan)
        with (
            patch.object(cp_utils, "mpu", groups),
            patch.object(device, "make_current_torch_device", return_value=torch.device("cpu")),
        ):
            passes = []
            for _ in range(2):
                it.reset()
                assignments = []
                for mb, indices in enumerate(plan):
                    batch = it.get_next(list(rows))
                    size = cp_utils.dynamic_cp_split_data(batch, 16)
                    chosen = batch["ids"]
                    if mb == 1:
                        assert size == 2 and chosen == [ids[i] for i in indices]
                        continue
                    assert size == 1
                    assert len(chosen) == (2 if cp == mb % 2 else 1)
                    assert all(
                        torch.equal(m, masks[ids.index(i)]) for i, m in zip(chosen, batch["loss_masks"], strict=True)
                    )
                    assignments.append(chosen)
                    # Unique values on all response positions expose any incorrect sample permutation.
                    values = [torch.tensor([float(i), i + 0.5]) for i in chosen]
                    result = {
                        "logprobs": values,
                        "_dcp_meta": dict(
                            dynamic_cp_size=1,
                            dynamic_cp_rank=0,
                            total_lengths=batch["total_lengths"],
                            response_lengths=[2] * len(chosen),
                            partition_order=batch["_dcp_partition_order"],
                        ),
                    }
                    outputs = cp_utils.dynamic_cp_merge_output([result], 2, cp)[0]["logprobs"]
                    expected = [torch.tensor([float(ids[i]), ids[i] + 0.5]) for i in indices]
                    for actual, wanted in zip(outputs, expected, strict=True):
                        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                passes.append(assignments)
            assert passes[0] == passes[1]
    finally:
        dist.destroy_process_group()


def test_dynamic_cp_rotation_round_trip(tmp_path):
    """Ref and training agree on ownership, and outputs restore original order
    over DP3 x CP2."""
    mp.start_processes(_exercise_rank, args=((tmp_path / "gloo").as_uri(),), nprocs=6, start_method="fork")
