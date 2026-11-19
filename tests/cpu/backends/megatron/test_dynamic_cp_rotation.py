# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from datetime import timedelta
from time import monotonic
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _exercise_rank(rank, init_url):
    from relax.backends.megatron import cp_utils
    from relax.backends.megatron.data import DataIterator
    from relax.utils import device

    timeout = timedelta(seconds=45)
    dist.init_process_group("gloo", rank=rank, world_size=6, init_method=init_url, timeout=timeout)
    try:
        dp, cp = divmod(rank, 2)
        cp_groups = [dist.new_group([2 * i, 2 * i + 1], timeout=timeout) for i in range(3)]
        dp_groups = [dist.new_group(list(range(i, 6, 2)), timeout=timeout) for i in range(2)]
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
            cp1_tokens = []
            for mb, indices in enumerate(plan):
                batch = it.get_next(list(rows))
                size = cp_utils.dynamic_cp_split_data(batch, 16)
                chosen = batch["ids"]
                if mb == 1:
                    assert size == 2 and chosen == [ids[i] for i in indices]
                    continue
                assert size == 1
                cp1_tokens.append(sum(batch["total_lengths"]))
                assert all(
                    torch.equal(m, masks[ids.index(i)]) for i, m in zip(chosen, batch["loss_masks"], strict=True)
                )
                result = {
                    "logprobs": [torch.tensor([float(i), i + 0.5]) for i in chosen],
                    "_dcp_meta": dict(
                        dynamic_cp_size=1,
                        dynamic_cp_rank=0,
                        total_lengths=batch["total_lengths"],
                        response_lengths=[2] * len(chosen),
                        partition_order=batch["_dcp_partition_order"],
                    ),
                }
                outputs = cp_utils.dynamic_cp_merge_output([result], 2, cp)[0]["logprobs"]
                for actual, i in zip(outputs, indices, strict=True):
                    torch.testing.assert_close(actual, torch.tensor([float(ids[i]), ids[i] + 0.5]), rtol=0, atol=0)

            # Each side owns the heavier partition once in the last two CP1 microbatches.
            assert sorted(cp1_tokens[-2:]) == [6, 12]
            replay = it.reset().get_next(["total_lengths"])
            cp_utils.dynamic_cp_split_data(replay, 16)
            assert sum(replay["total_lengths"]) == cp1_tokens[0]
    finally:
        dist.destroy_process_group()


def test_dynamic_cp_rotation_round_trip(tmp_path):
    """Ref and training agree on ownership, and outputs restore original order
    over DP3 x CP2."""
    deadline = monotonic() + 120
    workers = mp.start_processes(
        _exercise_rank, args=((tmp_path / "gloo").as_uri(),), nprocs=6, start_method="fork", join=False
    )
    try:
        while (remaining := deadline - monotonic()) > 0:
            if workers.join(timeout=remaining):
                return
        pytest.fail("Dynamic CP Gloo test exceeded 120 seconds")
    finally:
        for worker in workers.processes:
            if worker.is_alive():
                worker.kill()
        for worker in workers.processes:
            worker.join(timeout=1)


@pytest.mark.parametrize("prefix", ["ref_", ""])
def test_streaming_forward_preserves_ordinals_and_full_logprobs(monkeypatch, prefix):
    from relax.backends.megatron import actor as actor_module
    from relax.utils.data import stream_dataloader

    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(max_tokens_per_gpu=16, dynamic_context_parallel=True, qkv_format="thd")
    actor.data_system_client = SimpleNamespace(async_put=AsyncMock(), get_meta=Mock())
    mpu = actor_module.mpu
    monkeypatch.setattr(mpu, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_data_parallel_world_size", lambda **_: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda: True)

    stream = []
    for ids in ([9, 4], [8, 3]):
        data = dict(ids=ids, total_lengths=[6, 6], response_lengths=[2, 2])
        stream.append((data, SimpleNamespace(samples=ids)))
    pending = iter(stream)
    monkeypatch.setattr(stream_dataloader, "get_data_from_transfer_queue", lambda *_, **__: next(pending))

    ordinals = []

    def compute_log_prob(iterators, counts, store_prefix):
        batch = iterators[0].get_next(["ids"])
        ordinals.append(batch["replay_micro_batch_index"])
        return {store_prefix + "log_probs": [torch.tensor([i, i + 0.5]) for i in batch["ids"]]}

    actor.compute_log_prob = compute_log_prob
    key = prefix + "log_probs"
    actor._streaming_fwd_putback(0, key, prefix, ["ids"])
    assert ordinals == [0, 1]
    written = []
    for call in actor.data_system_client.async_put.await_args_list:
        ids = call.kwargs["metadata"].samples
        for i, row in zip(ids, call.kwargs["data"][key].unbind(), strict=True):
            written.append((i, row.tolist()))
    assert sorted(written) == [(i, [i, i + 0.5]) for i in (3, 4, 8, 9)]
