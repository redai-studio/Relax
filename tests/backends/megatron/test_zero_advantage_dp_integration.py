"""CPU integration of pruning with the production DP microbatch iterator."""

import importlib
import runpy
import sys
import threading
import types
from argparse import Namespace
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
filter_rows = runpy.run_path(str(ROOT / "relax/backends/megatron/zero_advantage.py"))["filter_zero_advantage_rows"]


def _load_data_module(monkeypatch):
    # Stub only Megatron imports; scheduling, partitioning and iteration stay real.
    names = (
        "megatron",
        "megatron.core",
        "megatron.core.mpu",
        "megatron.core.packed_seq_params",
        "megatron.training",
        "megatron.training.global_vars",
        "relax.utils.tracking_utils",
    )
    modules = {name: types.ModuleType(name) for name in names}
    modules["megatron.core"].mpu = modules["megatron.core.mpu"]
    modules["megatron.core.packed_seq_params"].PackedSeqParams = type("PackedSeqParams", (), {})
    modules["megatron.training.global_vars"].get_args = lambda: None
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "relax.backends.megatron.data", raising=False)
    return importlib.import_module("relax.backends.megatron.data")


class _CPUCollectives:
    """Two concurrent ranks with actual elementwise MAX and all-gather
    semantics."""

    ReduceOp = torch.distributed.ReduceOp

    def __init__(self):
        self.local = threading.local()
        self.barrier = threading.Barrier(2, timeout=10)
        self.lock = threading.Lock()
        self.rounds = {}

    def _exchange(self, tensor, kind):
        assert tensor.device.type == "cpu"
        ordinal = getattr(self.local, "ordinal", 0)
        self.local.ordinal = ordinal + 1
        with self.lock:
            round_data = self.rounds.setdefault(ordinal, {"kind": kind, "values": {}})
            assert round_data["kind"] == kind
            round_data["values"][self.local.rank] = tensor.clone()
        self.barrier.wait()
        return [round_data["values"][rank] for rank in range(2)]

    def all_reduce(self, tensor, op=None, group=None):
        assert op == self.ReduceOp.MAX
        values = self._exchange(tensor, "max")
        tensor.copy_(torch.stack(values).amax(dim=0))
        self.barrier.wait()

    def all_gather(self, outputs, tensor, group=None):
        values = self._exchange(tensor, "gather")
        for output, value in zip(outputs, values, strict=True):
            output.copy_(value)
        self.barrier.wait()

    def get_world_size(self, group=None):
        return 2

    def get_rank(self, group=None):
        return self.local.rank


def _batch(rank, active_rows):
    # Two turns per trajectory, including fully pruned trajectories.
    count = 8
    return {
        "total_lengths": [8] * 6 + [2, 1],
        "rewards": [1.0 if index < active_rows else 0.0 for index in range(count)],
        "sample_indices": [rank * 100 + index // 2 for index in range(count)],
        "sample_index_mask_sums": [2] * count,
        "tokens": [torch.tensor([rank * 1000 + index]) for index in range(count)],
        "loss_masks": [torch.tensor([1.0]) for _ in range(count)],
        "rollout_mini_local_sample_counts": [count],
        "rollout_mini_global_sample_counts": [592],
        "rollout_mini_prompt_group_counts": [74],
    }


@pytest.mark.parametrize("secondary_active_rows", [0, 2], ids=["zero-local-dp", "unequal-active-dp"])
@pytest.mark.parametrize("token_budget", [8, 16], ids=["train-budget", "logprob-budget"])
def test_filtered_rows_follow_real_dp_max_and_dummy_schedule(monkeypatch, secondary_active_rows, token_budget):
    data_module = _load_data_module(monkeypatch)
    collectives = _CPUCollectives()
    monkeypatch.setattr(data_module, "dist", collectives)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_world_size", lambda **kwargs: 2, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group", lambda: "cpu-dp", raising=False)
    monkeypatch.setattr(data_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(data_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    args = Namespace(
        advantage_estimator="grpo",
        loss_type="policy_loss",
        compute_advantages_and_returns=True,
        use_agentic_rollout=True,
        use_dynamic_batch_size=True,
        num_iters_per_train_update=1,
        entropy_coef=0.0,
        kl_coef=0.0,
        use_kl_loss=True,
        kl_loss_coef=0.02,
        kl_loss_type="k3",
        global_batch_size=592,
        balance_data=True,
        dynamic_context_parallel=True,
        max_tokens_per_gpu=8,
        micro_batch_size=1,
    )
    originals = [_batch(0, 6), _batch(1, secondary_active_rows)]
    filtered = [filter_rows(args, batch, reference_matches_actor=True)[0] for batch in originals]
    assert [len(batch["rewards"]) for batch in filtered] == [6, max(secondary_active_rows, 1)]
    if secondary_active_rows == 0:
        assert filtered[1]["tokens"][0].item() == 1007  # shortest zero row retained

    def run_rank(rank):
        collectives.local.rank = rank
        return data_module.get_data_iterator(args, object(), filtered[rank], max_tokens_per_gpu=token_budget)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_rank, rank) for rank in range(2)]
        outputs = [future.result(timeout=20) for future in futures]
    schedules = [output[1] for output in outputs]
    expected_microbatches = 48 // token_budget
    assert schedules == [[expected_microbatches], [expected_microbatches]]
    assert len(schedules[0]) == len(schedules[1]) == 1  # one optimizer/scheduler update
    assert args.global_batch_size == 592
    # Production scheduling uses two MAX reductions; diagnostic gathers are not required.
    assert [round_data["kind"] for round_data in collectives.rounds.values()] == ["max", "max"]
    local_microbatches = collectives.rounds[1]["values"]
    assert local_microbatches[0].item() > local_microbatches[1].item()

    for rank, (iterators, schedule) in enumerate(outputs):
        batch = filtered[rank]
        assert batch["rollout_mini_global_sample_counts"] == [592]
        assert batch["rollout_mini_prompt_group_counts"] == [74]
        assert batch["sample_index_mask_sums"] == [2] * len(batch["rewards"])
        iterator = iterators[0]
        keys = ("tokens", "rewards", "sample_indices", "sample_index_mask_sums", "loss_masks")
        emitted = [iterator.get_next(keys) for _ in range(schedule[0])]
        real_batches = [micro for micro in emitted if not micro.get("__is_dummy__", False)]
        dummy_batches = [micro for micro in emitted if micro.get("__is_dummy__", False)]
        seen = [token.item() for micro in real_batches for token in micro["tokens"]]
        expected = [token.item() for token in batch["tokens"]]
        assert Counter(seen) == Counter(expected)  # every retained turn exactly once
        active_seen = [
            token.item()
            for micro in real_batches
            for token, reward in zip(micro["tokens"], micro["rewards"], strict=True)
            if reward
        ]
        active_expected = [
            token.item()
            for token, reward in zip(originals[rank]["tokens"], originals[rank]["rewards"], strict=True)
            if reward
        ]
        assert Counter(active_seen) == Counter(active_expected)  # no lost or repeated gradient rows
        assert len(dummy_batches) == max(0, schedule[0] - len(batch["rewards"]))
        assert [micro["replay_micro_batch_index"] for micro in emitted] == list(range(schedule[0]))
        if rank == 1:
            assert dummy_batches  # exercises production padding and reordered dummy offsets
        iterator.reset()
        replayed = [iterator.get_next(keys) for _ in range(schedule[0])]
        assert [micro.get("__is_dummy__", False) for micro in replayed] == [
            micro.get("__is_dummy__", False) for micro in emitted
        ]
