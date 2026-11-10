# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise DCS dispatch with Bridge's TP-local shard contract and real Gloo.

The small converter isolates model dependencies; the production DCS handler
must keep both TP ranks in its collectives. Model-specific mapping tests live
in test_bridge_ple_shards.py.
"""

import time
from argparse import Namespace
from datetime import timedelta
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing

from relax.distributed.checkpoint_service.backends import device_direct
from relax.distributed.checkpoint_service.backends.device_direct import DeviceDirectBackend
from relax.models.deepseek_v4.weight_sync import split_fused_weight_pairs


PLE_NAME = "module.module.decoder.layers.0.per_layer_embedding.ple_embedding.ngram_embedding.weight"


def _backend(converter, gather, send_hf, send_raw, *, source=True, skip_base=False, control_group=None):
    backend = DeviceDirectBackend.__new__(DeviceDirectBackend)
    backend.args = Namespace(update_weight_buffer_size=8)
    backend._is_pp_src_rank = source
    backend._use_bridge = True
    backend._bridge_converter = converter
    backend._megatron = SimpleNamespace(
        all_gather_param=gather,
        mpu=SimpleNamespace(
            get_tensor_and_data_parallel_group=lambda **kw: control_group or dist.group.WORLD,
        ),
    )
    backend._lora_enabled = False
    backend._lora_merge_mode = False
    backend._lora_skip_rollout_base = skip_base
    backend._rollout_bucket_splitter = None
    backend._update_bucket_weights_from_distributed = send_hf
    backend._update_bucket_weights_from_distributed_for_actor_fwd_ref = send_raw
    return backend


def _collective_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    try:
        # HF-only (Hybrid), raw+HF, raw-only, and adapter updates after base sync.
        for rollout_only, actor_fwd_only, skip_base in [
            (True, False, False),
            (False, False, False),
            (False, True, False),
            (True, False, True),
            (False, False, True),
        ]:
            for step in range(2):
                table = torch.arange(24, dtype=torch.float32).reshape(12, 2) + 100 * step
                local_rows = table[rank * 6 : (rank + 1) * 6].clone()
                emitted, raw, shard_events, gathered = [], [], [], []

                def gather(args, name, param):
                    if name == PLE_NAME:
                        assert not rollout_only, "Hybrid must not materialize the full PLE table"
                    gathered.append(name)
                    parts = [torch.empty_like(param) for _ in range(2)]
                    dist.all_gather(parts, param, group=dist.group.WORLD)
                    return torch.cat(parts)

                def shards(name, param):
                    assert name == PLE_NAME
                    assert param.data_ptr() == local_rows.data_ptr()
                    torch.testing.assert_close(param, table[rank * 6 : (rank + 1) * 6])
                    for index in range(3):
                        if rank == 0:
                            # Each shard must reach the receiver before exporting the next.
                            assert shard_events == ["sent"] * index
                        start, end = index * 4, (index + 1) * 4
                        shard = torch.zeros(4, 2)
                        lo, hi = max(start, rank * 6), min(end, (rank + 1) * 6)
                        if hi > lo:
                            shard[lo - start : hi - start] = param[lo - rank * 6 : hi - rank * 6]
                        dist.all_reduce(shard, group=dist.group.WORLD)
                        yield f"hf.ple.shard_{index}.weight", shard

                def convert(name, param):
                    assert name != PLE_NAME, "PLE must use the collective shard iterator"
                    return [(f"hf.{name}", param)]

                def send_hf(weights, pbar=None):
                    assert rank == 0
                    for name, tensor in weights:
                        emitted.append((name, tensor.clone()))
                        if name.startswith("hf.ple."):
                            shard_events.append("sent")

                def send_raw(weights):
                    assert rank == 0
                    raw.extend((name, tensor.clone()) for name, tensor in weights)

                backend = _backend(
                    SimpleNamespace(iter_ple_shards=shards, convert=convert),
                    gather,
                    send_hf,
                    send_raw,
                    source=rank == 0,
                    skip_base=skip_base,
                )
                converted, original, size = [], [], 0
                for name, param in [
                    ("before", torch.tensor([rank + step], dtype=torch.float32)),
                    (PLE_NAME, local_rows),
                    ("after", torch.tensor([rank + step + 10], dtype=torch.float32)),
                ]:
                    size = backend._update_weight_from_distributed(
                        name, param, converted, original, size, rollout_only, actor_fwd_only
                    )
                if rank == 0:
                    if original and not rollout_only:
                        send_raw(original)
                    if converted and not actor_fwd_only:
                        send_hf(converted)
                    if not actor_fwd_only and not skip_base:
                        assert [name for name, _ in emitted] == [
                            "hf.before",
                            *(f"hf.ple.shard_{i}.weight" for i in range(3)),
                            "hf.after",
                        ]
                        for index, (_, tensor) in enumerate(emitted[1:4]):
                            torch.testing.assert_close(tensor, table[index * 4 : (index + 1) * 4])
                        torch.testing.assert_close(emitted[0][1], torch.tensor([step, step + 1.0]))
                        torch.testing.assert_close(emitted[-1][1], torch.tensor([step + 10.0, step + 11.0]))
                    else:
                        assert emitted == []
                    if not rollout_only:
                        assert [name for name, _ in raw] == ["before", PLE_NAME, "after"]
                        torch.testing.assert_close(raw[1][1], table)
                    else:
                        assert raw == []
                else:
                    assert emitted == raw == []
                assert gathered.count(PLE_NAME) == int(not rollout_only)
                dist.barrier(group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()


def _run_workers(worker, tmp_path, nprocs):
    context = multiprocessing.start_processes(
        worker,
        args=((tmp_path / "gloo").as_uri(),),
        nprocs=nprocs,
        join=False,
        start_method="spawn",
    )
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if context.join(timeout=1):
                return
        pytest.fail("DCS PLE collective regression exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


def test_dcs_ple_all_tp_ranks_stream_updated_shards_and_preserve_raw_consumers(tmp_path):
    _run_workers(_collective_worker, tmp_path, 2)


@pytest.fixture
def single_rank_control(monkeypatch):
    monkeypatch.setattr(device_direct.dist, "get_global_rank", lambda group, rank: 0)
    monkeypatch.setattr(device_direct.dist, "broadcast_object_list", lambda *args, **kwargs: None)


def test_dcs_ple_keeps_pending_fused_weights_across_shards(single_rank_control):
    pending = ("model.layers.0.mlp.experts.w1_weight", torch.ones(2))
    sent = []

    def shards(name, param):
        yield "hf.ple.shard_0.weight", param
        assert sent == ["hf.ple.shard_0.weight"]
        yield "hf.ple.shard_1.weight", param + 1

    backend = _backend(
        SimpleNamespace(iter_ple_shards=shards),
        lambda *args: pytest.fail("unexpected full-table gather"),
        lambda weights, pbar=None: sent.extend(name for name, _ in weights),
        lambda weights: pytest.fail("unexpected raw consumer"),
    )
    backend._rollout_bucket_splitter = partial(
        split_fused_weight_pairs, pair_key=lambda name: "pair" if name == pending[0] else None
    )
    converted, original = [pending], []
    size = backend._update_weight_from_distributed(PLE_NAME, torch.ones(2), converted, original, 8, True)
    assert converted == [pending]
    assert size == 8
    assert sent == ["hf.ple.shard_0.weight", "hf.ple.shard_1.weight"]


def test_dcs_ple_closes_export_iterator_when_receiver_fails(single_rank_control):
    state = SimpleNamespace(pp_group=object())
    original_group = state.pp_group

    def shards(name, param):
        state.pp_group = None
        try:
            yield "hf.ple.shard_0.weight", param
            pytest.fail("receiver failure must stop shard export")
        finally:
            state.pp_group = original_group

    def fail_send(weights, pbar=None):
        raise RuntimeError("receiver failed")

    backend = _backend(SimpleNamespace(iter_ple_shards=shards), lambda *args: None, fail_send, lambda weights: None)
    with pytest.raises(RuntimeError, match="receiver failed"):
        backend._update_weight_from_distributed(PLE_NAME, torch.ones(2), [], [], 0, True)
    assert state.pp_group is original_group


def _failure_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=8))
    try:
        groups = [
            dist.new_group(ranks=ranks, backend="gloo", timeout=timedelta(seconds=8)) for ranks in ([0, 1], [2, 3])
        ]
        tp_group = groups[rank // 2]
        # Four ranks are necessary to test DP peers outside the source TP group
        # and a PP-local source whose global rank is not zero. Other PP stages
        # need not own a PLE table or enter the same per-shard control sequence.
        for layout in ["tp2_dp2", "pp2_tp2"]:
            source = 0 if layout == "tp2_dp2" else 2
            control = dist.group.WORLD if layout == "tp2_dp2" else groups[1]
            for phase in ["preflush", "raw", "shard"]:
                if layout == "pp2_tp2" and rank < 2:
                    dist.barrier(group=dist.group.WORLD)
                    continue
                assert (rank == source) == (dist.get_rank(control) == 0)
                table = torch.arange(24, dtype=torch.float32).reshape(12, 2)
                local_rows = table[(rank % 2) * 6 : (rank % 2 + 1) * 6].clone()
                state = SimpleNamespace(exported=[], closed=False)

                def shards(name, param):
                    try:
                        for index in range(3):
                            state.exported.append(index)
                            shard = torch.zeros(4, 2)
                            start = index * 4
                            lo, hi = max(start, rank % 2 * 6), min(start + 4, (rank % 2 + 1) * 6)
                            if hi > lo:
                                shard[lo - start : hi - start] = param[lo - rank % 2 * 6 : hi - rank % 2 * 6]
                            dist.all_reduce(shard, group=tp_group)
                            yield f"hf.ple.shard_{index}.weight", shard
                    finally:
                        state.closed = True

                def gather(args, name, param):
                    assert phase == "raw"
                    parts = [torch.empty_like(param) for _ in range(2)]
                    dist.all_gather(parts, param, group=tp_group)
                    return torch.cat(parts)

                def fail_send(*args, **kwargs):
                    assert rank == source
                    raise ValueError(f"injected {phase} receiver failure")

                backend = _backend(
                    SimpleNamespace(iter_ple_shards=shards),
                    gather,
                    fail_send,
                    fail_send,
                    source=rank == source,
                    control_group=control,
                )
                converted = [("before", torch.ones(2))] if phase == "preflush" else []
                # The peer must receive the source's error before requesting a
                # second shard, rather than failing on Gloo's 8-second timeout.
                with pytest.raises((ValueError, RuntimeError), match=f"injected {phase} receiver failure") as error:
                    backend._update_weight_from_distributed(
                        PLE_NAME, local_rows, converted, [], 0, rollout_only=phase != "raw"
                    )
                if rank == source:
                    assert type(error.value) is ValueError
                assert state.exported == ([0] if phase == "shard" else [])
                assert state.closed == (phase == "shard")
                dist.barrier(group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()


def test_dcs_ple_send_failures_reach_all_stage_ranks_before_next_collective(tmp_path):
    _run_workers(_failure_worker, tmp_path, 4)
