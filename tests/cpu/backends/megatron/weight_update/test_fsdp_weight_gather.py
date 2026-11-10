# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reconstruct FSDP weights with real CPU collectives before TP export."""

import ast
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard


def _gather_functions(mpu):
    # The numerical gather only requires torch; avoid importing optional
    # Megatron/Transformer Engine training backends in this CPU regression.
    path = Path(__file__).resolve().parents[5] / "relax/backends/megatron/weight_update/common.py"
    names = {
        "_allgather_dim0_uneven",
        "fsdp_gather_local_param",
        "_check_fsdp_gathered_shape",
        "_restore_tp_weight",
        "_gather_tp_weights",
    }
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": torch, "dist": dist, "mpu": mpu, "is_npu_available": False}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _gather_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("dp_shard",))
        for expert in (False, True):

            def unexpected_group(*args, **kwargs):
                raise AssertionError("Gather selected the wrong data-parallel group")

            def data_group(*, with_context_parallel):
                assert with_context_parallel
                return dist.group.WORLD

            mpu = SimpleNamespace(
                get_data_parallel_group=unexpected_group if expert else data_group,
                get_expert_data_parallel_group=(lambda: dist.group.WORLD) if expert else unexpected_group,
            )
            gather = _gather_functions(mpu)["_gather_tp_weights"]
            name = "decoder.layers.0.mlp.experts.linear_fc1.weight0" if expert else "decoder.layers.0.weight"
            # Five rows exercise uneven shards; one row leaves rank 1 empty.
            for rows in (5, 1):
                expected = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
                shard_size = (rows + 1) // 2
                local = expected[rank * shard_size : (rank + 1) * shard_size].clone()
                param = DTensor.from_local(
                    local, mesh, [Shard(0)], run_check=False, shape=expected.shape, stride=expected.stride()
                )
                param.tensor_model_parallel = False
                param.partition_dim = 0
                for async_op in (False, True):
                    [actual] = gather([(name, param)], split_sizes={}, async_op=async_op)
                    assert not isinstance(actual, DTensor)
                    torch.testing.assert_close(actual, expected)
    finally:
        dist.destroy_process_group()


def test_fsdp_export_reconstructs_uneven_and_empty_dense_and_expert_shards(tmp_path):
    # Two ranks are needed to catch missing FSDP gathers and collective-group
    # mismatches that single-rank or mocked gather tests cannot detect.
    processes = mp.spawn(_gather_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=False)
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            if processes.join(timeout=max(0.0, deadline - time.monotonic())):
                return
        raise AssertionError("FSDP CPU weight gather timed out")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
