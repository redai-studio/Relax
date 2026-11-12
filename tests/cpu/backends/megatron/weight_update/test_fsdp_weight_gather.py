# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reconstruct FSDP weights with real CPU collectives before TP export."""

import ast
import importlib.util
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from relax.utils.training import tensor_backper


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
        "_maybe_get_cpu_backup",
    }
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "torch": torch,
        "dist": dist,
        "mpu": mpu,
        "is_npu_available": False,
        "torch_memory_saver_preloaded": lambda: False,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _offloader(model):
    path = Path(__file__).resolve().parents[5] / "relax/backends/megatron/weight_update/train_offload.py"
    spec = importlib.util.spec_from_file_location("fsdp_offload_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Exercise the storage lifecycle on CPU; no pinned transfers or CUDA runtime.
    module._PIN_MEMORY = module._NON_BLOCKING = False
    module.device_module = SimpleNamespace(empty_cache=lambda: None)
    module.device_utils = SimpleNamespace(make_current_torch_device=lambda: torch.device("cpu"))
    offloader = module._SelectiveOffloadStrategy([model], None, SimpleNamespace())
    offloader._iter_optimizers = lambda: iter(())
    return offloader


def _check_device_snapshot(param):
    # TP metadata helpers are optional Megatron dependencies. Exercise real
    # snapshot allocation/copy/restore on CPU DTensors without loading CUDA.
    tp_helpers = SimpleNamespace(
        copy_tensor_model_parallel_attributes=lambda snapshot, source: None,
        set_defaults_if_not_set_tensor_model_parallel_attributes=lambda snapshot: None,
    )
    source = param.detach().clone()
    expected = source.to_local().clone()
    backuper = tensor_backper.TensorBackuper.create(lambda: [("weight", source)], single_tag=None)
    with patch.dict(sys.modules, {"megatron.core.tensor_parallel": tp_helpers}):
        backuper.backup("actor", on_device=True)
        snapshot = backuper.get("actor")["weight"]
        assert isinstance(snapshot, DTensor)
        assert snapshot.device_mesh == source.device_mesh
        assert snapshot.placements == source.placements
        assert snapshot.shape == source.shape
        assert snapshot.to_local().shape == expected.shape
        torch.testing.assert_close(snapshot.to_local(), expected)
        source.to_local().add_(1)
        torch.testing.assert_close(snapshot.to_local(), expected)
        backuper.backup("actor", on_device=True)
        assert backuper.get("actor")["weight"] is snapshot
        source.to_local().zero_()
        backuper.restore("actor")
        torch.testing.assert_close(source.to_local(), expected + 1)


def _fsdp_adapter(model, source_root):
    adapter_type = torch.nn.Module
    if source_root is not None:
        # Use the real adapter class while bypassing its CUDA construction.
        # Its nn.Module base supplies parameter traversal and attribute lookup.
        source = Path(source_root) / "megatron/core/distributed/fsdp/mcore_fsdp_adapter.py"
        tree = ast.parse(source.read_text())
        adapter = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FullyShardedDataParallel"
        )
        namespace = {"torch": torch, "nn": torch.nn, "_BaseDataParallel": torch.nn.Module}
        future = ast.parse("from __future__ import annotations").body[0]
        exec(compile(ast.Module(body=[future, adapter], type_ignores=[]), str(source), "exec"), namespace)
        adapter_type = namespace["FullyShardedDataParallel"]
    wrapper = adapter_type.__new__(adapter_type)
    torch.nn.Module.__init__(wrapper)
    wrapper.module = model
    wrapper.param_and_grad_buffer = model.param_and_grad_buffer
    return wrapper


def _gather_worker(rank, rendezvous, source_root):
    torch.set_num_threads(1)
    tensor_backper.device_module = SimpleNamespace(synchronize=lambda: None)
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
            functions = _gather_functions(mpu)
            gather = functions["_gather_tp_weights"]
            name = "decoder.layers.0.mlp.experts.linear_fc1.weight0" if expert else "decoder.layers.0.weight"
            # Five rows exercise uneven shards; one row leaves rank 1 empty.
            for rows in (5, 1):
                expected = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
                shard_size = (rows + 1) // 2
                local = expected[rank * shard_size : (rank + 1) * shard_size].clone()
                # Parameters are views with nonzero offsets in a flat allocation.
                flat = torch.cat([torch.full((2,), -1.0), local.flatten()])
                local = flat[2:].view(local.shape)
                param = DTensor.from_local(
                    local, mesh, [Shard(0)], run_check=False, shape=expected.shape, stride=expected.stride()
                )
                param.tensor_model_parallel = False
                param.partition_dim = 0
                _check_device_snapshot(param)
                for async_op in (False, True):
                    [actual] = gather([(name, param)], split_sizes={}, async_op=async_op)
                    assert not isinstance(actual, DTensor)
                    torch.testing.assert_close(actual, expected)
                model = torch.nn.Module()
                model.register_parameter("weight", torch.nn.Parameter(param))
                normalized = []
                model._replace_param_with_distributed_if_needed = lambda: normalized.append(True)
                buffer = SimpleNamespace(data=flat)
                model.param_and_grad_buffer = SimpleNamespace(
                    parameter_groups=[SimpleNamespace(model_weight_buffer=buffer)]
                )
                offloader = _offloader(_fsdp_adapter(model, source_root))
                backuper = tensor_backper.TensorBackuper.create(
                    source_getter=lambda: [(name, functions["_maybe_get_cpu_backup"](model.weight))],
                    single_tag="actor",
                )
                cpu_storage = None
                for cycle in range(2):
                    backuper.backup("actor")
                    local_hash = tensor_backper._compute_hash_tensor(local)
                    offloader.offload()
                    assert len(normalized) == cycle + 1
                    assert local.untyped_storage().nbytes() == 0
                    snapshot = backuper.get("actor")[name]
                    assert snapshot is not model.weight
                    assert tensor_backper._compute_hash_tensor(snapshot) == local_hash
                    assert snapshot.device_mesh == param.device_mesh
                    assert snapshot.placements == param.placements
                    if cpu_storage is not None:
                        assert buffer._relax_cpu_data is cpu_storage
                    cpu_storage = buffer._relax_cpu_data
                    [actual] = gather([(name, snapshot)], split_sizes={}, async_op=False)
                    torch.testing.assert_close(actual, expected + cycle)
                    offloader.reload()
                    assert not hasattr(model.weight, "_relax_cpu_offload_data")
                    torch.testing.assert_close(local, (expected + cycle)[rank * shard_size : (rank + 1) * shard_size])
                    local.add_(1)
    finally:
        dist.destroy_process_group()


def _hsdp_gather_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4, timeout=timedelta(seconds=20)
    )
    try:
        dense_mesh = DeviceMesh("cpu", [[0, 1], [2, 3]], mesh_dim_names=("dp_outer", "dp_shard"))
        expert_mesh = DeviceMesh("cpu", [[0, 2], [1, 3]], mesh_dim_names=("dp_outer", "dp_shard"))
        mpu = SimpleNamespace(
            get_data_parallel_group=lambda **kwargs: dist.group.WORLD,
            get_expert_data_parallel_group=lambda: dist.group.WORLD,
        )
        gather = _gather_functions(mpu)["_gather_tp_weights"]
        expected = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        for name, mesh in (("decoder.weight", dense_mesh), ("decoder.mlp.experts.weight0", expert_mesh)):
            # Two outer replicas, each with two uneven shards. Each inner group
            # must reconstruct exactly one copy, including for a different EDP mesh.
            local = expected.chunk(2)[mesh.get_local_rank("dp_shard")].clone()
            param = DTensor.from_local(
                local,
                mesh,
                [Replicate(), Shard(0)],
                run_check=False,
                shape=expected.shape,
                stride=expected.stride(),
            )
            param.tensor_model_parallel = False
            for async_op in (False, True):
                [actual] = gather([(name, param)], split_sizes={}, async_op=async_op)
                torch.testing.assert_close(actual, expected)
    finally:
        dist.destroy_process_group()


def _run_workers(worker, args, nprocs):
    processes = mp.spawn(worker, args=args, nprocs=nprocs, join=False)
    deadline = time.monotonic() + 45
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


@pytest.mark.parametrize("native_adapter", [False, True])
def test_fsdp_export_reconstructs_uneven_and_empty_dense_and_expert_shards(tmp_path, native_adapter):
    # Two ranks are needed to catch missing FSDP gathers and collective-group
    # mismatches that single-rank or mocked gather tests cannot detect.
    source_root = os.environ.get("RELAX_MEGATRON_PATCHED_SOURCE") if native_adapter else None
    if native_adapter and source_root is None:
        pytest.skip("Set RELAX_MEGATRON_PATCHED_SOURCE to test the native FSDP adapter without GPU imports")
    _run_workers(_gather_worker, (str(tmp_path / "rendezvous"), source_root), nprocs=2)


def test_hsdp_export_gathers_shards_without_outer_replicas(tmp_path):
    # Four CPU ranks are necessary to have both a nontrivial replication axis
    # and a nontrivial shard axis; the existing two-rank test cannot cover both.
    _run_workers(_hsdp_gather_worker, (str(tmp_path / "rendezvous"),), nprocs=4)
