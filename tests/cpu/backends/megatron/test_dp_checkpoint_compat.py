# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regressions for legacy DP optimizer bucket checkpoint geometry."""

import functools
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("megatron.core", exc_type=ImportError)

from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedObject, ShardedTensor


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def compat():
    # Avoid backend __init__ and its optional CUDA/DeepEP model imports.
    root = Path(__file__).resolve().parents[4]
    return _load_file("dp_checkpoint_compat", root / "relax/backends/megatron/checkpoint_compat.py")


@pytest.fixture
def legacy():
    path = Path(__file__).with_name("fixtures") / "dp_reshardable_legacy.py"
    return _load_file("legacy_dp_checkpoint", path).sharded_param_state_dp_reshardable


def _parameter(start, end, value=1.0):
    return {
        "param": torch.full((end - start,), value, dtype=torch.bfloat16),
        "exp_avg": torch.full((end - start,), value + 1),
        "exp_avg_sq": torch.full((end - start,), value + 2),
        "step": torch.tensor(7),
        "gbuf_local_start": start,
        "gbuf_local_end": end,
    }


def _optimizer(buckets, *, rank=0, world_size=1):
    dtype = (torch.bfloat16, torch.float32)
    optimizer = SimpleNamespace(
        data_parallel_group=SimpleNamespace(rank=lambda: rank, size=lambda: world_size),
        data_parallel_group_idx=2,
        distributed_optimizer_instance_id=3,
        gbuf_ranges=[{dtype: None}],
        buffers=[
            SimpleNamespace(
                buckets=[
                    SimpleNamespace(grad_data=torch.empty(padded), numel_unpadded=unpadded)
                    for padded, unpadded, _ in buckets
                ]
            )
        ],
    )

    def state():
        # Structural copies only: real optimizer tensors must keep their aliases.
        return {
            "per_bucket_numel": [padded for padded, _, _ in buckets],
            "per_bucket_numel_unpadded": [unpadded for _, unpadded, _ in buckets],
            0: {dtype: [[dict(parameter) for parameter in parameters] for _, _, parameters in buckets]},
        }

    optimizer.get_parameter_state_dp_reshardable = state
    return optimizer, dtype


def _peers(monkeypatch, optimizer, dtype, templates):
    expected = optimizer.data_parallel_group

    def gather(result, local, *, group):
        assert group is expected
        result[expected.rank()] = local
        for rank in range(expected.size()):
            if rank != expected.rank():
                result[rank] = {
                    (0, dtype, index): {
                        tuple(
                            sorted(
                                (key, value.dtype)
                                for key, value in template.items()
                                if isinstance(value, torch.Tensor) and key != "step"
                            )
                        )
                    }
                    for index, template in enumerate(templates)
                }

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)


def test_legacy_tail_padding_is_clipped_without_copying_optimizer_tensors(compat, legacy):
    parameter = _parameter(0, 64)
    optimizer, dtype = _optimizer([(128, 96, [parameter])])
    # The old serializer makes a shard ending at 128 despite global size 96.
    try:
        old = legacy(optimizer, {})[0][dtype][0][-1]["param"]
    except (AssertionError, ValueError):
        pass
    else:
        assert old.global_offset[0] + old.local_shape[0] > old.global_shape[0]
    result = compat._sharded_param_state_dp_reshardable(optimizer, {})
    first, padding = result[0][dtype][0]
    for key in ("param", "exp_avg", "exp_avg_sq"):
        assert isinstance(first[key], ShardedTensor)
        assert first[key].data.data_ptr() == parameter[key].data_ptr()
        torch.testing.assert_close(first[key].data, parameter[key], rtol=0, atol=0)
        assert padding[key].local_shape == (32,)
        assert padding[key].global_offset == (64,)
        assert padding[key].global_shape == (96,)
    assert isinstance(first["step"], LocalNonpersistentObject)
    assert first["step"].unwrap() is parameter["step"]
    assert "step" not in padding
    assert first["padding"].unwrap() is False and padding["padding"].unwrap() is True
    assert result["per_bucket_numel"].replica_id == (3, 0, 0)
    assert isinstance(result["per_bucket_numel"], ShardedObject)


def test_front_and_internal_padding_keep_native_keys_and_ranges(compat):
    optimizer, dtype = _optimizer([(128, 96, [_parameter(8, 16), _parameter(32, 64, 4.0)])])
    state = compat._sharded_param_state_dp_reshardable(optimizer, {})[0][dtype][0]
    assert [(entry["param"].global_offset[0], entry["param"].local_shape[0]) for entry in state] == [
        (0, 8),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 32),
    ]
    assert all(
        entry["param"].key
        == "optimizer.distributed.dp_group_idx_2.gbuf_idx_0.dtype_" + str(dtype) + ".bucket_idx_0.param"
        for entry in state
    )
    assert all(entry["param"].replica_id == (3, 0, 0) for entry in state)


@pytest.mark.parametrize("parameters,expected", [([_parameter(0, 16)], [(64, 16), (80, 16)]), ([], [(64, 32)])])
def test_dp_tail_rank_uses_local_unpadded_boundary(compat, monkeypatch, parameters, expected):
    optimizer, dtype = _optimizer([(128, 96, parameters)], rank=1, world_size=2)
    _peers(monkeypatch, optimizer, dtype, [_parameter(0, 64)])
    state = compat._sharded_param_state_dp_reshardable(optimizer, {})[0][dtype][0]
    assert [(entry["param"].global_offset[0], entry["param"].local_shape[0]) for entry in state] == expected
    assert all(entry["param"].global_shape == (96,) for entry in state)


def test_empty_tail_bucket_preserves_slot_before_nonempty_bucket(compat, monkeypatch):
    parameter = _parameter(0, 16)
    optimizer, dtype = _optimizer([(128, 32, []), (128, 96, [parameter])], rank=1, world_size=2)
    _peers(monkeypatch, optimizer, dtype, [_parameter(0, 32), _parameter(0, 64)])
    state = compat._sharded_param_state_dp_reshardable(optimizer, {})[0][dtype]
    assert len(state) == 2 and len(state[0]) == 1
    empty = state[0][0]
    assert empty["padding"].unwrap() is True
    assert empty["param"].local_shape == (0,) and empty["param"].global_offset == (32,)
    assert empty["param"].data.numel() == 0
    assert state[1][0]["param"].data.data_ptr() == parameter["param"].data_ptr()
    assert ".bucket_idx_1.param" in state[1][0]["param"].key


def test_invalid_peer_schema_is_rejected(compat, monkeypatch):
    optimizer, dtype = _optimizer([(128, 96, [_parameter(0, 64)])], world_size=2)
    incompatible = _parameter(0, 32)
    incompatible["exp_avg"] = incompatible["exp_avg"].double()
    _peers(monkeypatch, optimizer, dtype, [incompatible])
    with pytest.raises(ValueError, match="dtype schemas"):
        compat._sharded_param_state_dp_reshardable(optimizer, {})


def test_installer_recognizes_legacy_source_and_is_idempotent(compat, legacy):
    class Optimizer:
        sharded_param_state_dp_reshardable = legacy

    assert compat._serializer_fingerprint(legacy) == compat._LEGACY_SERIALIZER
    assert compat._patch_dp_reshardable_checkpoint(Optimizer)
    assert Optimizer.sharded_param_state_dp_reshardable is compat._sharded_param_state_dp_reshardable
    assert not compat._patch_dp_reshardable_checkpoint(Optimizer)


def test_installer_keeps_unknown_and_external_wrappers(compat, legacy):
    def newer(self, model_sharded_state_dict, is_loading=False, metadata=None):
        return "native"

    @functools.wraps(legacy)
    def wrapped(*args, **kwargs):
        return legacy(*args, **kwargs)

    for method in (newer, wrapped):
        optimizer_cls = type("Optimizer", (), {"sharded_param_state_dp_reshardable": method})
        assert not compat._patch_dp_reshardable_checkpoint(optimizer_cls)
        assert optimizer_cls.sharded_param_state_dp_reshardable is method


def test_installer_keeps_file_patched_serializer(compat, legacy, tmp_path):
    import subprocess

    path = tmp_path / "fixed.py"
    path.write_text(Path(__file__).with_name("fixtures").joinpath("dp_reshardable_legacy.py").read_text())
    root = Path(__file__).resolve().parents[4]
    subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", str(path)],
        input=(root / "docker/patch/megatron/20261028-dp-reshardable-checkpoint.patch").read_text(),
        text=True,
        capture_output=True,
        check=True,
    )
    fixed = _load_file("fixed_dp_checkpoint", path).sharded_param_state_dp_reshardable
    optimizer_cls = type("Optimizer", (), {"sharded_param_state_dp_reshardable": fixed})
    assert compat._serializer_fingerprint(fixed) == compat._FIXED_SERIALIZER
    assert not compat._patch_dp_reshardable_checkpoint(optimizer_cls)
    assert optimizer_cls.sharded_param_state_dp_reshardable is fixed


def _cpu_dcp_roundtrip_worker(rank, root, directory):
    from datetime import timedelta

    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.strategies.torch import (
        MCoreSavePlanner,
        TorchDistSaveShardedStrategy,
        _replace_state_dict_keys_with_sharded_keys,
        mcore_to_pyt_state_dict,
    )
    from torch.distributed import checkpoint as dcp

    compat = _load_file("worker_dp_compat", Path(root) / "relax/backends/megatron/checkpoint_compat.py")
    directory = Path(directory)

    class CPUSaveStrategy(TorchDistSaveShardedStrategy):
        def save(self, sharded_state_dict, checkpoint_dir):
            shards, _, _ = _replace_state_dict_keys_with_sharded_keys(sharded_state_dict, True)
            dcp.save(
                mcore_to_pyt_state_dict(shards, False),
                storage_writer=dcp.FileSystemWriter(checkpoint_dir),
                planner=MCoreSavePlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
            )

    def state(value):
        buckets = (
            [(128, 32, [_parameter(0, 32, value)]), (128, 96, [_parameter(0, 64, value)])]
            if rank == 0
            else [(128, 32, []), (128, 96, [_parameter(0, 16, value + 3)])]
        )
        optimizer, dtype = _optimizer(buckets, rank=rank, world_size=2)
        optimizer.data_parallel_group = torch.distributed.group.WORLD
        optimizer.distributed_optimizer_instance_id = 0
        return compat._sharded_param_state_dp_reshardable(optimizer, {}), dtype

    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{directory / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        saved, dtype = state(2.0)
        path = directory / "checkpoint"
        dist_checkpointing.save(saved, str(path), sharded_strategy=CPUSaveStrategy("torch_dist", 1))
        template, _ = state(0.0)
        restored = dist_checkpointing.load(template, str(path))[0][dtype]
        assert len(restored) == 2
        if rank == 1:
            assert restored[0][0]["param"].shape == (0,)
        parameter = restored[1][0]
        value = 2.0 if rank == 0 else 5.0
        for key, offset in (("param", 0), ("exp_avg", 1), ("exp_avg_sq", 2)):
            torch.testing.assert_close(parameter[key], torch.full_like(parameter[key], value + offset), rtol=0, atol=0)
        assert parameter["step"].item() == 7
        assert parameter["padding"] is False
    finally:
        torch.distributed.destroy_process_group()


def test_cpu_dcp_roundtrip_keeps_empty_bucket_before_later_parameter(tmp_path):
    from multiprocessing import get_context

    root = Path(__file__).resolve().parents[4]
    (tmp_path / "checkpoint").mkdir()
    # Two CPU ranks are needed to exercise the real schema collective and a
    # zero-length tail shard without inventing multiple local DP ranks.
    context = get_context("spawn")
    processes = [
        context.Process(target=_cpu_dcp_roundtrip_worker, args=(rank, str(root), str(tmp_path))) for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    assert [process.exitcode for process in processes] == [0, 0]
