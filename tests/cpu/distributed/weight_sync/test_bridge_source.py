import multiprocessing
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from relax.distributed.weight_sync.source_map import TensorMapping


def _gdn_export_worker(rank, rendezvous, output):
    from contextlib import ExitStack
    from unittest.mock import patch

    import torch
    import torch.distributed as dist
    from megatron.core import mpu

    from relax.backends.megatron.weight_update import hf_weight_iterator_direct as direct
    from relax.backends.megatron.weight_update.adaptive_source import AdaptiveSourceExporter
    from relax.backends.megatron.weight_update.bridge_source import BridgeSourceAdapter
    from relax.backends.megatron.weight_update.common import gdn_component_sizes
    from relax.utils.types import ParamInfo

    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=30))
    try:
        config = SimpleNamespace(
            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=2, linear_value_head_dim=4
        )
        parts, infos, split_sizes = {}, [], {}
        for kind, columns in (("conv1d", 3), ("in_proj", 4)):
            name = f"module.module.decoder.layers.0.self_attention.{kind}.weight"
            split = gdn_component_sizes(config, 2, name)
            split_sizes[name] = split
            count = sum(split) * columns
            parts[name] = [
                (torch.arange(count).reshape(sum(split), columns) + 500 * peer).to(torch.bfloat16) for peer in range(2)
            ]
            tensor = parts[name][rank]
            infos.append(
                ParamInfo(
                    name=name,
                    dtype=tensor.dtype,
                    shape=tensor.shape,
                    attrs={"tensor_model_parallel": True, "partition_dim": 0},
                    size=tensor.numel() * tensor.element_size(),
                    src_rank=rank,
                )
            )

        class Converter:
            def init_tasks(self):
                pass

            def broadcast_and_apply_configs(self):
                pass

            def convert(self, name, tensor):
                return [(name + ".logical", tensor)]

        iterator = SimpleNamespace(
            lora_merge_mode=False,
            lora_adapter_mode=False,
            quantization_config=None,
            args=SimpleNamespace(),
            model_name="qwen3next",
            _expert_buckets=[],
            _non_expert_buckets=[infos],
            _gdn_split_sizes=split_sizes,
            _bridge_converter=Converter(),
        )
        with ExitStack() as stack:
            for key, value in {
                "get_tensor_model_parallel_world_size": 2,
                "get_tensor_model_parallel_rank": rank,
                "get_tensor_model_parallel_group": dist.group.WORLD,
                "get_pipeline_model_parallel_world_size": 1,
                "get_expert_model_parallel_world_size": 1,
            }.items():
                stack.enter_context(patch.object(mpu, key, return_value=value))
            stack.enter_context(patch.object(direct.device_utils, "make_current_torch_device", return_value="cpu"))
            stack.enter_context(patch.object(direct, "device_module", SimpleNamespace(synchronize=lambda: None)))
            stack.enter_context(patch.object(direct, "monkey_patch_torch_reductions", lambda: None))
            adapter = BridgeSourceAdapter(iterator)
            exporter = AdaptiveSourceExporter(adapter, dist.group.WORLD, Path(output), io_timeout=30)
            try:
                for update in range(2):
                    if update:
                        for peers in parts.values():
                            peers[1].view(torch.int16).reshape(-1)[-1] = 0x3F80
                    result = exporter.prepare({name: peers[rank] for name, peers in parts.items()})
                    assert exporter.statistics["FULL"] == 2
                    if rank == 0:
                        for name, peers in parts.items():
                            components = [peer.split(split_sizes[name], dim=0) for peer in peers]
                            expected = torch.cat(
                                [
                                    peer[component]
                                    for component in range(len(split_sizes[name]))
                                    for peer in components
                                ],
                                dim=0,
                            )
                            assert (
                                result[name + ".logical"][2].read_bytes()
                                == expected.view(torch.uint8).numpy().tobytes()
                            )
                    exporter.confirm()
            finally:
                exporter.close()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_bridge_gdn_full_export_preserves_components_across_two_ranks(tmp_path):
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_gdn_export_worker, args=(rank, f"file://{tmp_path / 'rendezvous'}", str(tmp_path)))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=90)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


@pytest.mark.parametrize(
    "shape,dimension,kind",
    [((6, 4), 0, "gather_gate_up"), ((3, 4), 1, "identity"), ((3, 4), 0, "identity"), ((3,), None, "identity")],
)
def test_gather_coordinates_match_full_concatenation(shape, dimension, kind):
    shards = [np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) + 40000 * rank for rank in range(2)]
    if kind == "gather_gate_up":
        parts = [np.split(value, 2, axis=0) for value in shards]
        complete = np.concatenate([part[0] for part in parts] + [part[1] for part in parts], axis=0)
    elif dimension is not None:
        complete = np.concatenate(shards, axis=dimension)
    else:
        complete = shards[0]
    mapping = TensorMapping(shape, 2, dimension, (("source", complete.shape),), kind=kind)
    reconstructed = np.zeros_like(complete).reshape(-1)
    for rank in range(2 if dimension is not None else 1):
        positions, targets = mapping.destinations(rank, np.arange(np.prod(shape)))["source"]
        reconstructed[targets] = shards[rank].reshape(-1)[positions]
    assert reconstructed.tobytes() == complete.tobytes()


def test_bridge_reuse_validates_logical_cache_and_reconverts(tmp_path):
    torch = pytest.importorskip("torch")
    from relax.backends.megatron.weight_update.bridge_source import BridgeSourceAdapter
    from relax.distributed.weight_sync.checkpoint import snapshot_tensors
    from relax.distributed.weight_sync.source_patch import file_digest

    calls = []

    class Converter:
        def convert(self, name, tensor):
            calls.append(name)
            return [("logical", tensor.reshape(2, 4).T.contiguous())]

    adapter = object.__new__(BridgeSourceAdapter)
    adapter.iterator = SimpleNamespace(_bridge_converter=Converter())
    info = SimpleNamespace(name="source")
    original = torch.arange(8, dtype=torch.bfloat16)
    table = snapshot_tensors([("source", original)], tmp_path / "raw")
    first = tmp_path / "first"
    first.mkdir()
    result = adapter.finish_group(info, table, first, "FULL", None)
    previous = {
        "logical": result,
        "logical_groups": {"source": ["logical"]},
        "logical_hashes": {"logical": file_digest(result["logical"][2])},
    }
    second = tmp_path / "second"
    second.mkdir()
    adapter.finish_group(info, table, second, "REUSE", previous)
    assert len(calls) == 1
    result["logical"][2].write_bytes(b"\0" * 16)
    third = tmp_path / "third"
    third.mkdir()
    repaired = adapter.finish_group(info, table, third, "REUSE", previous)
    assert len(calls) == 2
    assert (
        repaired["logical"][2].read_bytes()
        == original.reshape(2, 4).T.contiguous().view(torch.uint8).numpy().tobytes()
    )
