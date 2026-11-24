# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Actual Bridge conversion and local sparse projection over two Gloo ranks."""

from argparse import Namespace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist


pytest.importorskip("megatron.bridge.models.conversion.param_mapping", exc_type=ImportError)


def _parameter(model, name, value, *, tp):
    prefix, field = name.rsplit(".", 1)
    parent = model
    for component in prefix.split("."):
        if not hasattr(parent, component):
            parent.add_module(component, torch.nn.Module())
        parent = getattr(parent, component)
    param = torch.nn.Parameter(value, requires_grad=False)
    param.tensor_model_parallel = tp
    param.partition_dim = 0
    param.partition_stride = 1
    param.parallel_mode = None
    parent.register_parameter(field, param)
    return param


def _run(rank, rendezvous):
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.conversion.param_mapping import ColumnParallelMapping, FusedGatedExpertMapping
    from megatron.core import mpu

    from relax.backends.megatron.weight_update import hf_weight_iterator_bridge as bridge
    from relax.backends.megatron.weight_update.bridge_converter import BridgeConverter
    from relax.backends.megatron.weight_update.common import named_params_and_buffers
    from relax.backends.megatron.weight_update.delta_export import ShardDeltaExporter
    from relax.backends.megatron.weight_update.expert_gather import Bf16ExpertGather
    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.utils.types import ParamInfo

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=25))
    group = dist.group.WORLD
    singletons = [dist.new_group([r], backend="gloo") for r in (0, 1)]
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(mpu, "is_initialized", lambda: True)
            patch.setattr(mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
            patch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 2)
            patch.setattr(mpu, "get_expert_model_parallel_world_size", lambda: 2)
            patch.setattr(mpu, "get_expert_tensor_parallel_world_size", lambda: 1)
            patch.setattr(mpu, "get_pipeline_model_parallel_group", lambda: singletons[rank])
            patch.setattr(mpu, "get_tensor_model_parallel_group", lambda: group)
            patch.setattr(mpu, "get_expert_tensor_and_model_parallel_group", lambda: group)
            patch.setattr(mpu, "get_expert_model_parallel_group", lambda: group)
            patch.setattr(mpu, "get_expert_tensor_parallel_group", lambda: singletons[rank])
            patch.setattr(bridge.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
            patch.setattr(bridge.device_module, "synchronize", lambda *args: None)

            args = Namespace(vocab_size=32, update_weight_buffer_size=128, hf_checkpoint="fixture")
            model = torch.nn.Module()
            model.config = SimpleNamespace(share_embeddings_and_output_weights=False)
            converter = BridgeConverter(args, [model], None)
            converter._bridge_task_map = {}
            ordinary, experts, tasks = [], [], []
            ordinary_name = "decoder.layers.0.attn.weight"
            dtype = torch.bfloat16
            for name, hf_name, owner, expert in (
                (ordinary_name, "model.attn.weight", rank, False),
                ("decoder.layers.0.mlp.experts.linear_fc1.weight0", "model.mlp.experts.gate_up_proj", 0, True),
                ("decoder.layers.0.mlp.experts.linear_fc1.weight1", "model.mlp.experts.gate_up_proj", 1, True),
            ):
                info = ParamInfo(
                    "module.module." + name,
                    dtype,
                    torch.Size((4, 2)),
                    {"tensor_model_parallel": True, "partition_dim": 0, "partition_stride": 1, "parallel_mode": None},
                    16,
                    owner,
                )
                (experts if expert else ordinary).append(info)
                if owner != rank:
                    continue
                param = _parameter(model, name, torch.arange(8, dtype=dtype).reshape(4, 2) + rank * 16, tp=not expert)
                mapping = FusedGatedExpertMapping(name, hf_name) if expert else ColumnParallelMapping(name, hf_name)
                for child in converter.collect_all_mappings(mapping):
                    if hasattr(child, "_get_or_create_mapping") and child._mapping is None:
                        child._mapping = child._get_or_create_mapping("replicated")
                task = SimpleNamespace(
                    mapping=mapping, param_weight=param, megatron_module=model, global_param_name=name
                )
                converter._bridge_task_map[name] = task
                tasks.append(task)

            patch.setattr(
                AutoBridge,
                "from_hf_pretrained",
                lambda *args, **kwargs: SimpleNamespace(get_conversion_tasks=lambda _: tasks),
            )
            shards = dict(named_params_and_buffers(args, [model], convert_to_global_name=False))
            iterator = bridge.HfWeightIteratorBridge.__new__(bridge.HfWeightIteratorBridge)
            iterator.args = args
            iterator.model = [model]
            iterator._bridge_converter = converter
            iterator._vanilla_key_map = {
                info.name: "vp_stages.0." + info.name.removeprefix("module.module.") for info in [*ordinary, *experts]
            }
            iterator._non_expert_buckets = [ordinary]
            iterator._expert_buckets = iterator._bf16_expert_buckets = [experts]
            iterator._bf16_expert_gathers = [Bf16ExpertGather()]
            iterator._expert_broadcast_caches = [({}, {})]
            iterator._gdn_split_sizes = {}
            iterator._async_weight_gather = iterator._packed_expert_sync = iterator.lora_merge_mode = False
            iterator._expert_router = None
            converter._configs_broadcast_done = True
            config = DeltaSyncConfig(stream="gloo", bucket_bytes=128, gpu_budget_bytes=8192, staging_budget_bytes=32)
            native_chunks = iterator.get_hf_weight_chunks
            exported_fulls = []

            def count_native_export(snapshot):
                exported_fulls.append(True)
                yield from native_chunks(snapshot)

            patch.setattr(iterator, "get_hf_weight_chunks", count_native_export)
            source = ShardDeltaExporter(iterator, snapshot_budget_bytes=4096, config=config)

            def publication_rows():
                source.begin(shards)
                chunks = list(source.export(emit=rank == 0, _owner_group=group))
                if rank != 0:
                    assert not any(chunks)
                return chunks

            def native_full():
                return dict(item for chunk in native_chunks(shards) for item in chunk)

            def words(value):
                return value.contiguous().reshape(-1).view(torch.int16)

            original = native_full()
            seed = [entry for chunk in publication_rows() for entry in chunk]
            with source.publication(group):
                source.finish_delivery()
                assert not source._acknowledged
            if rank == 0:
                assert {entry.name for entry in seed} == set(original)
                assert all(
                    entry.full and torch.equal(words(entry.values), words(original[entry.name])) for entry in seed
                )

            def assert_native_delta(delta):
                assert len(delta) > 1
                assert all(0 < sum(16 + item.positions.numel() * 6 for item in chunk) <= 32 for chunk in delta)
                changed = {name: [] for name in original}
                projected = {name: words(tensor).clone() for name, tensor in original.items()}
                for chunk in delta:
                    for entry in chunk:
                        assert not entry.full
                        changed[entry.name].extend(entry.positions.tolist())
                        projected[entry.name].index_copy_(0, entry.positions.long(), words(entry.values))
                assert all(torch.equal(restored, words(expected[name])) for name, restored in projected.items())
                assert all(
                    sorted(positions) == (words(original[name]) != words(expected[name])).nonzero().view(-1).tolist()
                    and positions
                    for name, positions in changed.items()
                )

            confirmed = source._acknowledged
            for value in shards.values():
                words(value)[rank::4] = -32768
                words(value)[rank + 2] = 0x3FA0 + rank
            expected = native_full()
            with source.publication(group):
                delta = publication_rows()
                source.finish_delivery()
                assert source._acknowledged is confirmed
            if rank == 0:
                assert_native_delta(delta)
            assert source._acknowledged is not confirmed
            assert all(torch.equal(words(value), words(source._acknowledged[key])) for key, value in shards.items())

            confirmed = source._acknowledged
            original = expected
            for value in shards.values():
                words(value)[rank::4] = 0x4020 + rank
                words(value)[rank + 2] = 0x4100 + rank
            expected = native_full()
            with pytest.raises(RuntimeError, match="continue_generation"):
                with source.publication(group):
                    delta = publication_rows()
                    source.finish_delivery()
                    assert source._acknowledged is confirmed
                    if rank == 1:
                        raise RuntimeError("continue_generation failed after install")
            if rank == 0:
                assert_native_delta(delta)
            assert source._acknowledged is confirmed and source._pending is None
            assert len(exported_fulls) == 1

            with source.publication(group):
                retry = [entry for chunk in publication_rows() for entry in chunk]
                assert all(
                    entry.full and torch.equal(words(entry.values), words(expected[entry.name])) for entry in retry
                )
                source.finish_delivery()
            assert all(torch.equal(words(value), words(source._acknowledged[key])) for key, value in shards.items())
            with source.publication(group):
                assert not any(publication_rows())
                source.finish_delivery()
            assert len(exported_fulls) == 2

            # Ordinary TP and EP can both be one while expert TP spans both
            # ranks. Only rank one changes; losing its contribution leaves the
            # published HF tensor silently stale.
            from relax.backends.megatron.weight_update.sharded_source import BridgeShardSource
            from relax.backends.megatron.weight_update.verl_projection import build_export_index

            patch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1)
            patch.setattr(mpu, "get_expert_model_parallel_world_size", lambda: 1)
            patch.setattr(mpu, "get_expert_tensor_parallel_world_size", lambda: 2)
            patch.setattr(mpu, "get_tensor_model_parallel_group", lambda: singletons[rank])
            patch.setattr(mpu, "get_expert_model_parallel_group", lambda: singletons[rank])
            patch.setattr(mpu, "get_expert_tensor_parallel_group", lambda: group)
            name, hf = "decoder.mlp.experts.linear_fc2.weight0", "layer.experts.0.down_proj.weight"
            mapping = ColumnParallelMapping(name, hf)
            before = torch.tensor([[4 + rank]], dtype=dtype)
            now = before + rank
            task = SimpleNamespace(mapping=mapping, param_weight=now, global_param_name=name, megatron_module=model)
            # The dense oracle uses the real Bridge ETP gather.
            target = mapping.megatron_to_hf(now, model)[hf].reshape(-1)
            old = mapping.megatron_to_hf(before, model)[hf].reshape(-1)
            source = BridgeShardSource(None)
            source.index = build_export_index(
                SimpleNamespace(get_conversion_tasks=lambda _: [task]), [], {}, metadata_group=group
            )
            source.keys, source.dtypes, source.output_dtypes = ["weight"], [dtype], [dtype]
            source.schema = {hf: (tuple(target.shape), dtype)}
            delta = list(
                source.chunks(
                    {"weight": now},
                    {"weight": before},
                    full_names=set(),
                    metadata_group=group,
                    device="cpu",
                    max_round_bytes=64,
                    staging_budget_bytes=64,
                )
            )
            if rank == 0:
                [entry] = [e for batch in delta for e in batch]
                assert entry.positions.tolist() == [1] and entry.name == hf
                old[entry.positions.long()] = entry.values
                assert torch.equal(words(old), words(target))

            def missing_layout(name, values):
                if rank == 1:
                    raise ValueError("missing native test layout")
                return values

            with pytest.raises(RuntimeError, match="missing native test layout"):
                build_export_index(
                    SimpleNamespace(get_conversion_tasks=lambda _: [task]),
                    [],
                    {},
                    metadata_group=group,
                    postprocess=missing_layout,
                )
    finally:
        dist.destroy_process_group()


def test_bridge_local_changes_match_native_full_and_retry_after_failed_publication(tmp_path):
    torch.multiprocessing.start_processes(
        _run, args=((tmp_path / "store").as_uri(),), nprocs=2, join=True, start_method="fork"
    )
