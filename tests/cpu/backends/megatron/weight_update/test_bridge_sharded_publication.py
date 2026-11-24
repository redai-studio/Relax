# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two real Gloo contributors exercise the production Bridge, sharded export
and late ACK."""

from argparse import Namespace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist


pytest.importorskip("megatron.bridge.models.conversion.param_mapping", exc_type=ImportError)


def _run(rank, rendezvous):
    from megatron.bridge.models.conversion.param_mapping import DirectMapping, FusedGatedExpertMapping
    from megatron.core import mpu

    from relax.backends.megatron.weight_update import hf_weight_iterator_bridge as bridge
    from relax.backends.megatron.weight_update.bridge_converter import BridgeConverter
    from relax.backends.megatron.weight_update.delta_export import ShardDeltaExporter
    from relax.backends.megatron.weight_update.expert_gather import Bf16ExpertGather
    from relax.utils.types import ParamInfo

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=25))
    group = dist.group.WORLD
    singletons = [dist.new_group([r], backend="gloo") for r in (0, 1)]
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
            patch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 2)
            patch.setattr(mpu, "get_expert_model_parallel_world_size", lambda: 2)
            patch.setattr(mpu, "get_expert_tensor_parallel_world_size", lambda: 1)
            patch.setattr(mpu, "get_tensor_model_parallel_group", lambda: group)
            patch.setattr(mpu, "get_expert_tensor_and_model_parallel_group", lambda: group)
            patch.setattr(mpu, "get_expert_model_parallel_group", lambda: group)
            patch.setattr(mpu, "get_expert_tensor_parallel_group", lambda: singletons[rank])
            patch.setattr(bridge.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
            patch.setattr(bridge.device_module, "synchronize", lambda *args: None)

            args = Namespace(vocab_size=32, update_weight_buffer_size=128)
            converter = BridgeConverter(args, [], None)
            converter._bridge_task_map = {}
            ordinary, experts = [], []
            shards = {}
            dtype = torch.bfloat16
            for key, hf, owner, expert in (
                ("decoder.layers.0.attn.weight", "model.attn.weight", rank, False),
                ("decoder.layers.0.mlp.experts.linear_fc1.weight0", "model.mlp.experts.gate_up_proj", 0, True),
                ("decoder.layers.0.mlp.experts.linear_fc1.weight1", "model.mlp.experts.gate_up_proj", 1, True),
            ):
                name = "module.module." + key
                mapping = FusedGatedExpertMapping(key, hf) if expert else DirectMapping(key, hf)
                for child in converter.collect_all_mappings(mapping):
                    if hasattr(child, "_get_or_create_mapping") and getattr(child, "_mapping", None) is None:
                        child._mapping = child._get_or_create_mapping("replicated")
                info = ParamInfo(
                    name,
                    dtype,
                    torch.Size((4, 2)),
                    {"tensor_model_parallel": True, "partition_dim": 0, "partition_stride": 1, "parallel_mode": None},
                    16,
                    owner,
                )
                (experts if expert else ordinary).append(info)
                if owner == rank:
                    shards[name] = torch.arange(8, dtype=dtype).reshape(4, 2) + rank * 16
                    converter._bridge_task_map[key] = SimpleNamespace(
                        mapping=mapping,
                        param_weight=shards[name],
                        megatron_module=SimpleNamespace(config=None),
                    )

            it = bridge.HfWeightIteratorBridge.__new__(bridge.HfWeightIteratorBridge)
            it.args = args
            it._bridge_converter = converter
            it._vanilla_key_map = {name: name for name in shards}
            it._non_expert_buckets = [ordinary]
            it._expert_buckets = it._bf16_expert_buckets = [experts]
            it._bf16_expert_gathers = [Bf16ExpertGather()]
            it._expert_broadcast_caches = [({}, {})]
            it._gdn_split_sizes = {}
            it._async_weight_gather = it._packed_expert_sync = it.lora_merge_mode = False
            it._expert_router = None
            converter._configs_broadcast_done = True
            config = SimpleNamespace(bucket_bytes=128, gpu_budget_bytes=4096, staging_budget_bytes=128)
            source = ShardDeltaExporter(it, None, snapshot_budget_bytes=4096, config=config)
            source.sharded = it.make_sharded_source()

            def export():
                source.begin(shards)
                return [item for chunk in source.export(emit=rank == 0, _owner_group=group) for item in chunk]

            full = export()
            assert not source.sharded.full_reason
            assert not any(r.fallback for r in source.sharded.rows.values())
            assert source.sharded.rows[ordinary[0].name].owners == (0, 1)
            assert [source.sharded.rows[i.name].owners for i in experts] == [(0,), (1,)]
            source.commit()
            for value in shards.values():
                value.view(torch.int16).view(-1)[rank::4] = 0x7FC2 + rank
                value.view(torch.int16).view(-1)[rank + 2] = -32768
            expected = dict(item for chunk in it.get_hf_weight_chunks(shards) for item in chunk)
            delta = export()
            if rank == 0:
                assert [s.name for s in delta] == [s.name for s in full] == list(expected)
                for seed, change in zip(full, delta, strict=True):
                    assert not change.full and change.positions.numel() > 0
                    restored = seed.values.clone().view(torch.int16)
                    restored.index_copy_(0, change.positions.long(), change.values.view(torch.int16))
                    assert torch.equal(restored, expected[seed.name].reshape(-1).view(torch.int16))
                stats = source.sharded.metrics
                assert stats["native_fallback_rows"] == 0
                assert all(stats[k]["value_bytes"] > 0 for k in ("ordinary_tp", "routed_expert"))

            confirmed = source._acknowledged
            with pytest.raises(RuntimeError, match="continue_generation"):
                with source.publication(group):
                    source.finish_delivery()
                    if rank == 1:
                        raise RuntimeError("continue_generation failed after install")
            assert source._acknowledged is confirmed and source._pending is None
            assert all(item.full for item in export())
            source.commit()

            # Force the recipient's native requirement for one ordinary row. The
            # actual DELTA call must disclose it and keep the expert gather sparse.
            shards[ordinary[0].name].fill_(9)
            source.begin(shards)
            changes = [
                item
                for chunk in source.export(full_names={"model.attn.weight"}, emit=rank == 0, _owner_group=group)
                for item in chunk
            ]
            if rank == 0:
                assert [item.name for item in changes if item.full] == ["model.attn.weight"]
                assert source.sharded.metrics["native_fallback_rows"] == 1
                assert source.sharded.metrics["native_fallback_wire_bytes"] == 32
            source.commit()
    finally:
        dist.destroy_process_group()


def test_bridge_local_changes_native_bits_and_failed_publication(tmp_path):
    torch.multiprocessing.start_processes(
        _run, args=((tmp_path / "store").as_uri(),), nprocs=2, join=True, start_method="fork"
    )
