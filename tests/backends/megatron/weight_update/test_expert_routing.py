# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Local CPU tests: no Ray connection, CUDA allocation or cluster jobs."""

from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.backends.megatron.weight_update.expert_routing import (
    ExpertLayout,
    ExpertRouter,
    TensorSpec,
    build_transfer_plan,
    validate_server_layout,
)


def _name(expert: int, suffix: str = "packed") -> str:
    return f"language_model.model.layers.2.block_sparse_moe.experts.{expert}.w1.weight_{suffix}"


def _values(expert: int, step: int = 0) -> list[tuple[str, torch.Tensor]]:
    # Mixed dtype + odd byte count exercises alignment across tensors and ranks.
    return [
        (_name(expert), torch.arange(15, dtype=torch.uint8).reshape(3, 5).t() + step),
        (_name(expert, "scale"), torch.full((3,), expert + step, dtype=torch.int32)),
    ]


def _specs(values):
    return tuple(TensorSpec(name, tuple(t.shape), t.dtype) for name, t in values)


def test_expert_routing_kimi_128_rank_coverage_and_payload():
    layout = ExpertLayout(896, 16, tuple(range(0, 128, 16)), 128)
    assert layout.destinations(_name(0)) == tuple(range(0, 128, 16))
    assert layout.destinations(_name(895)) == tuple(range(15, 128, 16))
    # EP32 * PP4 sources; every expert belongs to exactly one source.
    schemas = [[] for _ in range(128)]
    for expert in range(896):
        schemas[expert // 28].extend(_specs(_values(expert)))
    plans = [build_transfer_plan(schemas, layout, rank) for rank in range(128)]
    for destination, plan in enumerate(plans):
        names = [s.name for specs in plan.receive_specs for s in specs]
        expected = [
            _name(expert, suffix)
            for expert in range(896)
            if expert // 56 == destination % 16
            for suffix in ("packed", "scale")
        ]
        assert names == expected
        for source in range(128):
            assert plans[source].send_splits[destination] == plan.receive_splits[source]
        assert plan.replicated_payload_bytes == 16 * plan.total_payload_bytes
        assert plan.destination_bytes == tuple(sum(other.receive_splits) for other in plans)


@pytest.mark.parametrize("offsets", [(), (0,), (0, 0), (-1,), (3,)])
def test_expert_routing_rejects_invalid_placement(offsets):
    with pytest.raises(ValueError):
        ExpertLayout(8, 2, offsets, 4)


def test_expert_routing_rejects_missing_pairs_duplicates_and_unknown_names():
    layout = ExpertLayout(8, 2, (0, 2), 4)
    schema = _specs(_values(0))
    for schemas in (
        [schema[:1], (), (), ()],
        [schema, schema, (), ()],
        [(TensorSpec("model.norm.weight", (1,), torch.float32),), (), (), ()],
        [_specs(_values(8)), (), (), ()],
    ):
        with pytest.raises(ValueError):
            build_transfer_plan(schemas, layout, 0)


@pytest.mark.parametrize(
    "override",
    [
        {"tp_size": 4},
        {"moe_dp_size": 2},
        {"enable_eplb": True},
        {"ep_num_redundant_experts": 1},
        {"init_expert_location": "random"},
        {"ep_join_mode": "scale"},
        {"speculative_algorithm": "EAGLE"},
        {"moe_runner_backend": "triton"},
    ],
)
def test_expert_routing_checks_resolved_server_overrides(override):
    config = dict(tp_size=2, ep_size=2, pp_size=1, moe_dp_size=1, moe_runner_backend="flashinfer_mxfp4")
    validate_server_layout(config, 2, 2)
    with pytest.raises(ValueError):
        validate_server_layout(config | override, 2, 2)


def _exchange_worker(rank: int, rendezvous: str, with_bridge: bool) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        layout = ExpertLayout(8, 2, (0, 2), 4)
        router = ExpertRouter(layout, payload_group=dist.group.WORLD, metadata_group=dist.group.WORLD)
        previous = []
        gather = dist.all_gather_object
        for step in range(4):
            router.reset_metrics()
            # Two common buckets. Some sources and receivers are empty in each.
            for bucket in range(2):
                experts = [bucket * 4 + i for i in range(4)]
                local = _values(experts[rank], step) if rank < 3 else []
                if step >= 2 and rank == 1:
                    local[1] = (local[1][0], torch.full((5,), step, dtype=torch.int64))
                if step == 3 and rank == 2:
                    router._plans.clear()
                with patch.object(dist, "all_gather_object", wraps=gather) as collect:
                    result = router.exchange(bucket, local, torch.device("cpu"))
                    assert collect.call_count == (0 if step == 1 else 1)
                expected = []
                for source, expert in enumerate(experts[:3]):
                    if rank in layout.destinations(_name(expert)):
                        values = _values(expert, step)
                        if step >= 2 and source == 1:
                            values[1] = (values[1][0], torch.full((5,), step, dtype=torch.int64))
                        expected.extend(values)
                assert [name for name, _ in result] == [name for name, _ in expected]
                for (_, actual), (_, target) in zip(result, expected, strict=True):
                    torch.testing.assert_close(actual, target)
                for old, snapshot in previous:
                    torch.testing.assert_close(old, snapshot)
                previous.extend((tensor, tensor.clone()) for _, tensor in result)
            assert router.payload_bytes * 2 == router.replicated_bytes
        # All ranks receive the same invalid schema and fail before payload exchange.
        bad = _values(0) if rank in (0, 1) else []
        with pytest.raises(ValueError, match="unique owner"):
            router.exchange(10, bad, torch.device("cpu"))
        assert router.exchange(11, [], torch.device("cpu")) == []
        with pytest.raises(RuntimeError, match="conversion failed"):
            router.exchange(12, [], torch.device("cpu"), error="injected conversion failure" if rank == 2 else None)
        local = _values(rank)
        # Both fresh and cached schemas must reject on every rank before any
        # packing allocation or payload collective, including empty receivers.
        for _ in range(2):
            with (
                patch.object(torch, "cat", side_effect=AssertionError("allocated send buffer")),
                patch.object(dist, "all_to_all_single", side_effect=AssertionError("entered payload collective")),
                pytest.raises(ValueError, match="exceeds the IPC budget"),
            ):
                router.exchange(13, local, torch.device("cpu"), receive_budget=1)
        if with_bridge:
            _check_bridge_chunk_rounds(rank, layout, interleaved=False)
            _check_bridge_chunk_rounds(rank, layout, interleaved=True)
    finally:
        dist.destroy_process_group()


def _check_bridge_chunk_rounds(rank: int, layout: ExpertLayout, *, interleaved: bool) -> None:
    from types import SimpleNamespace

    from relax.backends.megatron.weight_update import hf_weight_iterator_bridge as bridge
    from relax.utils.types import ParamInfo

    iterator = object.__new__(bridge.HfWeightIteratorBridge)
    iterator.args = SimpleNamespace(update_weight_buffer_size=2304)
    # Alternate receiving EP shards, exercising aggregation across disjoint
    # destinations as well as common flush boundaries and empty receivers.
    order = (0, 4, 1, 5, 2, 6, 3, 7) if interleaved else tuple(range(8))
    infos = [
        ParamInfo(
            name=f"decoder.layers.{i}.mlp.experts.linear_fc1.weight{expert}",
            dtype=torch.float32,
            shape=torch.Size([1]),
            attrs={},
            size=96,
            src_rank=0,
        )
        for i, expert in enumerate(order)
    ]
    iterator._expert_buckets = [infos]
    iterator._vanilla_key_map = {info.name: info.name for i, info in enumerate(infos) if i % 3 == rank}
    iterator._non_expert_buckets = []
    iterator._expert_broadcast_caches = []
    iterator.lora_merge_mode = False
    iterator._bridge_converter = SimpleNamespace(
        init_tasks=lambda: None,
        broadcast_and_apply_configs=lambda: None,
        convert=lambda name, tensor: _values(int(name.rsplit("weight", 1)[1]), int(tensor[0])),
    )
    with (
        patch.object(bridge.mpu, "get_expert_tensor_parallel_world_size", return_value=1),
        patch.object(bridge.mpu, "get_expert_model_parallel_world_size", return_value=2),
        patch.object(bridge.mpu, "get_pipeline_model_parallel_world_size", return_value=2),
        patch.object(bridge.device_utils, "make_current_torch_device", return_value=torch.device("cpu")),
    ):
        iterator.configure_expert_routing(layout, payload_group=dist.group.WORLD, metadata_group=dist.group.WORLD)
        assert len(iterator._expert_buckets) < 8
        # Isolate the IPC aggregation regression with one expert per round.
        iterator._expert_buckets = [
            [info]
            for info in sorted(
                (info for bucket in iterator._expert_buckets for info in bucket), key=lambda info: info.name
            )
        ]
        iterator.args.update_weight_buffer_size = 288
        for step in range(2):
            weights = {name: torch.tensor([step]) for name in iterator._vanilla_key_map}
            chunks = list(iterator.get_hf_weight_chunks(weights))
            # Each destination gets 3 * 32 bytes in the first IPC, then 32.
            # Summing bucket maxima would flush at 3+3+2 (three requests).
            if interleaved:
                assert len(chunks) == 2
                assert [len(chunk) for chunk in chunks] == [6, 2]
            else:
                assert len(chunks) == 3
                assert any(not chunk for chunk in chunks)
            actual = dict(pair for chunk in chunks for pair in chunk)
            expected = dict(
                pair
                for expert in range(8)
                if rank in layout.destinations(_name(expert))
                for pair in _values(expert, step)
            )
            assert actual.keys() == expected.keys()
            for name in actual:
                torch.testing.assert_close(actual[name], expected[name])


def _require_bridge_runtime() -> None:
    pytest.importorskip("megatron.core", reason="Bridge iterator checks require Megatron Core", exc_type=ImportError)
    pytest.importorskip(
        "megatron.bridge", reason="Bridge iterator checks require Megatron Bridge", exc_type=ImportError
    )


@pytest.mark.parametrize("with_bridge", [False, True])
def test_expert_routing_multirank_values_empty_ranks_and_cache_refresh(tmp_path, with_bridge):
    if with_bridge:
        _require_bridge_runtime()
    mp.start_processes(
        _exchange_worker,
        args=((tmp_path / "rendezvous").as_uri(), with_bridge),
        nprocs=4,
        join=True,
        start_method="spawn",
    )


def test_expert_routing_rejects_model_expert_count_mismatch():
    config = dict(tp_size=2, ep_size=2, pp_size=1, moe_dp_size=1, moe_runner_backend="flashinfer_mxfp4", num_experts=8)
    validate_server_layout(config, 2, 2, 8)
    with pytest.raises(ValueError, match="expert counts"):
        validate_server_layout(config, 2, 2, 16)
    with pytest.raises(ValueError, match="model config overrides"):
        validate_server_layout(config | {"json_model_override_args": '{"num_experts": 16}'}, 2, 2, 8)


def test_expert_routing_parallel_owner_buckets_respect_fanin_and_uneven_owners():
    _require_bridge_runtime()
    from relax.backends.megatron.weight_update.hf_weight_iterator_bridge import _bucket_experts_by_owner
    from relax.utils.types import ParamInfo

    layout = ExpertLayout(8, 2, (0, 2), 4)
    infos = [
        ParamInfo(
            name=f"decoder.layers.{layer}.mlp.experts.linear_fc1.weight{owner * 2}",
            dtype=torch.bfloat16,
            shape=torch.Size([48]),
            attrs={},
            size=96,
            src_rank=owner,
        )
        for owner in range(4)
        for layer in range(owner + 1)
    ]
    buckets = _bucket_experts_by_owner(infos, 1152, layout)
    assert len(buckets) == 2
    assert sorted(info.name for bucket in buckets for info in bucket) == sorted(info.name for info in infos)
    for bucket in buckets:
        assert all(sum(info.size for info in bucket if info.src_rank == owner) <= 192 for owner in range(4))
    with pytest.raises(ValueError, match="exceeding owner budget"):
        _bucket_experts_by_owner(infos, 288, layout)
