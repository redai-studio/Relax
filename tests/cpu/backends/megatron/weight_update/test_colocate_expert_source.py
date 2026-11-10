# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


pytest.importorskip("megatron.core")


@pytest.fixture
def module(monkeypatch):
    from relax.backends.megatron.weight_update import update_weight_from_tensor

    monkeypatch.setattr(update_weight_from_tensor, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(update_weight_from_tensor.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(
        update_weight_from_tensor.dist,
        "all_gather_object",
        lambda out, local, **kwargs: out.__setitem__(slice(None), [local] * len(out)),
    )
    return update_weight_from_tensor


@pytest.mark.parametrize(
    "mode,failure",
    [(mode, failure) for mode in ("expert", "ordinary", "mixed") for failure in (None, "prepare", "copy")]
    + [("mixed", "warm_copy")],
)
def test_first_direct_update_uses_one_native_transaction(module, monkeypatch, failure, mode):
    monkeypatch.delenv("RELAX_P2P_WEIGHT_CONSISTENCY_CHECK", raising=False)
    updater = object.__new__(module.UpdateWeightFromTensor)
    updater.shard_p2p_requested = True
    updater.actor_weight_retention_enabled = updater.actor_weight_retention_active = True
    updater._colocate_expert_ipc_enabled = mode != "ordinary"
    updater._colocate_tp_ipc_enabled = mode != "expert"
    updater._route_experts, updater.quantization_config = False, None
    updater.lora_enabled = updater.lora_merge_mode = updater.lora_adapter_mode = updater.use_distribute = False
    updater.rollout_engines, updater.distributed_rollout_engines = [object()], []
    updater.args, updater.weight_version = SimpleNamespace(rollout_num_gpus_per_engine=1), 0
    weights = {"expert": torch.ones(1)}
    updater.weights_getter = lambda: weights
    events = []
    updater._routed_engine_call = lambda engines, method, **kwargs: events.append(method)
    updater._send_hf_params = lambda chunk, **kwargs: (["nonexpert"], chunk)
    updater._wait_for_ipc = lambda refs, **kwargs: events.append("ack") if refs else None

    def chunks(current, *, experts):
        assert current is weights
        events.append("full_native" if experts else "non_expert_native")
        skipped = updater._hf_weight_iterator._shard_p2p_skip_names
        assert skipped == (frozenset() if helper.needs_seed else frozenset({"ordinary_source"}))
        yield [("dense", torch.ones(1))]

    updater._hf_weight_iterator = SimpleNamespace(
        get_non_expert_hf_weight_chunks=lambda current: chunks(current, experts=False),
        get_hf_weight_chunks=lambda current: chunks(current, experts=True),
    )
    helper = SimpleNamespace(
        needs_seed=True,
        failed=False,
        _phase=lambda operation: operation(),
        source=SimpleNamespace(
            has_ordinary=mode != "expert",
            has_experts=mode != "ordinary",
            reset=lambda: None,
            begin_native_seed=lambda: events.append("begin_native_seed"),
            observe_native=lambda chunk: events.append("observe_native"),
            native_skip_names=frozenset({"ordinary_source"}),
        ),
    )

    def transfer(current, phase):
        assert current is weights and updater.weight_version == (1 if helper.needs_seed else 2)
        assert events.count("begin_weight_update") == 1 and "ack" in events
        assert "end_weight_update" not in events
        events.append(phase)
        if failure == phase or failure == "warm_copy" and not helper.needs_seed:
            raise RuntimeError("expert " + phase + " failed")

    helper.prepare = lambda current, **kwargs: transfer(current, "prepare")
    helper.update = lambda current, **kwargs: transfer(current, "copy")
    updater._colocate_expert_ipc = helper
    monkeypatch.setattr(module.dist, "get_rank", lambda *args: 0)
    monkeypatch.setattr(module.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(module.device_utils, "maybe_backend_barrier_on_weight_chunk", lambda **kwargs: None)
    if failure in ("prepare", "copy"):
        with pytest.raises(RuntimeError, match="expert .* failed"):
            updater.update_weights()
        assert helper.failed and "end_weight_update" not in events
        assert "update_weights_from_tensor" not in events
        assert "update_weight_version" not in events
    else:
        updater.update_weights()
        assert events.index("prepare") < events.index("copy") < events.index("end_weight_update")
        assert events.count("end_weight_update") == events.count("update_weights_from_tensor") == 1
        assert events.count("update_weight_version") == 1
        assert events.index("update_weights_from_tensor") < events.index("update_weight_version")
        assert ("full_native" in events) is (mode != "expert")
        if mode != "expert":
            assert events.index("begin_native_seed") < events.index("observe_native") < events.index("prepare")
            helper.needs_seed = False
            events.clear()
            if failure == "warm_copy":
                with pytest.raises(RuntimeError, match="expert copy failed"):
                    updater.update_weights()
                assert helper.failed
                assert not {"end_weight_update", "update_weight_version", "continue_generation"}.intersection(events)
            else:
                updater.update_weights()
                assert events.index("copy") < events.index("update_weight_version")
            assert not {"prepare", "observe_native", "begin_native_seed"}.intersection(events)
            assert ("full_native" in events) is (mode == "ordinary")
    assert updater.weight_version == (2 if failure in (None, "warm_copy") and mode != "expert" else 1)
    assert updater.actor_weight_retention_active
    assert "continue_generation" not in events
    assert not updater._hf_weight_iterator._shard_p2p_skip_names


@pytest.mark.parametrize("needs_seed", [False, True])
def test_public_expert_dispatch_rejects_failed_session_before_any_native_update(module, monkeypatch, needs_seed):
    from relax.distributed.checkpoint_service.backends.shard_p2p.synchronizer import ShardP2PSynchronizer

    updater = object.__new__(module.UpdateWeightFromTensor)
    updater.shard_p2p_requested = True
    updater.actor_weight_retention_enabled = updater.actor_weight_retention_active = True
    updater._colocate_expert_ipc_enabled = True
    updater._route_experts, updater.quantization_config = False, None
    updater.lora_enabled = updater.lora_merge_mode = updater.lora_adapter_mode = updater.use_distribute = False
    updater._update_weights_native, updater.weights_getter = Mock(), Mock()

    def agree(refs, *, error=None):
        if error:
            raise RuntimeError(error)

    updater._wait_for_ipc = agree
    helper = object.__new__(ShardP2PSynchronizer)
    helper.failed, helper.needs_seed = True, needs_seed
    helper.prepare = Mock()
    updater._colocate_expert_ipc = helper
    with pytest.raises(RuntimeError, match="dispatch disagreed or failed"):
        updater.update_weights()
    updater._update_weights_native.assert_not_called()
    updater.weights_getter.assert_not_called()
    helper.prepare.assert_not_called()
    assert updater.actor_weight_retention_active and helper.failed


@pytest.mark.parametrize("failure", ["layout", "receiver", "collect", "device"])
def test_expert_source_lifetime_and_cleanup_failure_agreement(module, monkeypatch, failure):
    from relax.distributed.checkpoint_service.backends.shard_p2p import synchronizer as ipc

    held = {"flat": SimpleNamespace(device=torch.device("cuda", int(failure == "device")))}
    plans = [[{"source": "flat", "offset": 0}], [{"source": "flat", "offset": 1}]]
    helper = object.__new__(ipc.ShardP2PSynchronizer)
    helper.geometry = dict(tp_size=2, source_world_size=8)
    helper.rank, helper.failed, helper.needs_seed = 0, False, False
    helper.completed_version, helper._pending = 6, None
    helper.session, helper._plan_hashes = "session", ["tp0", "tp1"]
    helper._source_device = torch.device("cuda", 0)
    helper._needed_keys_by_slot = tuple(frozenset(write["source"] for write in plan) for plan in plans) * 4
    helper._receiver_plan_hashes = tuple(f"delivery-{slot}" for slot in range(8))

    def wait(refs, *, error=None):
        if error:
            raise RuntimeError(error)
        if refs and failure == "receiver":
            raise RuntimeError("receiver failed")

    helper.engines = [Mock() for _ in range(4)]
    helper._wait_for_receivers = wait
    helper._drain_receivers = lambda refs: None

    def local_plan(*args, **kwargs):
        assert kwargs["sources_only"] is True
        if failure == "layout":
            raise ValueError("layout changed")
        return held, None

    helper.source = SimpleNamespace(local_plan=local_plan)
    exports = Mock(side_effect=lambda value: {"handle": object()})
    monkeypatch.setattr(ipc, "export_tensor", exports)
    helper._gather = lambda value: [value] * 8

    def fanout(commands):
        assert helper.retained_sources is held
        assert len(commands) == 4 and all(len(pair) == 2 for pair in commands)
        for pair in commands:
            for tp, command in enumerate(pair):
                assert {item["source"] for item in command["bindings"]} == ({"flat"})
        handles = [
            item["payload"]["handle"]
            for pair in commands
            for command in pair
            for item in command["bindings"]
            if item["source_rank"] == 0
        ]
        assert len(handles) == len(set(handles)), "Every consumer requires a fresh IPC reference"
        return ["reader"], None

    def collect():
        assert helper.retained_sources is None
        if failure == "collect":
            raise RuntimeError("collect failed")

    helper._dispatch = fanout
    monkeypatch.setattr(torch.cuda, "ipc_collect", collect)
    with pytest.raises(RuntimeError, match=failure):
        helper.update({}, version=7)
    assert helper.failed
    assert helper.completed_version == 6
    assert helper.retained_sources is (held if failure == "receiver" else None)
    with pytest.raises(RuntimeError, match="reconnect"):
        helper.invalidate_receivers()
    assert exports.call_count == (0 if failure in ("layout", "device") else 8)


def _expert_partial_dispatch_worker(rank, rendezvous, directory):
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist

    from relax.backends.megatron.weight_update import update_weight_from_tensor as module
    from relax.distributed.checkpoint_service.backends.shard_p2p.synchronizer import (
        ShardP2PSynchronizer,
        _PendingUpdate,
    )

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    # Exercise the real cold collective: one unsupported owner makes all
    # owners keep canonical bindings for this session.
    from relax.distributed.checkpoint_service.backends.shard_p2p import synchronizer as sync

    cold = object.__new__(ShardP2PSynchronizer)
    cold.rank, cold.group, cold.session, cold.failed = rank, dist.group.WORLD, "cold", False
    cold.geometry, cold.engines = dict(tp_size=1, source_world_size=2), [object()]
    cold.completed_version = None
    writes = [
        [
            dict(
                id=f"owner-{rank}",
                target="tp-0",
                source_rank=rank,
                source="flat",
                param="p",
                src_offset=0,
                src_stride=2,
                dst_offset=rank * 2,
                dst_stride=2,
                rows=1,
                row_bytes=2,
            )
        ]
    ]
    cold.source = SimpleNamespace(
        reset=lambda: None,
        local_plan=lambda weights: ({"flat": torch.ones(1, 1, dtype=torch.bfloat16)}, writes),
        raw_recipe_available=rank == 0,
        select_raw_recipe=Mock(),
        complete_geometry=lambda plans, geometry: geometry,
        target_layouts=lambda geometry: {"p": dict(nbytes=4)},
    )
    endpoint = dict(
        boot_id="00000000-0000-0000-0000-000000000001",
        hostname="cpu-test",
        pid=rank + 1,
        gpu_uuid=f"gpu-{rank}",
        device_index=rank,
        generation="cold",
    )
    cold._audit_expert_replicas = lambda weights: None
    cold._actor_endpoint = lambda weights: endpoint
    cold._negotiate_delivery = lambda commands, endpoints, builders: None
    cold._fanout = lambda commands: None
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sync, "worker_identity", lambda *args: endpoint)
        cold.prepare({}, version=0, audit_native=False)
    cold.source.select_raw_recipe.assert_called_once_with(False)
    assert not cold.needs_seed

    # A single owner's late binding failure must stop every owner before the
    # next seed collective, using the same error agreement as the IPC path.
    import json

    from relax.backends.megatron.weight_update.combined_shard_source import CombinedShardSource

    combined = object.__new__(CombinedShardSource)
    combined.rank, combined.group, combined._tp_size, combined.has_experts = rank, dist.group.WORLD, 1, False

    def bind(weights, *, seed):
        assert seed
        if rank == 1:
            raise ValueError("ordinary source binding failed")
        return {}

    combined.ordinary = SimpleNamespace(
        catalogue=lambda weights: dict(
            rank=rank, global_rank=rank, replica_rank=0, replica_ranks=[rank], native_specs={}, groups={}
        ),
        bind=bind,
    )

    def describe(commands):
        if rank:
            return None
        reply = dict(
            description_hash=commands[0][0]["description_hash"],
            tp_rank=0,
            selected_hf=[],
            components=[],
            target_shapes={},
            target_hf={},
            writes=[],
        )
        return [dict(success=True, message=json.dumps(dict(shard_p2p_tp_targets=[reply])))]

    cold._fanout = describe
    with pytest.raises(RuntimeError, match="ordinary source binding failed"):
        combined.prepare_targets(cold, {}, version=1)
    assert cold.failed
    value = torch.tensor(1)
    dist.all_reduce(value)
    assert value.item() == 2

    marker = Path(directory) / "late-reader-fenced"
    helper = object.__new__(ShardP2PSynchronizer)
    held = {"source": object()}
    helper.group, helper.failed, helper.completed_version = dist.group.WORLD, False, 6
    helper._pending = _PendingUpdate(
        7, held, ["late-reader"] if rank == 0 else [], "partial submit" if rank == 0 else None
    )

    def get(ref):
        assert helper.retained_sources is held
        if isinstance(ref, list):
            raise RuntimeError("partial submit acknowledgement failed")
        assert ref == "late-reader"
        # Both failure drain paths can observe this same completed reader.
        # Publish atomically so the peer cannot read a second write's truncate.
        temporary = marker.with_suffix(".tmp")
        temporary.write_text("fenced")
        temporary.replace(marker)

    module.ray.get = get
    native_reduce = dist.all_reduce
    collect = Mock(side_effect=AssertionError("failed readers must retain IPC storage"))
    torch.cuda.ipc_collect = collect

    def agree_after_drain(value, **kwargs):
        if rank == 0:
            assert marker.read_text() == "fenced" and helper.retained_sources is held
        return native_reduce(value, **kwargs)

    dist.all_reduce = agree_after_drain
    try:
        with pytest.raises(RuntimeError, match="partial submit"):
            helper.finish(version=7)
        assert marker.read_text() == "fenced"
        assert helper.failed and helper.completed_version == 6 and helper.retained_sources is held
        with pytest.raises(RuntimeError, match="reconnect"):
            helper.invalidate_receivers()
        collect.assert_not_called()
        value = torch.tensor(1)
        native_reduce(value, group=dist.group.WORLD)
        assert value.item() == 2 and not torch.cuda.is_initialized()
        Path(directory, f"actor-{rank}.done").write_text("retained")
    finally:
        dist.destroy_process_group()


def test_partial_dispatch_real_actor_agreement_retains_sources(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_expert_partial_dispatch_worker, args=((tmp_path / "rdzv").as_uri(), str(tmp_path)), nprocs=2, join=True)
    assert all((tmp_path / f"actor-{rank}.done").read_text() == "retained" for rank in range(2))


@pytest.mark.parametrize("fault", [None, "missing", "duplicate", "membership", "content", "empty"])
def test_expert_replica_audit_uses_real_permuted_groups(module, fault):
    from relax.backends.megatron.weight_update.shard_p2p_source import _validate_replicas

    records = []
    for group in ((6, 1), (4, 7), (3, 0), (5, 2)):
        for replica, rank in enumerate(group):
            records.append(
                dict(
                    rank=rank,
                    role=(group, replica, ()),
                    parameters={f"expert-{group[0]}": ((4, 6), "torch.bfloat16", "digest")},
                )
            )
    if fault == "missing":
        records.pop()
    elif fault == "duplicate":
        records.append(records[0])
    elif fault == "membership":
        records[0]["role"] = ((6, 7), 0, ())
    elif fault == "content":
        records[0]["parameters"] = {"expert-6": ((4, 6), "torch.bfloat16", "different")}
    elif fault == "empty":
        records[0]["parameters"] = records[1]["parameters"] = {}
    if fault:
        with pytest.raises(ValueError, match="replica|Replica"):
            _validate_replicas(records)
    else:
        _validate_replicas(records)


@pytest.mark.parametrize("replica", [0, 1])
def test_source_role_reads_expert_dp_coordinates_not_dense_dp_or_global_order(module, monkeypatch, replica):
    from megatron.core import parallel_state as mpu

    from relax.backends.megatron.weight_update import shard_p2p_source as ipc

    expert_group, actor_group = object(), object()
    ranks = (6, 1)
    helper = object.__new__(ipc.MegatronShardP2PSource)
    helper.group, helper.rank = actor_group, ranks[replica]
    monkeypatch.setattr(mpu, "get_expert_data_parallel_group", lambda: expert_group)
    monkeypatch.setattr(
        ipc.dist, "get_process_group_ranks", lambda group: list(ranks) if group is expert_group else list(range(8))
    )
    monkeypatch.setattr(ipc.dist, "get_rank", lambda group: replica if group is expert_group else helper.rank)
    monkeypatch.setattr(ipc.dist, "get_global_rank", lambda group, rank: rank)
    for axis in (
        "expert_model_parallel",
        "expert_tensor_parallel",
        "tensor_model_parallel",
        "pipeline_model_parallel",
        "context_parallel",
    ):
        monkeypatch.setattr(mpu, "get_" + axis + "_world_size", lambda: 1)
        monkeypatch.setattr(mpu, "get_" + axis + "_rank", lambda: 0)
    role = helper.source_role()
    assert role[:2] == (ranks, replica)
    assert (role[1] == 0) == (helper.rank == 6)
    if replica:
        helper._replica_signature, helper.raw_recipe_enabled = role, False
        helper.iterator, helper.layout = object(), SimpleNamespace(tp_size=2)
        helper._raw_source_recipe = None
        monkeypatch.setattr(ipc, "local_expert_plan", Mock(side_effect=AssertionError("Nonowner duplicate export")))
        assert helper.local_plan({}) == ({}, [[], []])
        assert helper.local_plan({}, sources_only=True) == ({}, None)


@pytest.mark.parametrize(
    "rejected",
    [
        None,
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_tensor_parallel_size",
        "receiver",
        "layout",
        "peer",
        "remote_source",
    ],
)
def test_expert_consumer_qualifies_without_enabling_backup_overlap(module, monkeypatch, rejected):
    monkeypatch.setenv("RELAX_COLOCATE_OVERLAP_WEIGHT_BACKUP", "0")
    updater = object.__new__(module.UpdateWeightFromTensor)
    updater.shard_p2p_requested = True
    dimensions = dict(pipeline_model_parallel_size=1, context_parallel_size=1, expert_tensor_parallel_size=1)
    if rejected in dimensions:
        dimensions[rejected] = 2
    updater.args = SimpleNamespace(**dimensions)
    updater.actor_weight_retention_enabled = True
    updater._expert_ipc_receiver_support = rejected != "receiver"
    updater._direct_expert_layout = None if rejected == "layout" else SimpleNamespace(ep_size=1, moe_dp_size=1)
    monkeypatch.setattr(module.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(
        module.dist, "all_reduce", lambda value, **kwargs: value.zero_() if rejected == "peer" else None
    )
    updater._configure_colocate_expert_ipc(["node", "other-node" if rejected == "remote_source" else "node"])
    assert updater._colocate_expert_ipc_enabled == (rejected is None)
    assert updater.actor_weight_retention_enabled
    assert not module.Envs.RELAX_COLOCATE_OVERLAP_WEIGHT_BACKUP


def test_ordinary_receiver_requalification_keeps_native_fallback_and_rejects_active_capability_loss(
    module, monkeypatch
):
    from relax.backends.megatron.weight_update.combined_shard_source import CombinedShardSource
    from relax.distributed.checkpoint_service.backends.shard_p2p.synchronizer import ShardP2PSynchronizer

    updater = object.__new__(module.UpdateWeightFromTensor)
    updater.shard_p2p_requested = updater.actor_weight_retention_enabled = True
    updater.args = SimpleNamespace(rollout_num_gpus_per_engine=1)
    updater._direct_expert_layout = None
    updater._expert_ipc_receiver_support = updater._ordinary_tp_receiver_support = False
    updater.lora_enabled = updater.quantization_config = updater._route_experts = updater.use_distribute = False
    updater._hf_weight_iterator = SimpleNamespace(_bridge_converter=object())
    updater.rollout_engines = [object()]
    monkeypatch.setattr(module.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(module.dist, "all_reduce", lambda value, **kwargs: None)
    updater._configure_colocate_expert_ipc(["node"])
    assert not updater._colocate_tp_ipc_enabled and not updater._colocate_expert_ipc_enabled

    updater._ordinary_tp_receiver_support = True
    updater._configure_colocate_expert_ipc(["node"])
    assert updater._colocate_tp_ipc_enabled
    source = object.__new__(CombinedShardSource)
    source.expert, source.has_experts, source.has_ordinary = None, False, True
    from relax.backends.megatron.weight_update.tp_shard_source import OrdinaryTPSource

    source.ordinary = object.__new__(OrdinaryTPSource)
    source.ordinary.native_specs, source.ordinary.selected = {"stale": {}}, {"old"}
    source._plans = [["old"]]
    helper = object.__new__(ShardP2PSynchronizer)
    helper.source, helper.group, helper.failed, helper._pending = source, "gloo", False, None
    helper.needs_seed = False
    updater._colocate_expert_ipc = helper
    # Same-capability reconnect retains the session but must observe and plan anew.
    updater._configure_colocate_expert_ipc(["node"])
    replacements = [object()]
    helper.invalidate_receivers(replacements)
    source.begin_native_seed()
    assert source.ordinary.native_specs == {}
    source.observe_native([("fresh", torch.ones(2, dtype=torch.bfloat16))])
    source.reset()
    assert helper.needs_seed and helper.engines is replacements and not helper.failed
    assert set(source.ordinary.native_specs) == {"fresh"} and not source.ordinary.selected and source._plans is None

    updater._ordinary_tp_receiver_support = False
    with pytest.raises(RuntimeError, match="capabilities changed"):
        updater._configure_colocate_expert_ipc(["node"])
    assert helper.failed
    with pytest.raises(RuntimeError, match="session failed"):
        helper._phase(lambda: pytest.fail("Failed session must not submit another native or direct update"))
