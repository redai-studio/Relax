# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from relax.backends.megatron.weight_update import async_shard as module
from relax.backends.megatron.weight_update.combined_shard_source import CombinedShardSource
from relax.backends.megatron.weight_update.hf_weight_iterator_bridge import HfWeightIteratorBridge
from relax.backends.megatron.weight_update.tp_shard_source import OrdinaryTPSource
from relax.backends.sglang import shard_p2p_weight_loader as loader
from relax.backends.sglang.shard_p2p_weight_loader import apply_command
from relax.distributed.checkpoint_service.backends.shard_p2p import ipc
from relax.distributed.checkpoint_service.backends.shard_p2p.plan import fingerprint


def test_ep1_experts_stay_native_when_direct_export_is_unqualified(monkeypatch):
    from megatron.core import parallel_state as mpu

    monkeypatch.setattr(module.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(module.dist, "get_global_rank", lambda group, rank: rank)
    monkeypatch.setattr(module.dist, "get_process_group_ranks", lambda group: [0])
    monkeypatch.setattr(module.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda *args, **kwargs: None)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_data_parallel_group", lambda **kwargs: "replicas")
    monkeypatch.setattr(module, "ShardP2PSynchronizer", lambda *args, **kwargs: SimpleNamespace(needs_seed=False))
    sync = object.__new__(module.AsyncShardSync)
    sync.rank, sync.group, sync.agent_future = 0, "group", None
    sync._checked = lambda action: action()
    sync.iterator = object.__new__(HfWeightIteratorBridge)
    # EP=1 uses the ordinary exporter, without the EP>1 BF16 gather buckets.
    sync.iterator._packed_expert_sync = False
    sync.iterator._direct_expert_layout = sync.iterator._bf16_expert_buckets = None
    sync.backend = SimpleNamespace(
        args=SimpleNamespace(num_layers=1, rollout_num_gpus_per_engine=1, update_weight_buffer_size=1024),
        rollout_engines={0: object()},
    )
    sync._initialize(
        dict(
            num_experts=2,
            hidden_size=4,
            moe_intermediate_size=4,
            tp_size=1,
            ep_size=1,
            moe_dp_size=1,
            pp_size=1,
            dtype="bfloat16",
            disable_shared_experts_fusion=True,
            is_cuda=True,
            moe_runner_backend="triton",
        )
    )
    sync.native_phase = True
    sync.source.native_skip_names = frozenset(("direct",))
    assert not sync.source.has_experts and sync.source.expert is None
    assert not sync.iterator._packed_expert_sync
    names = ["direct", "fallback", "model.layers.0.mlp.experts.0.gate_proj.weight"]
    assert [name for name in names if not sync.skips_native(name)] == names[1:]


def test_topology_shrink_retires_survivor_with_its_old_identity(monkeypatch):
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(module.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.dist, "all_gather_object", lambda output, value, **kwargs: output.__setitem__(0, value))
    monkeypatch.setattr(module.ray, "get", lambda values: values)
    registration = Mock()
    receiver = SimpleNamespace(
        _relax_expert_ipc=dict(
            session="session",
            engine=1,
            tp_rank=0,
            version=3,
            persistent_targets=True,
            failed=False,
            epoch=None,
            seed_pending=False,
            persistent_registration=registration,
        )
    )
    args = SimpleNamespace(
        tp_size=1,
        pp_size=1,
        dp_size=1,
        ep_size=1,
        quantization=None,
        enable_dp_attention=False,
        enable_lora=False,
        speculative_algorithm=None,
    )

    def receive(*, serialized_named_tensors, **kwargs):
        loader._control_tp(receiver, serialized_named_tensors[0], args, 0, "group")
        return dict(success=True)

    sync = object.__new__(module.AsyncShardSync)
    sync.rank, sync.group = 0, "group"
    sync._checked = lambda action: action()
    sync.engine_indices = {0: 0, 1: 1}
    # Both proxies have been replaced. Only old engine1 survives in new slot0.
    sync.backend = SimpleNamespace(
        rollout_engines={1: SimpleNamespace(update_weights_from_tensor=SimpleNamespace(remote=receive))}
    )
    sync.core = object.__new__(module.ShardP2PSynchronizer)
    core = sync.core
    core.rank, core.group, core.geometry = 0, "group", dict(tp_size=1)
    core.session, core.completed_version, core.failed, core._pending = "session", 3, False, None
    core.serialize = lambda values, **kwargs: values
    core._writer = SimpleNamespace(disconnect_targets=Mock())
    sync._retire(["session"])
    assert not hasattr(receiver, "_relax_expert_ipc")
    registration.release.assert_called_once()
    core._writer.disconnect_targets.assert_called_once()


@pytest.mark.parametrize("invalid", [None, "incomplete", "version", "topology", "audit"])
def test_initial_native_schema_reuse_preserves_trained_version_and_native_fallback(monkeypatch, invalid):
    monkeypatch.setenv("RELAX_P2P_WEIGHT_CONSISTENCY_CHECK", "1" if invalid == "audit" else "0")
    monkeypatch.setattr(module.ray, "get", lambda refs, **kwargs: [])
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda values, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(ipc, "import_tensor", lambda payload, device: payload)
    weights = {name: torch.ones(4, dtype=torch.bfloat16) for name in ("direct", "fallback")}
    seed = dict(
        specs={},
        ready=False,
        version=0 if invalid == "version" else 1,
        signature="changed" if invalid == "topology" else "topology",
    )
    backend = SimpleNamespace(
        weight_version=1,
        _async_shard_initial_seed=seed,
        args=None,
        model=None,
        rollout_engines={},
        _rollout_control_request=lambda *args: None,
        _megatron=SimpleNamespace(named_params_and_buffers=lambda *args, **kwargs: weights.items()),
    )
    module.observe_initial_native(backend, list(weights.items()))
    seed["ready"] = invalid != "incomplete"
    sync = object.__new__(module.AsyncShardSync)
    sync.backend, sync.rank, sync.group, sync.core = backend, 0, "group", None
    sync._checked = lambda action: action()
    sync._probe = lambda: dict(signature="topology", sessions=[None], layout={})
    source = object.__new__(CombinedShardSource)
    source.ordinary = object.__new__(OrdinaryTPSource)
    source.ordinary.native_specs = {}
    source.has_experts, source.native_skip_names = False, frozenset()
    events = []
    direct_complete = Event()
    cache_calls = []

    def release_cache():
        if cache_calls or invalid == "audit":
            assert direct_complete.wait(timeout=3), "Native cache release did not overlap the direct update"
        cache_calls.append("complete")

    monkeypatch.setattr(torch.cuda, "empty_cache", release_cache)
    sync.pool, sync.cache_future = ThreadPoolExecutor(max_workers=1), None
    target = torch.zeros_like(weights["direct"])
    receiver = SimpleNamespace(named_parameters=lambda: [("direct", target)])
    geometry = dict(tp_size=1, source_world_size=1, expert_enabled=False, ordinary_targets={"direct": [4]})
    writes = [
        dict(
            id="direct",
            param="direct",
            source="source",
            source_rank=0,
            src_offset=0,
            dst_offset=0,
            src_stride=8,
            dst_stride=8,
            rows=1,
            row_bytes=8,
        )
    ]
    command = dict(
        session="session",
        engine=0,
        tp_rank=0,
        geometry=geometry,
        writes=writes,
        plan_hash=fingerprint(dict(geometry=geometry, writes=writes)),
    )

    def prepare(current, *, version, audit_native):
        assert current == weights and source.ordinary.native_specs["direct"]["shape"] == [4]
        events.append(("prepare", version, audit_native))
        apply_command(
            receiver,
            dict(
                command,
                op="seed",
                version=version,
                nonce="seed",
                batched=False,
                audit_native=audit_native,
                consistency_check=invalid == "audit",
            ),
            0,
        )
        sync.core.needs_seed = False
        source.native_skip_names = frozenset(("direct",))

    def initialize(layout):
        sync.source = source
        sync.core = SimpleNamespace(
            needs_seed=True,
            prepare=prepare,
            update=direct,
        )

    def direct(current, *, version):
        apply_command(
            receiver,
            dict(
                command,
                op="apply",
                version=version,
                nonce="update",
                bindings=[dict(source_rank=0, source="source", payload=current["direct"].view(1, 4))],
            ),
            0,
        )
        events.append(("direct", version))
        direct_complete.set()

    def native(**kwargs):
        if invalid != "audit":
            assert cache_calls == ["complete"]
        backend.weight_version += 1
        values = [(name, value) for name, value in weights.items() if not sync.skips_native(name)]
        events.append(("native", [name for name, _ in values]))
        if any(name == "direct" for name, _ in values):
            target.copy_(weights["direct"])
        sync.observe_native(values)
        sync.release_native_cache()

    sync._initialize, backend.update_weights_for_rollout = initialize, native

    def control(endpoint, payload=None):
        if endpoint == "/continue_generation":
            assert direct_complete.is_set() and sync.cache_future is None
            assert len(cache_calls) == (1 if invalid == "audit" else 2)

    backend._rollout_control_request = control
    sync.update(rollout_only=True, actor_fwd_only=False)
    sync.pool.shutdown(wait=True)
    assert len(cache_calls) == (1 if invalid == "audit" else 2)
    assert sync.cache_future is None
    assert backend.weight_version == 2 and backend._async_shard_initial_seed is None
    assert events[-1] == ("direct", 2)
    torch.testing.assert_close(target, weights["direct"], rtol=0, atol=0)
    assert receiver._relax_expert_ipc["version"] == 2
    if invalid is None:
        assert events[:2] == [("prepare", 2, False), ("native", ["fallback"])]
    else:
        assert events[:2] == [("native", ["direct", "fallback"]), ("prepare", 2, invalid == "audit")]
