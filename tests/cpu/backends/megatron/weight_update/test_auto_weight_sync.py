# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


pytest.importorskip("megatron.core", reason="Weight sync tests require Megatron Core", exc_type=ImportError)
pytest.importorskip("megatron.bridge", reason="Weight sync tests require Megatron Bridge", exc_type=ImportError)
pytest.importorskip("sglang.srt.utils", reason="Weight sync tests require SGLang serialization", exc_type=ImportError)

from relax.backends.megatron.weight_update import update_weight_from_tensor as sync  # noqa: E402
from relax.backends.megatron.weight_update.expert_layout import DirectExpertLayout  # noqa: E402
from relax.backends.megatron.weight_update.hf_weight_iterator_bridge import HfWeightIteratorBridge  # noqa: E402


def receiver(backend="triton", **overrides):
    return (
        dict(
            tp_size=2,
            ep_size=1,
            pp_size=1,
            moe_dp_size=1,
            moe_runner_backend=backend,
            moe_a2a_backend="none",
            is_cuda=True,
            dtype="bfloat16",
            disable_shared_experts_fusion=True,
            num_experts=4,
            hidden_size=6,
            moe_intermediate_size=8,
        )
        | overrides
    )


@pytest.mark.parametrize(
    "failure",
    [
        "none",
        "local_not_ready",
        "peer_not_ready",
        "configure",
        "query",
        "peer_reject",
        "unsupported_layout",
        "mixed_layout",
        "mixed_parallelism",
    ],
)
@pytest.mark.parametrize("backend", ["triton", "deep_gemm", "flashinfer_cutlass"])
def test_negotiation_agrees_before_selecting_or_falling_back(monkeypatch, failure, backend):
    updater = sync.UpdateWeightFromTensor.__new__(sync.UpdateWeightFromTensor)
    updater.args = SimpleNamespace(colocate=True)
    updater.use_distribute = updater._route_experts = updater.lora_enabled = False
    updater.quantization_config = None
    configured, reductions = [], []

    def configure(layout):
        configured.append(layout)
        if layout is not None and failure == "configure":
            raise ValueError("unsupported source mapping")
        return layout is not None

    updater._hf_weight_iterator = SimpleNamespace(
        configure_direct_experts=configure, _bf16_expert_buckets=[] if failure == "local_not_ready" else [object()]
    )
    engine = Mock()
    updater.rollout_engines = [engine]
    receivers = [receiver(backend)]
    if failure == "unsupported_layout":
        receivers[0]["moe_runner_backend"] = "triton_kernel"
    elif failure == "mixed_layout":
        receivers.append(receiver("triton" if backend == "flashinfer_cutlass" else "flashinfer_cutlass"))
        updater.rollout_engines.append(Mock())
    elif failure == "mixed_parallelism":
        receivers.append(receiver(backend, ep_size=2))
        updater.rollout_engines.append(Mock())
    get = Mock(side_effect=RuntimeError("query failed")) if failure == "query" else Mock(return_value=receivers)
    monkeypatch.setattr(sync.ray, "get", get)
    monkeypatch.setattr(sync, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(sync.dist, "get_rank", lambda **kwargs: 0)
    monkeypatch.setattr(sync.dist, "broadcast_object_list", lambda *args, **kwargs: None)

    def reduce(tensor, *, op, group):
        assert op == sync.dist.ReduceOp.MIN and group == "gloo" and tensor.device.type == "cpu"
        reductions.append(tensor.item())
        if failure == "peer_not_ready" and len(reductions) == 1 or failure == "peer_reject" and len(reductions) == 2:
            tensor.zero_()

    monkeypatch.setattr(sync.dist, "all_reduce", reduce)
    updater._configure_direct_experts([2] * len(receivers))
    assert (updater._direct_expert_layout is not None) == (failure == "none")
    if failure in ("local_not_ready", "peer_not_ready"):
        assert len(reductions) == 1 and configured == [None]
        get.assert_not_called()
    else:
        assert len(reductions) == 2
        if failure != "none":
            assert configured[-1] is None


@pytest.mark.parametrize(
    "topology",
    [
        dict(tp_size=2, ep_size=2, moe_a2a_backend="deepep"),
        dict(tp_size=4, ep_size=2),
        dict(tp_size=4, ep_size=2, moe_dp_size=2),
    ],
)
def test_negotiation_preserves_full_engine_placement_for_expert_parallelism(monkeypatch, topology):
    updater = sync.UpdateWeightFromTensor.__new__(sync.UpdateWeightFromTensor)
    updater.args = SimpleNamespace(colocate=True)
    updater.use_distribute = updater._route_experts = updater.lora_enabled = False
    updater.quantization_config = None
    configure = Mock(side_effect=lambda layout: layout is not None)
    updater._hf_weight_iterator = SimpleNamespace(configure_direct_experts=configure, _bf16_expert_buckets=[object()])
    updater.rollout_engines = [Mock()]
    config = receiver("deep_gemm", **topology)
    monkeypatch.setattr(sync.ray, "get", lambda _: [config])
    monkeypatch.setattr(sync, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(sync.dist, "get_rank", lambda **kwargs: 0)
    monkeypatch.setattr(sync.dist, "broadcast_object_list", lambda *args, **kwargs: None)
    monkeypatch.setattr(sync.dist, "all_reduce", lambda *args, **kwargs: None)
    updater._configure_direct_experts([config["tp_size"]])
    expected = DirectExpertLayout.from_receiver(config)
    assert updater._direct_expert_layout == expected
    configure.assert_called_once_with(expected)


@pytest.mark.parametrize("rank", [0, 3])
def test_reconnect_creates_groups_once_including_placeholder_ranks(monkeypatch, rank):
    updater = sync.UpdateWeightFromTensor.__new__(sync.UpdateWeightFromTensor)
    updater.args = SimpleNamespace(rollout_num_gpus_per_engine=2)
    updater._route_experts = False
    updater._ipc_placement = updater._ipc_gather_group = updater._ipc_gather_src = updater._ipc_engine = None
    updater._configure_direct_experts = Mock()
    first, replacement = Mock(), Mock()
    monkeypatch.setattr(sync.ray, "get", lambda refs: [{"node_id": "local"} for _ in refs])
    monkeypatch.setattr(sync.ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "local"))
    monkeypatch.setattr(sync, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(sync.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(sync.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(
        sync.dist, "all_gather_object", lambda out, local, **kw: out.__setitem__(slice(None), [local] * 4)
    )
    create = Mock(return_value="ipc")
    monkeypatch.setattr(sync.dist, "new_group", create)
    updater.connect_rollout_engines([first], None, [2], [0])
    updater.connect_rollout_engines([replacement], None, [2], [0])
    create.assert_called_once_with(ranks=[0, 1], backend="gloo")
    assert updater._ipc_engine is (replacement if rank == 0 else None)
    with pytest.raises(ValueError, match="changed IPC GPU placement"):
        updater.connect_rollout_engines([replacement], None, [2], [1])


@pytest.mark.parametrize("geometry", [(256, 2048, 512), (256, 7168, 2048)])
@pytest.mark.parametrize("slices", [False, True])
@pytest.mark.parametrize("ep", [8, 16, 32])
def test_packed_projection_memory_budget_uses_metadata_only(monkeypatch, geometry, slices, ep):
    from relax.backends.megatron.weight_update import hf_weight_iterator_bridge as bridge

    e, h, i = geometry
    iterator = HfWeightIteratorBridge.__new__(HfWeightIteratorBridge)
    iterator.args = SimpleNamespace(num_experts=e, update_weight_buffer_size=512 * 1024**2)
    iterator._packed_expert_sync = False
    iterator._direct_expert_layout = None
    infos = [
        SimpleNamespace(
            name=f"decoder.layers.0.mlp.experts.linear_fc{fc}.weight{expert}",
            dtype=torch.bfloat16,
            shape=shape,
            src_rank=expert // (e // ep),
            size=shape[0] * shape[1] * 2,
        )
        for fc, shape in [(1, (2 * i, h)), (2, (h, i))]
        for expert in range(e)
    ]
    iterator._bf16_expert_buckets = [infos]
    iterator._bridge_converter = SimpleNamespace(supports_packed_experts=lambda names: True)
    monkeypatch.setattr(bridge.mpu, "get_expert_model_parallel_group", lambda: "ep")
    monkeypatch.setattr(bridge.dist, "get_process_group_ranks", lambda group: list(range(ep)))
    enabled = h == 2048 or slices
    layout = DirectExpertLayout(e, h, i, 2, supports_expert_slices=slices)
    assert iterator.configure_direct_experts(layout) is enabled
    assert iterator._packed_expert_sync is enabled
    if enabled:
        gathers = iterator._bf16_expert_gathers
        assert len(gathers) == (2 if h == 2048 else 48)
        for bucket, gather in zip(iterator._bf16_expert_buckets, gathers, strict=True):
            full = gather._expert_ranges is None
            assert sum(info.size for info in bucket) <= iterator.args.update_weight_buffer_size * (2 if full else 1)
            if h == 7168 and ep == 8:
                assert {info.src_rank for info in bucket} == set(range(8))
                assert sum(info.size for info in bucket) == 448 * 1024**2
        assert sorted(info.name for bucket in iterator._bf16_expert_buckets for info in bucket) == sorted(
            info.name for info in infos
        )
        assert iterator.configure_direct_experts(layout)
        assert iterator._bf16_expert_gathers is gathers
        assert not iterator.configure_direct_experts(None)
        assert not iterator._packed_expert_sync


def test_ordinary_wait_drains_all_readers_without_adding_collectives(monkeypatch):
    updater = sync.UpdateWeightFromTensor.__new__(sync.UpdateWeightFromTensor)
    updater._route_experts = False
    calls = []

    def get(ref):
        calls.append(ref)
        if isinstance(ref, list) or ref == "failed":
            raise RuntimeError("reader failed")
        return {"success": True}

    monkeypatch.setattr(sync.ray, "get", get)
    reduce = Mock()
    monkeypatch.setattr(sync.dist, "all_reduce", reduce)
    with pytest.raises(RuntimeError, match="reader failed"):
        updater._wait_for_ipc(["failed", "pending"])
    assert calls == [["failed", "pending"], "failed", "pending"]
    reduce.assert_not_called()


@pytest.mark.parametrize("failure", ["conversion", "previous_ack", "dispatch"])
def test_pipeline_retains_both_chunks_until_submitted_readers_finish(monkeypatch, failure):
    import weakref

    class Storage:
        pass

    updater = sync.UpdateWeightFromTensor.__new__(sync.UpdateWeightFromTensor)
    updater._route_experts = updater.lora_enabled = updater.lora_merge_mode = False
    updater.quantization_config = None
    updater.weight_version = 0
    updater.rollout_engines = updater.distributed_rollout_engines = []
    updater.weights_getter = lambda: {}
    alive, completed = {}, []

    def chunks(weights):
        yield []
        if failure == "conversion":
            assert alive["chunk0"]() is not None
            raise RuntimeError("next conversion failed")
        yield []

    def send(tensors, *, dispatch_errors):
        key = "chunk" + str(len(alive))
        storage = Storage()
        alive[key] = weakref.ref(storage)
        if failure == "dispatch" and key == "chunk1":
            dispatch_errors.append("partial dispatch failed")
        return [key], storage

    def get(refs):
        names = refs if isinstance(refs, list) else [refs]
        for name in names:
            assert alive[name]() is not None, "IPC source storage was released while a reader was active"
        if failure == "previous_ack" and refs == ["chunk0"]:
            assert alive["chunk1"]() is not None
            raise RuntimeError("previous reader failed")
        completed.extend(names)
        return [{} for _ in names] if isinstance(refs, list) else {}

    updater._hf_weight_iterator = SimpleNamespace(get_hf_weight_chunks=chunks)
    updater._send_hf_params = send
    monkeypatch.setattr(sync.ray, "get", get)
    monkeypatch.setattr(sync.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(sync.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(sync, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(sync.device_utils, "maybe_backend_barrier_on_weight_chunk", lambda **kw: None)
    with pytest.raises(RuntimeError):
        updater.update_weights()
    assert set(completed) == set(alive)


@pytest.mark.parametrize("kind", ["replicated", "duplicated", "buffer", "sharded"])
def test_auto_async_plan_reads_gdn_config_only_for_sharded_parameters(monkeypatch, kind):
    from relax.backends.megatron.weight_update import hf_weight_iterator_bridge as bridge

    attrs = dict(
        tensor_model_parallel=kind in ("duplicated", "sharded"),
        parallel_mode="duplicated" if kind == "duplicated" else None,
    )
    info = SimpleNamespace(name="decoder.layers.0.self_attention.in_proj.weight", attrs=attrs)
    converter = SimpleNamespace(init_tasks=Mock(), can_convert=Mock())
    monkeypatch.setattr(bridge, "BridgeConverter", lambda **kwargs: converter)
    monkeypatch.setattr(bridge, "_build_param_info_buckets", lambda *args, **kwargs: ([], [[info]], {}, {}))
    monkeypatch.setattr(bridge.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bridge.mpu, "get_pipeline_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(bridge.mpu, "get_tensor_model_parallel_world_size", lambda: 4)
    config = Mock(
        return_value=SimpleNamespace(
            linear_key_head_dim=2, linear_value_head_dim=4, linear_num_key_heads=4, linear_num_value_heads=4
        )
    )
    monkeypatch.setattr(bridge, "get_hf_config", config)
    iterator = HfWeightIteratorBridge(SimpleNamespace(hf_checkpoint="test"), [], "any-model", None)
    assert iterator._async_weight_gather
    if kind == "sharded":
        config.assert_called_once_with("test")
        assert iterator._gdn_split_sizes == {info.name: (2, 2, 4, 4, 1, 1)}
    else:
        config.assert_not_called()
        assert iterator._gdn_split_sizes == {}


@pytest.mark.parametrize("size,enabled", [(56 * 1024**2, True), (513 * 1024**2, False)])
def test_expert_eligibility_uses_full_buffer_instead_of_per_owner_quota(monkeypatch, size, enabled):
    from relax.backends.megatron.weight_update import hf_weight_iterator_bridge as bridge

    converter = SimpleNamespace(init_tasks=Mock(), can_convert=Mock())
    info = SimpleNamespace(name="expert", src_rank=0, size=size, dtype=torch.bfloat16)
    monkeypatch.setattr(bridge, "BridgeConverter", lambda **kwargs: converter)
    monkeypatch.setattr(bridge, "_build_param_info_buckets", lambda *args, **kwargs: ([[info]], [], {}, {}))
    monkeypatch.setattr(bridge.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bridge.mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(bridge.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    monkeypatch.setattr(bridge.mpu, "get_expert_tensor_parallel_world_size", lambda: 1)
    monkeypatch.setattr(bridge.mpu, "get_expert_model_parallel_world_size", lambda: 16)
    monkeypatch.setattr(bridge.mpu, "get_expert_model_parallel_group", lambda: "ep")
    monkeypatch.setattr(bridge.dist, "get_process_group_ranks", lambda group: list(range(16)))
    iterator = HfWeightIteratorBridge(SimpleNamespace(update_weight_buffer_size=512 * 1024**2), [], "any-model", None)
    assert (iterator._bf16_expert_buckets is not None) is enabled
