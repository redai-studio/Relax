# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def setup(monkeypatch):
    pytest.importorskip("megatron.core")
    from megatron.bridge.models.conversion import param_mapping as pm
    from megatron.core import parallel_state as mpu

    from relax.backends.megatron.weight_update.bridge_converter import BridgeConverter
    from relax.backends.megatron.weight_update.shard_p2p_source import RawExpertRecipe

    for name, size in (
        ("expert_tensor_parallel", 1),
        ("expert_model_parallel", 8),
        ("tensor_model_parallel", 2),
        ("pipeline_model_parallel", 1),
        ("context_parallel", 1),
    ):
        monkeypatch.setattr(mpu, f"get_{name}_world_size", lambda size=size: size)
    monkeypatch.setattr(pm.MegatronParamMapping, "tp_size", property(lambda self: 1))
    converter = object.__new__(BridgeConverter)
    converter._quantization_config = None
    converter.init_tasks = Mock()
    converter._bridge_task_map = {}
    infos, keys = [], {}
    for expert in range(2):
        for fc, cls in ((1, pm.FusedGatedExpertMapping), (2, pm.FusedExpertMapping)):
            name = f"decoder.layers.0.mlp.experts.linear_fc{fc}.weight{expert}"
            mapping = cls(name, "model.layers.0.mlp.experts." + ("gate_up_proj" if fc == 1 else "down_proj"))
            mapping._detected_type = "column" if fc == 1 else "row"
            mapping._mapping = mapping._get_or_create_mapping(mapping._detected_type)
            converter._bridge_task_map[name] = SimpleNamespace(mapping=mapping, megatron_module=torch.nn.Module())
            shape = (12, 4) if fc == 1 else (4, 6)
            infos.append(
                SimpleNamespace(
                    name=name, src_rank=0, shape=shape, dtype=torch.bfloat16, size=shape[0] * shape[1] * 2, attrs={}
                )
            )
            keys[name] = name
    iterator = SimpleNamespace(
        _direct_expert_layout=SimpleNamespace(
            ep_size=1, moe_dp_size=1, tp_size=2, num_experts=2, hidden_size=4, intermediate_size=6, up_proj_first=False
        ),
        _expert_buckets=[infos],
        _vanilla_key_map=keys,
        _bridge_converter=converter,
    )
    return SimpleNamespace(iterator=iterator, converter=converter, pm=pm, mpu=mpu, recipe=RawExpertRecipe)


def _weights(iterator, version=0, *, separate=False, padding=3):
    # Unequal H/I catches accidental transposes; nonzero offsets exercise the
    # actual allocation-relative rectangles, including gate/up subviews.
    flat = (torch.arange(padding + 144 + 5, dtype=torch.float32) + version * 160).to(torch.bfloat16)
    weights, cursor = {}, padding
    for name in iterator._vanilla_key_map:
        rows, columns = (12, 4) if "linear_fc1" in name else (4, 6)
        tensor = flat[cursor : cursor + rows * columns].view(rows, columns)
        if separate:
            storage = torch.zeros(flat.numel(), dtype=torch.bfloat16)
            storage[cursor : cursor + tensor.numel()].copy_(tensor.flatten())
            tensor = storage[cursor : cursor + tensor.numel()].view(rows, columns)
        weights[name] = tensor
        cursor += rows * columns
    return weights


def _seed(setup, weights):
    from relax.backends.megatron.weight_update.shard_p2p_source import ExpertViewPlanCache, local_expert_plan

    sources, plans, _ = local_expert_plan(setup.iterator, weights, 0, plan_cache=ExpertViewPlanCache())
    return setup.recipe.compile(setup.iterator, weights, 0, sources), plans


def _assert_cpu(value):
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_cpu(key)
            _assert_cpu(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_cpu(item)
    else:
        assert value is None or type(value) in (str, int, bool)


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("up_first", [False, True])
@pytest.mark.parametrize("delegate", ["etp", "replicated"])
def test_raw_recipe_changed_allocations_match_all_native_targets(setup, monkeypatch, separate, up_first, delegate):
    from relax.distributed.checkpoint_service.backends.shard_p2p.ipc import copy_rectangle

    setup.iterator._direct_expert_layout.up_proj_first = up_first
    if delegate == "replicated":
        for task in setup.converter._bridge_task_map.values():
            task.mapping._mapping = task.mapping._get_or_create_mapping("replicated")
            # Bridge's fallback task can have only a config, not an nn.Module.
            task.megatron_module = SimpleNamespace(config=SimpleNamespace())
    weights = _weights(setup.iterator, separate=separate)
    recipe, plans = _seed(setup, weights)
    slow = setup.converter.local_expert_views
    monkeypatch.setattr(setup.converter, "local_expert_views", Mock(side_effect=AssertionError("hot view creation")))
    for version in range(6):
        weights = _weights(setup.iterator, version, separate=separate)
        sources = recipe.bind(setup.iterator, weights, 0)
        canonical = dict(view for name, tensor in weights.items() for view in slow(name, tensor))
        for tp, plan in enumerate(plans):
            targets = {
                "model.layers.0.mlp.experts.w13_weight": torch.full((2, 6, 4), float("nan"), dtype=torch.bfloat16),
                "model.layers.0.mlp.experts.w2_weight": torch.full((2, 4, 3), float("nan"), dtype=torch.bfloat16),
            }
            for write in plan:
                copy_rectangle(sources[write["source"]], targets[write["param"]], write)
            gate, up, down = [
                torch.stack([canonical[f"model.layers.0.mlp.experts.{e}.{part}_proj.weight"] for e in range(2)])
                for part in ("gate", "up", "down")
            ]
            gate, up = gate[:, tp * 3 : (tp + 1) * 3], up[:, tp * 3 : (tp + 1) * 3]
            torch.testing.assert_close(
                targets["model.layers.0.mlp.experts.w13_weight"],
                torch.cat((up, gate) if up_first else (gate, up), dim=1),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                targets["model.layers.0.mlp.experts.w2_weight"], down[:, :, tp * 3 : (tp + 1) * 3], rtol=0, atol=0
            )
        _assert_cpu(vars(recipe))
    setup.converter.local_expert_views.assert_not_called()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("change", ["offset", "split", "merge", "shape", "mapping", "spec", "sharding", "topology"])
def test_raw_recipe_rejects_changed_live_contract_before_export(setup, monkeypatch, change):
    iterator = setup.iterator
    recipe, _ = _seed(setup, _weights(iterator, separate=change == "merge"))
    weights = _weights(iterator)
    name = next(iter(weights))
    task = setup.converter._bridge_task_map[name]
    if change == "offset":
        weights[name] = weights[name].as_strided((12, 4), (4, 1), storage_offset=4)
    elif change == "split":
        weights[name] = weights[name].clone()
    elif change == "shape":
        weights[name] = weights[name].view(6, 8)
    elif change == "mapping":
        task.mapping.transpose_on_export = True
    elif change == "spec":
        specs = task.mapping.local_hf_param_specs(name)
        task.mapping.local_hf_param_specs = lambda name: tuple(reversed(specs))
    elif change == "sharding":
        task.mapping._local_hf_sharding = lambda *args: ("etp", 0, 2)
    elif change == "topology":
        monkeypatch.setattr(setup.mpu, "get_context_parallel_world_size", lambda: 2)
    with pytest.raises((ValueError, KeyError)):
        recipe.bind(iterator, weights, 0)
    _assert_cpu(vars(recipe))


def test_raw_recipe_seed_checks_real_bridge_views(setup, monkeypatch):
    weights = _weights(setup.iterator)
    from relax.backends.megatron.weight_update.shard_p2p_source import local_expert_plan

    sources, _, _ = local_expert_plan(setup.iterator, weights, 0)
    slow = setup.converter.local_expert_views
    monkeypatch.setattr(
        setup.converter, "local_expert_views", lambda *args: [(name, tensor.clone()) for name, tensor in slow(*args)]
    )
    with pytest.raises(ValueError, match="differs from Bridge"):
        setup.recipe.compile(setup.iterator, weights, 0, sources)


@pytest.mark.parametrize("change", ["offset", "narrow", "shape", "dtype"])
def test_raw_recipe_rejects_seed_that_does_not_expose_full_storage(setup, change):
    from relax.backends.megatron.weight_update.shard_p2p_source import local_expert_plan

    weights = _weights(setup.iterator)
    sources, _, _ = local_expert_plan(setup.iterator, weights, 0)
    key = next(iter(sources))
    value = sources[key]
    sources[key] = (
        value[:, 1:]
        if change == "offset"
        else value[:, :-1]
        if change == "narrow"
        else value.flatten()
        if change == "shape"
        else value.view(torch.float32)
    )
    with pytest.raises(ValueError, match="complete BF16 storage"):
        setup.recipe.compile(setup.iterator, weights, 0, sources)


def test_source_lifecycle_raw_cache_fallback_and_reconnect(setup, monkeypatch):
    from relax.backends.megatron.weight_update import shard_p2p_source as ipc

    source = object.__new__(ipc.MegatronShardP2PSource)
    source.iterator, source.layout, source.rank = setup.iterator, setup.iterator._direct_expert_layout, 0
    source._replica_signature = ((0,), 0, ((8, 0), (1, 0), (2, 0), (1, 0), (1, 0)))
    source.source_role = lambda: source._replica_signature
    source.reset()
    with pytest.raises(ValueError, match="native seed"):
        source.local_plan(_weights(setup.iterator), sources_only=True)
    sources, plans = source.local_plan(_weights(setup.iterator))
    first, keys = source._raw_source_recipe, set(sources)
    assert first is not None and all(plans)
    with monkeypatch.context() as patch:
        patch.setattr(ipc, "local_expert_plan", Mock(side_effect=AssertionError("hot rebuilt canonical plan")))
        sources, plans = source.local_plan(_weights(setup.iterator, 1), sources_only=True)
        assert set(sources) == keys and plans is None
    source.select_raw_recipe(False)
    views = Mock(wraps=setup.converter.local_expert_views)
    monkeypatch.setattr(setup.converter, "local_expert_views", views)
    for version in (2, 3):
        sources, plans = source.local_plan(_weights(setup.iterator, version), sources_only=True)
        assert sources and plans is None and source._raw_source_recipe is None
    assert views.call_count == 8
    with monkeypatch.context() as patch:
        patch.setattr(setup.converter, "local_expert_views", Mock(side_effect=ValueError("Bridge view changed")))
        with pytest.raises(ValueError, match="Bridge view changed"):
            source.local_plan(_weights(setup.iterator, 4), sources_only=True)
    source.reset()
    source.local_plan(_weights(setup.iterator, 4))
    assert source._raw_source_recipe is not first and source.raw_recipe_available
    _assert_cpu(vars(source._raw_source_recipe))

    class ParameterSubclass(torch.nn.Parameter):
        pass

    weights = {name: ParameterSubclass(value, requires_grad=False) for name, value in _weights(setup.iterator).items()}
    source.reset()
    sources, plans = source.local_plan(weights)
    assert sources and plans and not source.raw_recipe_available


def test_synchronizer_reconnect_rebuilds_bindings_and_releases_fresh_sources(setup, monkeypatch):
    import json
    import weakref

    import ray

    from relax.backends.megatron.weight_update.shard_p2p_source import MegatronShardP2PSource
    from relax.distributed.checkpoint_service.backends.shard_p2p import synchronizer as ipc
    from relax.distributed.checkpoint_service.backends.shard_p2p.plan import bind_delivery

    # Model one source rank; only process-group, RPC and CUDA boundaries are replaced.
    monkeypatch.setattr(ipc.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(ipc.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(ipc.dist, "get_global_rank", lambda group, rank: rank)
    monkeypatch.setattr(ipc.dist, "all_reduce", lambda value, **kwargs: None)
    monkeypatch.setattr(ipc.dist, "broadcast_object_list", lambda values, **kwargs: None)
    monkeypatch.setattr(ipc.dist, "gather_object", lambda value, out, **kwargs: out.__setitem__(0, value))
    monkeypatch.setattr(ipc.dist, "all_gather_object", lambda out, value, **kwargs: out.__setitem__(0, value))
    monkeypatch.setattr(ray, "get", lambda refs: [ref() for ref in refs])
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    endpoint = dict(
        boot_id="00000000-0000-0000-0000-000000000001",
        hostname="cpu",
        pid=1,
        gpu_uuid="gpu-0",
        device_index=0,
        generation="test",
    )
    monkeypatch.setattr(ipc, "worker_identity", lambda *args: endpoint)
    source = MegatronShardP2PSource(
        setup.iterator, setup.iterator._direct_expert_layout, process_group="gloo", num_layers=1, qualified=True
    )
    source.source_role = lambda: ((0,), 0, ((8, 0), (1, 0), (2, 0), (1, 0), (1, 0)))
    exports, calls = {}, []

    def export(tensor):
        key = len(exports)
        exports[key] = weakref.ref(tensor)
        return key

    def collect():
        assert helper.retained_sources is None and all(ref() is None for ref in exports.values())
        exports.clear()

    monkeypatch.setattr(ipc, "export_tensor", export)
    monkeypatch.setattr(torch.cuda, "ipc_collect", collect)

    def engine(pid):
        deliveries = {}

        def remote(serialized_named_tensors, **kwargs):
            def acknowledge():
                replies = []
                calls.append((pid, serialized_named_tensors[0]["op"]))
                for command in serialized_named_tensors:
                    tp = command["tp_rank"]
                    if command["op"] == "prepare":
                        binding = bind_delivery(
                            command["writes"],
                            command["geometry"],
                            command["sources"],
                            dict(endpoint, pid=pid + tp, generation=str(pid)),
                            layouts=source.target_layouts(command["geometry"]),
                        )
                        deliveries[tp] = binding
                        replies.append(
                            dict(
                                identity={
                                    key: command[key] for key in ("session", "engine", "tp_rank", "version", "nonce")
                                },
                                receiver=binding["delivery"]["receiver"],
                                logical_plan_hash=binding["logical_plan_hash"],
                                plan_hash=binding["plan_hash"],
                            )
                        )
                    elif command["op"] == "seed":
                        assert command["delivery"] == deliveries[tp]
                    else:
                        assert command["delivery_hash"] == deliveries[tp]["plan_hash"]
                        assert helper.completed_version == (version - 1 if version else None)
                        assert helper.retained_sources and command["version"] == version
                        pointers = {value.untyped_storage().data_ptr() for value in weights.values()}
                        observed = set()
                        for item in command["bindings"]:
                            tensor = exports[item["payload"]]()
                            assert tensor is not None
                            observed.add(tensor.untyped_storage().data_ptr())
                        assert observed == pointers
                return dict(success=True, message=json.dumps(dict(shard_p2p_receivers=replies)))

            return acknowledge

        return SimpleNamespace(update_weights_from_tensor=SimpleNamespace(remote=remote))

    helper = ipc.ShardP2PSynchronizer(
        source,
        process_group="gloo",
        engines=[engine(100)],
        serialize=lambda values, **kwargs: json.loads(values[0][0]),
        control_loader="control",
        update_loader="apply",
        validate_devices=lambda *args: None,
    )
    # Keep snapshots alive to prevent address reuse; reconnect changes offsets and alias groups.
    snapshots = [_weights(setup.iterator, v, padding=5 if v == 2 else 3, separate=v == 2) for v in range(3)]
    for version, weights in enumerate(snapshots):
        if version == 2:
            plans, deliveries = helper._plan_hashes, helper._receiver_plan_hashes
            helper.invalidate_receivers([engine(200)])
        if helper.needs_seed:
            helper.prepare(weights, version=version, audit_native=False)
        if version == 2:
            assert helper._plan_hashes != plans and helper._receiver_plan_hashes != deliveries
            helper.start(weights, version=version)
            assert helper.retained_sources and helper.completed_version == 1
            engines = helper.engines
            with pytest.raises(RuntimeError, match="Cannot reconnect"):
                helper.invalidate_receivers([engine(300)])
            assert helper.engines is engines and not helper.needs_seed
            helper.finish(version=version)
        else:
            helper.update(weights, version=version)
        assert helper.completed_version == version and helper.retained_sources is None and not helper.failed
    assert calls == [(100, op) for op in ("prepare", "seed", "apply", "apply")] + [
        (200, op) for op in ("prepare", "seed", "apply")
    ]
    assert not torch.cuda.is_initialized()
