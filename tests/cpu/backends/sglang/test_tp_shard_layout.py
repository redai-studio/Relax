# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Native-loader routing must preserve values and complete dependencies."""

from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang.tp_shard_layout import describe_tp_targets
from relax.distributed.checkpoint_service.backends.shard_p2p.ipc import copy_rectangle


def _description(values, *, grouped=False):
    records = []
    for rank in range(2):
        regions = []
        for name, tensor in values.items():
            rows, columns = tensor.shape
            regions.append(
                dict(
                    hf=name,
                    source=name,
                    global_shape=[rows, columns],
                    offset=[rank * rows // 2, 0],
                    shape=[rows // 2, columns],
                    storage_offset=0,
                    stride=columns,
                    dtype=str(tensor.dtype),
                    itemsize=2,
                )
            )
        records.append(
            dict(
                rank=rank,
                regions=regions,
                groups={"fused": list(values)} if grouped else {name: [name] for name in values},
            )
        )
    return dict(
        records=records,
        source_world_size=2,
        native_specs={name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in values.items()},
    )


class Native(torch.nn.Module):
    def __init__(self, tp_rank, *, transform=False, tied=False):
        super().__init__()
        self.tp_rank, self.transform = tp_rank, transform
        self.fused = torch.nn.Parameter(torch.full((4, 4), -9.0, dtype=torch.bfloat16), requires_grad=False)
        self.other = torch.nn.Parameter(torch.full((2, 4), -7.0, dtype=torch.bfloat16), requires_grad=False)
        if tied:
            self.tied = self.other

    def load_weights(self, values):
        for name, value in values:
            chunk = value.chunk(2)[self.tp_rank]
            if name == "gate":
                self.fused[:2].copy_(chunk)
            elif name == "up":
                self.fused[2:].copy_(chunk + 1 if self.transform else chunk)
            else:
                self.other.copy_(chunk)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_native_tp_plan_matches_loader_without_mutating_original(tp_rank):
    values = {
        name: torch.arange(16, dtype=torch.bfloat16).view(4, 4) + index * 32
        for index, name in enumerate(("gate", "up", "other"))
    }
    model = Native(tp_rank)
    original = {name: (value, value.detach().clone(), value.data_ptr()) for name, value in model.named_parameters()}
    reply = describe_tp_targets(model, _description(values), tp_rank)
    assert reply["selected_hf"] == ["gate", "other", "up"]
    assert ["gate", "up"] in reply["components"]
    actual = {name: torch.empty_like(value) for name, value in model.named_parameters()}
    for write in reply["writes"]:
        source = values[write["source"]].chunk(2)[write["source_rank"]]
        copy_rectangle(source, actual[write["param"]], write)
    for name, (value, snapshot, address) in original.items():
        assert dict(model.named_parameters())[name] is value and value.data_ptr() == address
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
    model.load_weights(values.items())
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, actual[name], rtol=0, atol=0)


@pytest.mark.parametrize("grouped", [False, True])
def test_numeric_transform_falls_back_with_complete_fused_dependency(grouped):
    values = {name: torch.ones((4, 4), dtype=torch.bfloat16) for name in ("gate", "up", "other")}
    reply = describe_tp_targets(Native(0, transform=True), _description(values, grouped=grouped), 0)
    assert reply["selected_hf"] == ([] if grouped else ["other"])
    assert not any(write["param"] == "fused" for write in reply["writes"])


def test_tied_native_targets_stay_native_without_disabling_other_components():
    values = {name: torch.ones((4, 4), dtype=torch.bfloat16) for name in ("gate", "up", "other")}
    model = Native(0, tied=True)
    reply = describe_tp_targets(model, _description(values), 0)
    assert model.other is model.tied
    assert reply["selected_hf"] == ["gate", "up"]


def test_probe_rejects_data_dependent_loader_without_modifying_model():
    class Dependent(Native):
        def load_weights(self, values):
            super().load_weights(values)
            if self.other[0, 0].item():
                self.transform = True

    values = {name: torch.ones((4, 4), dtype=torch.bfloat16) for name in ("gate", "up", "other")}
    model = Dependent(0)
    before = model.fused.detach().clone()
    with pytest.raises(NotImplementedError, match="data-dependent"):
        describe_tp_targets(model, _description(values), 0)
    torch.testing.assert_close(model.fused, before, rtol=0, atol=0)
    assert model.transform is False


@pytest.mark.parametrize("change", ["config", "buffer", "storage"])
def test_probe_rejects_derived_mutable_state_without_touching_original(change):
    class Mutating(Native):
        def __init__(self):
            super().__init__(0)
            self.config = SimpleNamespace(load_calls=0)

        def load_weights(self, values):
            super().load_weights(values)
            if change == "config":
                self.config.load_calls += 1
            elif change == "buffer":
                self.register_buffer("derived", self.other.clone())
            else:
                self.other.data = self.other.clone()

    model = Mutating()
    address = model.other.data_ptr()
    before = model.other.detach().clone()
    values = {name: torch.ones((4, 4), dtype=torch.bfloat16) for name in ("gate", "up", "other")}
    with pytest.raises(NotImplementedError, match="state|storage"):
        describe_tp_targets(model, _description(values), 0)
    assert model.config.load_calls == 0 and not hasattr(model, "derived")
    assert model.other.data_ptr() == address
    torch.testing.assert_close(model.other, before, rtol=0, atol=0)


def test_cross_dtype_buffer_alias_and_native_read_dependency_stay_native():
    class Aliased(Native):
        def __init__(self):
            super().__init__(0)
            self.register_buffer("bits", self.other.view(torch.int16))

    class Dependent(Native):
        def load_weights(self, values):
            self.other.copy_(self.fused[:2])
            super().load_weights([(name, value) for name, value in values if name != "other"])

    values = {name: torch.ones((4, 4), dtype=torch.bfloat16) for name in ("gate", "up", "other")}
    assert describe_tp_targets(Aliased(), _description(values), 0)["selected_hf"] == ["gate", "up"]
    assert describe_tp_targets(Dependent(0), _description(values), 0)["selected_hf"] == []


def test_runtime_group_coordinates_are_detached_and_communication_is_rejected():
    from relax.backends.sglang.tp_shard_layout import _meta_model

    group_type = type("GroupCoordinator", (), {"__module__": "sglang.srt.distributed.parallel_state"})
    group = group_type()
    group.rank = group.local_rank = group.rank_in_group = 0
    group.world_size = 1
    group.is_first_rank = group.is_last_rank = True
    group.cpu_group = object()
    model = Native(0)
    model.pp_group = group
    clone = _meta_model(model)
    assert clone.pp_group is not group and clone.pp_group.world_size == 1 and clone.pp_group.is_last_rank
    with pytest.raises(NotImplementedError, match="runtime handle"):
        clone.pp_group.cpu_group
    values = {name: torch.ones((4, 4), dtype=torch.bfloat16) for name in ("gate", "up", "other")}
    assert describe_tp_targets(model, _description(values), 0)["selected_hf"] == ["gate", "other", "up"]


def test_ordinary_target_geometry_survives_json_roundtrip():
    import json

    from relax.backends.sglang.shard_p2p_weight_loader import validate_expert_targets

    geometry = json.loads(
        json.dumps(dict(expert_enabled=False, source_world_size=1, ordinary_targets={"weight": [2, 4]}))
    )
    write = dict(
        id="0",
        target="tp-0",
        source="weight",
        source_rank=0,
        param="weight",
        src_offset=0,
        dst_offset=0,
        src_stride=16,
        dst_stride=16,
        rows=1,
        row_bytes=16,
    )
    validate_expert_targets([write], {"weight": torch.empty((2, 4), dtype=torch.bfloat16)}, geometry)


def test_nested_native_modules_preserve_graph_and_parameter_inventory():
    from relax.backends.sglang.tp_shard_layout import _meta_model

    class Nested(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16) for _ in range(4)]
            )
            self._relax_native_weight_guard = object()
            self.graph = {"owner": self}
            self.graph["cycle"] = self.graph

        def load_weights(self, values):
            for name, value in values:
                self.layers[int(name[-1])].weight.copy_(value.chunk(2)[0])

    model = Nested()
    expected = set(dict(model.named_parameters()))
    clone = _meta_model(model)
    assert set(dict(clone.named_parameters())) == expected
    assert len(list(clone.modules())) == len(list(model.modules()))
    assert not hasattr(clone, "_relax_native_weight_guard")
    assert clone.graph["owner"] is clone and clone.graph["cycle"] is clone.graph
    values = {f"layer{i}": torch.ones((8, 4), dtype=torch.bfloat16) for i in range(4)}
    reply = describe_tp_targets(model, _description(values), 0)
    assert reply["selected_hf"] == sorted(values) and set(reply["target_shapes"]) == expected
    assert set(dict(model.named_parameters())) == expected
    assert model.graph["owner"] is model and model.graph["cycle"] is model.graph


def test_real_gdn_packed_loader_closure_is_rebound_to_detached_modules():
    from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros((2, 4), dtype=torch.bfloat16), requires_grad=False)
            self.weight.weight_loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(self, self.load)

        def load(self, param, value, loaded_shard_id=None):
            param.copy_(value)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = Projection()

        def load_weights(self, values):
            for _, value in values:
                self.projection.weight.weight_loader(self.projection.weight, value.chunk(2)[0])

    model = Model()
    before = model.projection.weight.detach().clone()
    values = {"projection": torch.ones((4, 4), dtype=torch.bfloat16)}
    reply = describe_tp_targets(model, _description(values), 0)
    assert reply["selected_hf"] == ["projection"] and set(reply["target_shapes"]) == {"projection.weight"}
    torch.testing.assert_close(model.projection.weight, before, rtol=0, atol=0)


def test_builtin_tensor_method_is_rebound_and_module_code_has_no_live_model_reference():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros((2, 4), dtype=torch.bfloat16), requires_grad=False)
            self.copy_weight = self.weight.copy_
            self.activation = torch.relu

        def load_weights(self, values):
            for _, value in values:
                self.copy_weight(value.chunk(2)[0])

    model = Model()
    values = {"weight": torch.ones((4, 4), dtype=torch.bfloat16)}
    reply = describe_tp_targets(model, _description(values), 0)
    assert reply["selected_hf"] == ["weight"]
    torch.testing.assert_close(model.weight, torch.zeros_like(model.weight), rtol=0, atol=0)


def test_registered_torch_operators_are_code_not_opaque_model_owned_capsules():
    model = Native(0)
    model.operator = torch.ops.aten.add.Tensor
    model.packet = torch.ops.aten.add
    values = {name: torch.ones((4, 4), dtype=torch.bfloat16) for name in ("gate", "up", "other")}
    assert describe_tp_targets(model, _description(values), 0)["selected_hf"] == ["gate", "other", "up"]


@pytest.mark.parametrize("write_on_empty", [False, True])
@pytest.mark.parametrize("resolved", [False, True])
def test_native_lazy_cache_refresh_is_qualified_and_preserved(write_on_empty, resolved):
    from sglang.srt.utils.common import LazyValue

    from relax.backends.sglang.shard_p2p_weight_loader import refresh_native_metadata

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros((2, 4), dtype=torch.bfloat16), requires_grad=False)
            self.runtime_cache = LazyValue(lambda: False)
            assert self.runtime_cache.value is False

        def load_weights(self, values):
            for _, value in values:
                self.weight.copy_(value.chunk(2)[0])
            self.lazy = LazyValue(lambda: self.weight.data)
            if not values and write_on_empty:
                self.weight.zero_()

    model = Model()
    values = {"weight": torch.ones((4, 4), dtype=torch.bfloat16)}
    model.load_weights(list(values.items()))
    original_cache = model.lazy
    runtime_cache = model.runtime_cache
    if resolved:
        assert bool(torch.all(original_cache.value == 1))
    if write_on_empty:
        with pytest.raises(NotImplementedError, match="Empty native refresh"):
            describe_tp_targets(model, _description(values), 0)
        assert model.lazy is original_cache and bool(torch.all(model.weight == 1))
        return
    reply = describe_tp_targets(model, _description(values), 0)
    assert reply["selected_hf"] == ["weight"] and reply["native_refresh"] is True
    assert model.lazy is original_cache
    assert model.runtime_cache is runtime_cache and model.runtime_cache.value is False
    stale = model.lazy.value
    model.weight.data = torch.full_like(model.weight, 2)
    assert bool(torch.all(stale == 1))
    refresh_native_metadata(model, reply)
    assert model.lazy is not original_cache and bool(torch.all(model.lazy.value == 2))
    assert model.lazy.value.data_ptr() == model.weight.data_ptr()


@pytest.mark.parametrize("partial_refresh", [False, True])
def test_empty_native_reload_must_actually_invalidate_the_lazy_cache(partial_refresh):
    from sglang.srt.utils.common import LazyValue

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros((2, 4), dtype=torch.bfloat16), requires_grad=False)

        def load_weights(self, values):
            for _, value in values:
                self.weight.copy_(value.chunk(2)[0])
            if values:
                self.lazy = LazyValue(lambda: self.weight.data)
            if values or partial_refresh:
                self.other_cache = LazyValue(lambda: self.weight.data)

    model = Model()
    values = {"weight": torch.ones((4, 4), dtype=torch.bfloat16)}
    model.load_weights(list(values.items()))
    original_cache = model.lazy
    with pytest.raises(NotImplementedError, match="does not invalidate"):
        describe_tp_targets(model, _description(values), 0)
    assert model.lazy is original_cache
    torch.testing.assert_close(model.weight, torch.ones_like(model.weight), rtol=0, atol=0)


def test_an_existing_resolved_runtime_cache_must_not_change_during_native_loading():
    from sglang.srt.utils.common import LazyValue

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros((2, 4), dtype=torch.bfloat16), requires_grad=False)
            self.flag = LazyValue(lambda: False)
            assert self.flag.value is False

        def load_weights(self, values):
            for _, value in values:
                self.weight.copy_(value.chunk(2)[0])
            self.flag._value = True

    model = Model()
    values = {"weight": torch.ones((4, 4), dtype=torch.bfloat16)}
    with pytest.raises(NotImplementedError, match="mutates an existing lazy cache"):
        describe_tp_targets(model, _description(values), 0)
    assert model.flag.value is False
    torch.testing.assert_close(model.weight, torch.zeros_like(model.weight), rtol=0, atol=0)


@pytest.mark.parametrize("lazy_cache,constant_reset", [(False, False), (True, False), (False, True)])
def test_tensor_writes_to_persistent_state_keep_the_dependency_native(lazy_cache, constant_reset):
    from sglang.srt.utils.common import LazyValue

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros((2, 4), dtype=torch.bfloat16), requires_grad=False)
            self.cache = (
                LazyValue(lambda: torch.full_like(self.weight, 7)) if lazy_cache else torch.full_like(self.weight, 7)
            )
            if lazy_cache:
                assert self.cache.value.shape == self.weight.shape

        def load_weights(self, values):
            for _, value in values:
                self.weight.copy_(value.chunk(2)[0])
                cache = self.cache.value if lazy_cache else self.cache
                cache.zero_() if constant_reset else cache.copy_(self.weight)

    model = Model()
    cache = model.cache.value if lazy_cache else model.cache
    values = {"weight": torch.ones((4, 4), dtype=torch.bfloat16)}
    if lazy_cache or constant_reset:
        with pytest.raises(NotImplementedError, match="retains mutable state|without an HF dependency"):
            describe_tp_targets(model, _description(values), 0)
    else:
        reply = describe_tp_targets(model, _description(values), 0)
        assert reply["selected_hf"] == [] and reply["native_refresh"] is False
    assert (model.cache.value if lazy_cache else model.cache) is cache
    torch.testing.assert_close(cache, torch.full_like(cache, 7), rtol=0, atol=0)
    torch.testing.assert_close(model.weight, torch.zeros_like(model.weight), rtol=0, atol=0)


@pytest.mark.parametrize("axis", [0, -2, 1])
def test_bridge_local_regions_preserve_actual_shard_axis_and_storage(axis):
    from relax.backends.megatron.weight_update.tp_shard_source import local_tp_regions

    value = torch.arange(24, dtype=torch.bfloat16).view(6, 4)
    full = [12, 4] if axis in (0, -2) else [6, 8]
    mapping = SimpleNamespace(
        tp_rank=1,
        tp_size=2,
        local_hf_params=lambda weight, **kwargs: [
            SimpleNamespace(
                name="canonical", weight=weight, global_weight_shape=full, shard_dim=axis, shard_group="tp"
            )
        ],
    )
    task = SimpleNamespace(mapping=mapping, global_param_name="source", megatron_module=object())
    (region,) = local_tp_regions("source", value, task, 1, 2)
    assert region["offset"] == ([6, 0] if axis in (0, -2) else [0, 4])
    assert region["shape"] == [6, 4]
    value16 = value.to(torch.float16)
    with pytest.raises(NotImplementedError):
        local_tp_regions("source", value16, task, 1, 2)


def test_ordinary_source_rebinds_fresh_storage_and_requires_reseed_for_changed_regions(monkeypatch):
    from relax.backends.megatron.weight_update import tp_shard_source as module

    source = object.__new__(module.OrdinaryTPSource)
    source.group, source.rank, source.replica_rank = "actor", 0, 0
    source.replica_ranks, source.tp_rank, source.tp_size = (0,), 0, 2
    source.native_specs, source.selected, source._schema = {}, frozenset(), None
    monkeypatch.setattr(module.dist, "get_global_rank", lambda group, rank: rank)
    monkeypatch.setattr(module.dist, "get_process_group_ranks", lambda group: [0])
    monkeypatch.setattr(module.mpu, "get_data_parallel_group", lambda **kwargs: "replicas")
    offsets = {"gate": 0, "up": 4}
    mapping = SimpleNamespace(
        tp_rank=0,
        tp_size=2,
        local_hf_params=lambda weight, **kwargs: [
            SimpleNamespace(
                name=name,
                weight=weight.narrow(0, offset, 4),
                global_weight_shape=(8, 4),
                shard_dim=0,
                shard_group="tp",
            )
            for name, offset in offsets.items()
        ],
    )
    task = SimpleNamespace(mapping=mapping, global_param_name="fused", megatron_module=object())

    def views(weights, *, selected_only):
        # GPU inventory selection is exercised by training; this CPU test
        # exercises the real Bridge rectangles, catalogue and storage binding.
        return weights, module.local_tp_regions("fused", weights["fused"], task, 0, 2), {"fused": ["gate", "up"]}

    source._views = views
    weights = {"fused": torch.arange(32, dtype=torch.bfloat16).view(8, 4)}
    source.observe_native([(name, torch.zeros((8, 4), dtype=torch.bfloat16)) for name in offsets])
    source.catalogue(weights)
    source.selected = frozenset(weights)
    old = source.bind(weights, seed=True)["ordinary:storage-0"]
    fresh = {"fused": weights["fused"].clone() + 64}
    current = source.bind(fresh)["ordinary:storage-0"]
    assert current.data_ptr() == fresh["fused"].data_ptr() != old.data_ptr()
    torch.testing.assert_close(current.view_as(fresh["fused"]), fresh["fused"], rtol=0, atol=0)
    offsets.update(gate=4, up=0)
    with pytest.raises(ValueError, match="reseeding"):
        source.bind(fresh)
    source.catalogue(fresh)
    assert source.bind(fresh, seed=True)["ordinary:storage-0"].data_ptr() == current.data_ptr()
    assert source.bind(fresh)["ordinary:storage-0"].data_ptr() == current.data_ptr()
