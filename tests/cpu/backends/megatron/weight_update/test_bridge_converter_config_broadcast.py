# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""PP weight conversion must not serialize live training state."""

import pickle
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest


pytest.importorskip("megatron.core")
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402

from relax.backends.megatron.weight_update import bridge_converter  # noqa: E402


class _RuntimeState:
    def __reduce__(self):
        raise TypeError("cannot pickle rank-local state")

    def scale_loss(self, value):
        return value


@dataclass
class _Task:
    megatron_module: object = None


def test_config_broadcast_removes_runtime_state_and_preserves_local_config(monkeypatch):
    state = _RuntimeState()
    groups = ProcessGroupCollection(tp=state)
    config = SimpleNamespace(
        hidden_size=7168, num_attention_heads=96, _pg_collection=groups, grad_scale_func=state.scale_loss
    )
    remote_config = SimpleNamespace(hidden_size=1152, num_attention_heads=16)
    converter = bridge_converter.BridgeConverter.__new__(bridge_converter.BridgeConverter)
    converter._configs_broadcast_done = False
    converter._config_map = {"language_model": config}
    converter._bridge_task_map = {"vision_model.weight": _Task()}
    group = object()
    calls = []
    monkeypatch.setattr(bridge_converter.mpu, "get_pipeline_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(bridge_converter.mpu, "get_pipeline_model_parallel_group", lambda: group)

    def gather(*, obj, object_list, group):
        calls.append(group)
        wire = pickle.loads(pickle.dumps(obj))
        assert wire["language_model"].hidden_size == 7168
        assert wire["language_model"].num_attention_heads == 96
        assert wire["language_model"]._pg_collection is None
        assert wire["language_model"].grad_scale_func is None
        object_list[:] = [wire, {"vision_model": remote_config}]

    monkeypatch.setattr(bridge_converter.traffic, "all_gather_object", gather)
    converter.broadcast_and_apply_configs()
    converter.broadcast_and_apply_configs()
    assert calls == [group]
    assert converter._config_map["language_model"] is config
    assert config._pg_collection is groups
    assert config.grad_scale_func.__self__ is state
    assert converter._bridge_task_map["vision_model.weight"].megatron_module.config is remote_config


def test_config_broadcast_failure_does_not_mark_complete(monkeypatch):
    converter = bridge_converter.BridgeConverter.__new__(bridge_converter.BridgeConverter)
    converter._configs_broadcast_done = False
    converter._config_map = {}
    converter._bridge_task_map = {}
    monkeypatch.setattr(bridge_converter.mpu, "get_pipeline_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(bridge_converter.mpu, "get_pipeline_model_parallel_group", lambda: object())

    def fail(**kwargs):
        raise RuntimeError("collective failed")

    monkeypatch.setattr(bridge_converter.traffic, "all_gather_object", fail)
    with pytest.raises(RuntimeError, match="collective failed"):
        converter.broadcast_and_apply_configs()
    assert not converter._configs_broadcast_done


@pytest.mark.parametrize("bias", [False, True])
def test_registry_qkv_task_uses_gathered_tensor_without_local_module(monkeypatch, bias):
    import torch
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
    from megatron.bridge.models.conversion.param_mapping import QKVMapping, ReplicatedMapping

    from relax.utils import megatron_bridge_utils

    suffix = "bias" if bias else "weight"
    name = f"language_model.decoder.layers.0.self_attention.linear_qkv.{suffix}"
    mapping = QKVMapping(name, "q", "k", "v")
    config = SimpleNamespace(hidden_size=8, num_attention_heads=4, num_query_groups=2, kv_channels=2)
    donor = WeightConversionTask(
        param_name="language_model.norm.weight",
        global_param_name="language_model.norm.weight",
        mapping=ReplicatedMapping("language_model.norm.weight", "norm.weight"),
        megatron_module=SimpleNamespace(config=config),
        param_weight=torch.nn.Parameter(torch.ones(8)),
    )
    # Bridge can omit a local module while its registry still maps the source
    # parameter. This follows init_tasks' production fallback, not a prebuilt
    # task map that skips the failing initialization.
    packed = torch.arange(16 if bias else 128, dtype=torch.float32)
    if not bias:
        packed = packed.reshape(16, 8)
    parameter = torch.nn.Parameter(packed, requires_grad=False)
    registry = SimpleNamespace(megatron_to_hf_lookup=lambda key: mapping if key == name else None)
    bridge = SimpleNamespace(
        get_conversion_tasks=lambda models: [donor],
        _model_bridge=SimpleNamespace(mapping_registry=lambda: registry),
    )
    monkeypatch.setattr(AutoBridge, "from_hf_pretrained", lambda *args, **kwargs: bridge)
    monkeypatch.setattr(megatron_bridge_utils, "patch_megatron_model", lambda models: nullcontext())
    monkeypatch.setattr(bridge_converter, "named_params_and_buffers", lambda *args, **kwargs: [(name, parameter)])
    converter = bridge_converter.BridgeConverter(
        SimpleNamespace(hf_checkpoint="unused", vocab_size=32), [], quantization_config=None
    )
    actual = dict(converter.convert(name, packed))
    expected = {
        "q": torch.cat((packed[:4], packed[8:12])),
        "k": torch.cat((packed[4:6], packed[12:14])),
        "v": torch.cat((packed[6:8], packed[14:16])),
    }
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
