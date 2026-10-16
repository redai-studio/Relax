# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""PP weight conversion must not serialize live training state."""

import pickle
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

    monkeypatch.setattr(bridge_converter.dist, "all_gather_object", gather)
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

    monkeypatch.setattr(bridge_converter.dist, "all_gather_object", fail)
    with pytest.raises(RuntimeError, match="collective failed"):
        converter.broadcast_and_apply_configs()
    assert not converter._configs_broadcast_done
