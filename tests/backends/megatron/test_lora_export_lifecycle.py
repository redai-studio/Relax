# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Verify adapter export restores the caller's training residency state."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("suspended", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_export_restores_training_offload_state(monkeypatch, suspended, failure):
    module = pytest.importorskip("relax.backends.megatron.actor", reason="requires complete Megatron actor runtime")
    from relax.backends.megatron import checkpoint

    actor = module.MegatronTrainRayActor.__new__(module.MegatronTrainRayActor)
    actor.role = "actor"
    actor.model = object()
    actor.args = SimpleNamespace(lora_adapter_mode=True, offload_train=True, colocate=False, hybrid=False)
    actor._train_state_suspended = suspended
    events = []

    def wake():
        events.append("wake")
        actor._train_state_suspended = False

    def sleep():
        events.append("sleep")
        actor._train_state_suspended = True

    def export(model, output, args, *, seal):
        assert model is actor.model and not actor._train_state_suspended
        events.append("export")
        if failure:
            raise ValueError("cooperative export failure")
        return {"version_id": "A"}

    actor.wake_up, actor.sleep = wake, sleep
    monkeypatch.setattr(module, "reload_process_groups", lambda: None)
    monkeypatch.setattr(checkpoint, "export_lora_adapter", export)
    options = dict(
        version_id="A", store_dir="/store", base_model_digest="a" * 64, source_step=1, artifact_max_bytes=1024
    )
    if failure:
        with pytest.raises(ValueError, match="cooperative"):
            actor.export_lora_adapter("/staging", **options)
    else:
        assert actor.export_lora_adapter("/staging", **options) == {"version_id": "A"}
    assert events == (["wake", "export", "sleep"] if suspended else ["export"])
    assert actor._train_state_suspended is suspended


def test_export_boundary_consumes_intent_once_before_sleep():
    module = pytest.importorskip("relax.backends.megatron.actor", reason="requires complete Megatron actor runtime")
    actor = module.MegatronTrainRayActor.__new__(module.MegatronTrainRayActor)
    actor._lora_export_request = {"version_id": "B", "output_dir": "/staging"}
    actor._lora_export_result = None
    events = []

    def export(**kwargs):
        events.append(kwargs)
        return {"version_id": kwargs["version_id"], "source_train_step": kwargs["source_step"]}

    actor.export_lora_adapter = export
    actor._export_lora_at_boundary(8)
    actor._export_lora_at_boundary(8)
    assert events == [{"version_id": "B", "output_dir": "/staging", "source_step": 8}]
    assert actor.lora_export_result() == {"version_id": "B", "source_train_step": 8}
    assert actor._lora_export_request is None


@pytest.mark.parametrize("mode", ["sync", "hybrid", "fully_async"])
def test_bootstrap_exports_actor_weights_with_optional_backuper(monkeypatch, mode):
    module = pytest.importorskip("relax.backends.megatron.actor", reason="requires complete Megatron actor runtime")
    actor = module.MegatronTrainRayActor.__new__(module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(fully_async=mode != "sync", hybrid=mode == "hybrid")
    events = []
    actor.model = "actor" if mode == "fully_async" else "ref"

    def restore(tag):
        events.append(("restore", tag))
        actor.model = tag

    if mode != "fully_async":
        actor.weights_backuper = SimpleNamespace(backup_tags={"actor", "ref"}, restore=restore)
        actor._active_model_tag = "ref"
    monkeypatch.setattr(module.device_utils, "maybe_backend_process_on_model_switch", lambda: None)

    def export(**kwargs):
        assert actor.model == "actor"
        events.append(("export", kwargs))
        return {"version_id": kwargs["version_id"], "source_train_step": kwargs["source_step"]}

    actor.export_lora_adapter = export
    descriptor = {"version_id": "A", "output_dir": "/staging"}
    actor._bootstrap_lora_export(descriptor, 8)
    actor._export_lora_at_boundary(8)
    expected = [] if mode == "fully_async" else [("restore", "actor")]
    assert events == expected + [("export", {**descriptor, "source_step": 8})]
    assert actor.lora_export_result() == {"version_id": "A", "source_train_step": 8}
    assert actor._lora_export_request is None
    assert descriptor == {"version_id": "A", "output_dir": "/staging"}


def test_bootstrap_without_export_intent_leaves_weights_untouched():
    module = pytest.importorskip("relax.backends.megatron.actor", reason="requires complete Megatron actor runtime")
    actor = module.MegatronTrainRayActor.__new__(module.MegatronTrainRayActor)
    actor._bootstrap_lora_export(None, 8)
    assert actor._lora_export_request is None
    assert actor.lora_export_result() is None
