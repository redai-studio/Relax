# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


pytest.importorskip("megatron.core")
try:
    from relax.backends.megatron import actor as module
    from relax.backends.megatron.weight_update.common import named_params_and_buffers
except (ImportError, AssertionError) as error:
    pytest.skip(f"Megatron actor dependencies unavailable: {error}", allow_module_level=True)


@pytest.mark.parametrize("fused,keep_groups", [(True, False), (True, True), (False, True)])
@pytest.mark.parametrize("light_cleanup", [False, True])
@pytest.mark.parametrize("failure", [None, "begin", "update", "weights", "kv"])
def test_forward_update_restores_checkpoint_groups_and_retains_only_after_fused_sync(
    monkeypatch, fused, keep_groups, light_cleanup, failure
):
    from relax.utils.training.tensor_backper import _TensorBackuperNormal

    monkeypatch.setenv("RELAX_COLOCATE_KEEP_PROCESS_GROUPS", str(int(keep_groups)))
    monkeypatch.setenv("RELAX_COLOCATE_FUSED_WEIGHT_SYNC", "1")
    monkeypatch.setenv("RELAX_COLOCATE_LIGHT_CLEANUP", str(int(light_cleanup)))
    device_fences = Mock()
    monkeypatch.setattr(module.device_module, "synchronize", device_fences)
    monkeypatch.setattr(module.device_module, "current_device", lambda: 0)
    actor = object.__new__(module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        offload_train=True,
        offload_rollout=True,
        use_fault_tolerance=False,
        use_critic=False,
        colocate=True,
        distributed_timeout_minutes=5,
        ci_test=False,
        keep_old_actor=False,
    )
    model = torch.nn.Linear(4, 3, bias=False)
    model.register_buffer("persistent", torch.ones(3))
    actor.model = [model]
    getter = lambda: named_params_and_buffers(
        actor.args, actor.model, convert_to_global_name=False, include_persistent_buffers=True
    )
    expected = {name: tensor.detach().clone() for name, tensor in getter()}
    assert all(name.startswith("vp_stages.0.") for name in expected)
    actor.weights_backuper = _TensorBackuperNormal(getter)
    actor.weights_backuper._backups["actor"] = expected
    if fused:
        actor.weights_backuper.mark_actor_dirty()
    actor._active_model_tag, actor._fused_forward_pending = "actor", fused
    actor.genrm_manager = None
    monkeypatch.setattr(module.MegatronTrainRayActor, "_per_step_rollout", property(lambda self: True))
    actor._flush_resident_sft_parameters = Mock()
    events = []
    onload_handle = object()

    def get_result(value):
        if value is onload_handle or (isinstance(value, list) and value == [onload_handle]):
            events.append("weights_ready")
            if failure == "weights":
                raise RuntimeError("injected weights failure")
        return value

    def begin(weights):
        assert set(weights) == set(expected)
        events.append("begin")
        if failure == "begin":
            raise RuntimeError("injected begin failure")

    def update():
        # Simulate a checkpoint having destroyed NCCL groups before this call.
        assert "reload_groups" in events
        assert set(actor.weight_updater.weights_getter()) == set(expected)
        events.append("copy_and_close_ack")
        if failure == "update":
            raise RuntimeError("injected update failure")

    actor._train_state_offloader = SimpleNamespace(
        begin_forward_sync=begin,
        finish_forward_sync=lambda: events.append("release_actor") or expected,
        disable_during_update=nullcontext,
    )
    actor.weight_updater = SimpleNamespace(
        weight_version=1,
        weights_getter=lambda: expected,
        fused_forward_active=False,
        defer_generation=False,
        update_weights=update,
        resume_fused_generation=lambda: (
            events.append("continue"),
            setattr(actor.weight_updater, "defer_generation", False),
        ),
    )

    def kv():
        if failure == "kv":
            raise RuntimeError("injected kv failure")
        events.append("kv")

    actor.rollout_manager = SimpleNamespace(
        onload_weights=SimpleNamespace(remote=lambda: events.append("weights") or onload_handle),
        onload_kv=SimpleNamespace(remote=kv),
        get_rollout_engines_and_lock=SimpleNamespace(remote=lambda: ([], None, 0, [], [])),
    )
    monkeypatch.setattr(module, "get_gloo_group", lambda: None)
    monkeypatch.setattr("relax.utils.distributed_utils.get_gloo_group", lambda: None)
    monkeypatch.setattr(module.dist, "get_rank", lambda *a, **k: 0)
    monkeypatch.setattr(module.dist, "get_world_size", lambda *a, **k: 1)
    monkeypatch.setattr(module.dist, "all_gather_object", lambda output, value, **k: output.__setitem__(0, value))
    monkeypatch.setattr(module.dist, "barrier", lambda **k: None)
    monkeypatch.setattr(module.ray, "get", get_result)
    memory_log = Mock()
    monkeypatch.setattr(module, "print_memory", memory_log)
    monkeypatch.setattr(module, "append_managed_opd_teacher_onload_handle", lambda *a: None)
    monkeypatch.setattr(module, "reload_process_groups", lambda **k: events.append("reload_groups"))
    monkeypatch.setattr(module, "destroy_process_groups", lambda: events.append("destroy_groups"))
    if failure in ("update", "weights", "kv") or (failure == "begin" and fused):
        with pytest.raises(RuntimeError, match="injected"):
            module.MegatronTrainRayActor.update_weights.__wrapped__(actor)
        assert "continue" not in events
        if failure == "kv":
            assert actor._fused_forward_pending is False
            return
        assert "release_actor" not in events
        assert actor._fused_forward_pending == fused
        if fused:
            with pytest.raises(RuntimeError, match="awaiting completion"):
                actor.weights_backuper.get("actor")
        return
    module.MegatronTrainRayActor.update_weights.__wrapped__(actor)
    expected_events = (["begin"] if fused else []) + ["weights"]
    expected_events.append("weights_ready")
    expected_events += ["reload_groups", "copy_and_close_ack"]
    if fused:
        expected_events.append("release_actor")
    if not (fused and keep_groups):
        expected_events.append("destroy_groups")
    expected_events.append("kv")
    if fused:
        expected_events.append("continue")
    assert events == expected_events
    assert actor.weights_backuper.get("actor") is expected
    assert actor._fused_forward_pending is False
    assert actor.weight_updater.fused_forward_active is False
    assert actor.weight_updater.defer_generation is False
    assert device_fences.call_count == int(fused) + int(fused and light_cleanup)
    assert memory_log.call_args.kwargs["clear_before_print"] == (not (fused and light_cleanup))
