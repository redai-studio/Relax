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


@pytest.mark.parametrize(
    "mode,failure",
    [
        ("overlap", None),
        ("early", None),
        ("native", None),
        ("cold_reject", None),
        ("cold_query", None),
        ("serial", None),
        ("initial", None),
        ("initial_ft", None),
        ("initial_reused_ft", None),
        ("initial_session", None),
        ("initial_reject", None),
        ("initial", "prepare"),
        ("initial", "onload"),
        ("initial", "update"),
        ("initial", "begin"),
        ("overlap", "update"),
        ("overlap", "finish"),
        ("overlap", "kv"),
        ("overlap", "teacher"),
    ],
)
def test_retained_weights_release_after_both_readers_and_never_resume_on_failure(monkeypatch, mode, failure):
    from relax.utils.training.tensor_backper import _TensorBackuperNormal

    cold, initial = mode.startswith("cold_"), mode.startswith("initial")
    retain_actor_weights = mode in ("overlap", "early", "serial")
    overlap = mode in ("overlap", "early") or cold or initial
    expected_retained = retain_actor_weights or initial and mode != "initial_reject"
    monkeypatch.setenv("RELAX_COLOCATE_OVERLAP_WEIGHT_BACKUP", str(int(overlap)))
    monkeypatch.setenv("RELAX_P2P_WEIGHT_CONSISTENCY_CHECK", "0")
    monkeypatch.setenv("RELAX_COLOCATE_PINNED_NIXL_TARGET", str(int(initial)))
    monkeypatch.setattr(
        "relax.backends.sglang.sglang_engine._sglang_has_weight_update_session", lambda: mode == "initial_session"
    )
    monkeypatch.setattr(module.device_module, "synchronize", lambda: None)
    monkeypatch.setattr(module.device_module, "current_device", lambda: 0)
    actor = object.__new__(module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        offload_train=True,
        offload_rollout=True,
        use_fault_tolerance=mode.endswith("_ft"),
        use_critic=False,
        colocate=True,
        selective_offload=True,
        distributed_timeout_minutes=5,
        ci_test=False,
        keep_old_actor=False,
    )
    model = torch.nn.Linear(4, 3, bias=False)
    model.register_buffer("persistent", torch.ones(3))
    actor.role = "actor"
    actor.model = [model]
    getter = lambda: named_params_and_buffers(
        actor.args, actor.model, convert_to_global_name=False, include_persistent_buffers=True
    )
    expected = {name: tensor.detach().clone() for name, tensor in getter()}
    actor.weights_backuper = _TensorBackuperNormal(getter)
    actor.weights_backuper._backups["actor"] = expected
    actor._active_model_tag, actor._actor_weights_pending_sync = "actor", False
    actor._resident_sft_params_pending = False
    actor.genrm_manager = None
    monkeypatch.setattr(module.MegatronTrainRayActor, "_per_step_rollout", property(lambda self: True))
    actor._flush_resident_sft_parameters = Mock()
    events = []
    onload_handle = object()
    backup_event = object()
    teacher_handles = [object(), object()]

    def teacher_onload(index):
        events.append(f"teacher_{index}_submit")
        return teacher_handles[index]

    actor.teacher_manager = [
        SimpleNamespace(onload=SimpleNamespace(remote=lambda index=index: teacher_onload(index))) for index in range(2)
    ]
    if initial:
        actor.teacher_manager = []  # Initial retention qualifies a single Actor snapshot without teachers.

    def get_result(value):
        for ref in value if isinstance(value, list) else [value]:
            if ref is onload_handle:
                events.append("weights_ready")
                if failure == "onload":
                    raise RuntimeError("injected initial onload failure")
            for index, handle in enumerate(teacher_handles):
                if ref is handle:
                    events.append(f"teacher_{index}_ready")
                    if failure == "teacher" and index == 1:
                        raise RuntimeError("injected teacher onload failure")
        return value

    def begin(weights, *, overlap, initial_snapshot=None):
        assert overlap == (mode in ("overlap", "early") or initial)
        assert (initial_snapshot is expected) if initial else initial_snapshot is None
        assert set(weights) == set(expected)
        events.append("begin")
        if failure == "begin":
            raise RuntimeError("injected initial H2D failure")
        return backup_event

    def update(*, backup_done=None):
        assert backup_done is (backup_event if retain_actor_weights else None)
        # Simulate a checkpoint having destroyed NCCL groups before this call.
        assert "reload_groups" in events
        assert set(actor.weight_updater.weights_getter()) == set(expected)
        if failure == "update":
            raise RuntimeError("injected update failure")
        events.append("copy_and_close_ack")
        if mode == "early":
            actor.weight_updater._release_source()
            actor.weight_updater._release_source = None
            events.append("postprocess")

    def finish():
        assert "copy_and_close_ack" in events
        if failure == "finish":
            raise RuntimeError("injected D2H completion failure")
        events.append("release_actor")
        return expected

    def prepare_initial():
        assert "begin" in events and "weights_ready" not in events
        actor.weight_updater._colocate_expert_ipc = SimpleNamespace(failed=False)
        events.append("prepare_initial")
        if failure == "prepare":
            raise RuntimeError("injected initial prepare failure")

    actor._train_state_offloader = SimpleNamespace(
        begin_weight_sync=begin,
        initial_weight_sync_supported=lambda weights, snapshot: initial and mode != "initial_reject",
        abort_initial_weight_sync=lambda: events.append("undo_initial"),
        finish_weight_sync=finish,
        disable_during_update=nullcontext,
    )
    actor.weight_updater = SimpleNamespace(
        weight_version=0 if initial else 1,
        weights_getter=lambda: actor.weights_backuper.get("actor"),
        actor_weight_retention_enabled=retain_actor_weights or initial,
        _colocate_tp_ipc_enabled=initial,
        _initial_native_tp_specs=(0, {"weight": {}}) if initial else None,
        actor_weight_retention_active=False,
        _prepare_initial_colocate=prepare_initial,
        update_weights=update,
        resume_generation_after_sync=lambda: events.append("continue"),
    )

    def kv():
        if failure == "kv":
            raise RuntimeError("injected kv failure")
        events.append("kv")

    actor.rollout_manager = SimpleNamespace(
        onload_weights=SimpleNamespace(remote=lambda: events.append("weights") or onload_handle),
        onload_kv=SimpleNamespace(remote=kv),
        get_rollout_engines_and_lock=SimpleNamespace(remote=lambda: ([], None, 0, [], [])),
        get_rollout_id=SimpleNamespace(remote=lambda: 0 if mode == "initial_reused_ft" else -1),
        recover_rollout_engines=SimpleNamespace(remote=lambda: events.append("recover")),
    )
    monkeypatch.setattr(module, "get_gloo_group", lambda: None)
    monkeypatch.setattr("relax.utils.distributed_utils.get_gloo_group", lambda: None)
    monkeypatch.setattr(module.dist, "get_rank", lambda *a, **k: 0)
    monkeypatch.setattr(module.dist, "get_world_size", lambda *a, **k: 1)
    monkeypatch.setattr(module.dist, "all_gather_object", lambda output, value, **k: output.__setitem__(0, value))
    monkeypatch.setattr(module.dist, "barrier", lambda **k: None)
    monkeypatch.setattr(module.dist, "all_reduce", lambda *a, **k: None)
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda *a, **k: None)
    monkeypatch.setattr(module.ray, "get", get_result)
    cleanup = []
    monkeypatch.setattr(
        module,
        "print_memory",
        lambda label, **kwargs: (
            cleanup.append(kwargs["clear_before_print"]) if label == "after update_weights" else None
        ),
    )
    monkeypatch.setattr(module, "reload_process_groups", lambda **k: events.append("reload_groups"))
    monkeypatch.setattr(module, "destroy_process_groups", lambda: None)
    if cold:
        from relax.backends.megatron.weight_update import update_weight_from_tensor as sync_module

        def query():
            if mode == "cold_query":
                raise TimeoutError("receiver query timed out")
            return dict(device="cuda", tp_size=1, custom_weight_loader=[])

        updater = actor.weight_updater
        updater.shard_p2p_requested = False
        updater.actor_weight_retention_requested = True
        updater.args = SimpleNamespace(rollout_num_gpus_per_engine=1)
        updater.quantization_config, updater.lora_enabled, updater._route_experts = None, False, False
        updater.rollout_engines = [SimpleNamespace(get_expert_weight_sync_layout=SimpleNamespace(remote=query))]
        monkeypatch.setattr(sync_module, "get_gloo_group", lambda: None)
        sync_module.UpdateWeightFromTensor._check_receivers_for_actor_weight_retention(
            updater, ["node"], ["node"], [1], [0]
        )
        assert not updater.actor_weight_retention_enabled
        offloader = actor._train_state_offloader
        offloader.offload = lambda: events.append("offload")
        monkeypatch.setattr(module, "clear_memory", lambda **kwargs: None)
        with torch.no_grad():
            model.weight.add_(1)
        module.MegatronTrainRayActor._backup_actor_weights_after_train.__wrapped__(actor)
        assert all(torch.equal(actor.weights_backuper.get("actor")[name], value) for name, value in getter())
        assert not actor._actor_weights_pending_sync
        module.MegatronTrainRayActor.sleep.__wrapped__(actor)
        assert actor.args.selective_offload and actor._train_state_offloader is offloader
    if retain_actor_weights:
        module.MegatronTrainRayActor._backup_actor_weights_after_train.__wrapped__(actor)
        with pytest.raises(RuntimeError, match="Cannot sleep"):
            module.MegatronTrainRayActor.sleep.__wrapped__(actor)
        assert not events  # No memory cleanup or process-group teardown before the guard.
    if failure is not None:
        with pytest.raises(RuntimeError, match="injected"):
            module.MegatronTrainRayActor.update_weights.__wrapped__(actor)
        assert "continue" not in events
        if failure == "begin":
            assert "undo_initial" in events and "copy_and_close_ack" not in events
            assert not actor._actor_weights_pending_sync and not actor.weight_updater.actor_weight_retention_active
            assert actor.weights_backuper.get("actor") is expected
            return
        if failure in ("kv", "teacher"):
            assert actor._actor_weights_pending_sync is False
            assert "release_actor" in events and actor.weights_backuper.get("actor") is expected
            if failure == "teacher":
                assert events.count("teacher_0_ready") == events.count("teacher_1_ready") == 2
            return
        assert "release_actor" not in events
        if failure in ("prepare", "onload"):
            assert actor.weight_updater._colocate_expert_ipc.failed
            assert "copy_and_close_ack" not in events
        assert actor._actor_weights_pending_sync == expected_retained
        if expected_retained:
            with pytest.raises(RuntimeError, match="awaiting completion"):
                actor.weights_backuper.get("actor")
        return
    module.MegatronTrainRayActor.update_weights.__wrapped__(actor)
    if cold:
        assert events.index("offload") < events.index("weights_ready")
    assert events.index("weights_ready") < events.index("copy_and_close_ack") < events.index("kv")
    if mode in ("initial", "initial_ft"):
        assert events.index("prepare_initial") < events.index("weights_ready")
        if mode == "initial_ft":
            assert events.index("recover") < events.index("prepare_initial")
    elif mode in ("initial_session", "initial_reused_ft"):
        assert "prepare_initial" not in events and events.index("weights_ready") < events.index("begin")
        if mode == "initial_reused_ft":
            assert events.index("weights_ready") < events.index("recover")
    if expected_retained:
        assert events.index("begin") < events.index("copy_and_close_ack") < events.index("release_actor")
        assert events.count("release_actor") == 1 and actor.weight_updater._release_source is None
        if mode == "early":
            assert events.index("release_actor") < events.index("postprocess")
        assert events.index("release_actor") < events.index("kv") < events.index("continue")
        if not initial:
            assert events.index("release_actor") < events.index("teacher_0_submit") < events.index("teacher_1_ready")
            assert events.index("teacher_1_ready") < events.index("continue")
    else:
        assert not {"begin", "release_actor", "continue"}.intersection(events)
        if not initial:
            assert events.index("teacher_1_ready") < events.index("copy_and_close_ack")
    assert actor.weights_backuper.get("actor") is expected
    assert actor._actor_weights_pending_sync is False
    assert actor.weight_updater.actor_weight_retention_active is False
    assert cleanup == [not (expected_retained and overlap)]
