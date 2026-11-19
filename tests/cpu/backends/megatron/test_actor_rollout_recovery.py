# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


pytest.importorskip("megatron.core")
try:
    from relax.backends.megatron import actor as actor_module
except (ImportError, AssertionError) as error:
    pytest.skip(f"Megatron actor dependencies unavailable: {error}", allow_module_level=True)


@pytest.mark.parametrize("fails", [False, True])
def test_update_weights_shares_rollout_recovery_result_before_weight_sync(monkeypatch, fails):
    process_group = object()
    current_rank = 0
    shared_status = []
    reached_weight_sync = []
    recovery_ref, engines_ref = object(), object()
    manager = SimpleNamespace(
        recover_rollout_engines=SimpleNamespace(remote=Mock(return_value=recovery_ref)),
        get_rollout_engines_and_lock=SimpleNamespace(remote=Mock(return_value=engines_ref)),
    )
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        offload_train=False,
        offload_rollout=False,
        use_fault_tolerance=True,
    )
    actor.rollout_manager = manager

    class ReachedWeightSync(BaseException):
        pass

    def get(ref):
        if ref is recovery_ref:
            if fails:
                raise RuntimeError("Required PD router registration failed")
            return None
        assert ref is engines_ref
        reached_weight_sync.append(current_rank)
        raise ReachedWeightSync

    def broadcast(status, *, src, group):
        assert src == 0 and group is process_group
        if current_rank == 0:
            shared_status[:] = status
        else:
            status[:] = shared_status

    get_rank = Mock(side_effect=lambda group=None: current_rank)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: process_group)
    monkeypatch.setattr(actor_module.dist, "get_rank", get_rank)
    monkeypatch.setattr(actor_module.dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(actor_module.dist, "barrier", lambda **kwargs: pytest.fail("unmatched recovery barrier"))
    monkeypatch.setattr(actor_module.ray, "get", get)

    for current_rank in (0, 1):
        if fails:
            with pytest.raises(RuntimeError, match="Rollout engine recovery failed:.*Required PD router registration"):
                actor_module.MegatronTrainRayActor.update_weights.__wrapped__(actor)
        else:
            with pytest.raises(ReachedWeightSync):
                actor_module.MegatronTrainRayActor.update_weights.__wrapped__(actor)

    manager.recover_rollout_engines.remote.assert_called_once()
    assert all(call.args == (process_group,) for call in get_rank.call_args_list)
    assert reached_weight_sync == ([] if fails else [0, 1])
    assert manager.get_rollout_engines_and_lock.remote.call_count == (0 if fails else 2)


@pytest.mark.parametrize("after_scale_in", [False, True])
def test_baseline_snapshot_connects_with_legacy_ipc_arguments(monkeypatch, after_scale_in):
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        offload_train=False,
        offload_rollout=False,
        use_fault_tolerance=False,
    )
    actor._colocate_elastic = True
    actor._colocate_receivers = ("old-receivers", "retired-group") if after_scale_in else None
    baseline, lock = Mock(), object()
    actor._acquire_colocate_full = Mock(
        return_value={
            "receivers": [(baseline, 2, 0)],
            "lock": lock,
            "num_new_engines": 0 if after_scale_in else 1,
            "shared_count": 1,
            "group_name": None,
        }
    )
    actor.rollout_manager = SimpleNamespace(clear_num_new_engines=Mock())
    actor._train_state_offloader = SimpleNamespace(disable_during_update=nullcontext)
    actor.weight_updater = Mock()
    actor.weight_updater.update_weights.side_effect = RuntimeError("IPC failed")
    monkeypatch.setattr(actor_module, "run_on_training_rank_zero", lambda function, **kwargs: function())
    monkeypatch.setattr(actor_module, "print_memory", Mock())
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(actor_module.dist, "barrier", Mock())
    monkeypatch.setattr(actor_module.ray, "get", lambda ref: ref)

    with pytest.raises(RuntimeError, match="IPC failed"):
        actor_module.MegatronTrainRayActor.update_weights.__wrapped__(actor)

    actor.weight_updater.connect_rollout_engines.assert_called_once_with(
        [baseline], lock, engine_gpu_counts=[2], engine_gpu_offsets=[0]
    )
    assert actor._colocate_receivers is None


@pytest.mark.parametrize("failing_operation", ["connect_rollout_engines", "update_weights"])
def test_mixed_full_failure_is_reported_before_actor_raises(monkeypatch, failing_operation):
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        offload_train=False,
        offload_rollout=False,
        use_fault_tolerance=False,
    )
    actor._colocate_elastic = True
    actor._colocate_receivers = None
    actor._train_state_offloader = SimpleNamespace(disable_during_update=nullcontext)
    actor._acquire_colocate_full = Mock(
        return_value={
            "receivers": [(Mock(), 2, 0), (Mock(), 2, 0)],
            "shared_count": 1,
            "lock": object(),
            "num_new_engines": 0,
            "group_name": "mixed",
        }
    )
    actor.rollout_manager = SimpleNamespace(fail_colocate_full=Mock())
    actor.weight_updater = Mock()
    getattr(actor.weight_updater, failing_operation).side_effect = RuntimeError("transport failed")
    actor._publish_colocate_full = Mock()
    monkeypatch.setattr(actor_module, "run_on_training_rank_zero", lambda function, **kwargs: function())
    monkeypatch.setattr(actor_module, "print_memory", Mock())
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(actor_module.dist, "barrier", Mock())
    monkeypatch.setattr(actor_module.ray, "get", lambda ref, **kwargs: ref)

    with pytest.raises(RuntimeError, match="transport failed"):
        actor_module.MegatronTrainRayActor.update_weights.__wrapped__(actor)

    actor.rollout_manager.fail_colocate_full.remote.assert_called_once_with("mixed")
    actor._publish_colocate_full.assert_not_called()


def test_colocate_admission_probes_before_requesting_full_snapshot(monkeypatch):
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    engines = [Mock(), Mock()]
    source, final_snapshot = object(), {"receivers": []}
    acquire_full = Mock(side_effect=[({"precheck": (4, engines, 2, 130)}, None), (final_snapshot, None)])
    actor.rollout_manager = SimpleNamespace(
        acquire_colocate_full=SimpleNamespace(remote=acquire_full),
    )
    probe = Mock(return_value=(False, True, ()))
    monkeypatch.setattr(actor_module, "precheck_colocate_receiver", probe)
    monkeypatch.setattr(actor_module, "time", SimpleNamespace(time=lambda: 120))
    monkeypatch.setattr(actor_module.ray, "get_runtime_context", lambda: SimpleNamespace(current_actor=source))
    monkeypatch.setattr(actor_module.ray, "get", lambda ref: ref)

    assert actor._acquire_colocate_full() is final_snapshot
    probe.assert_called_once_with(engines, 2, 10)
    assert [c.args for c in acquire_full.call_args_list] == [(source, None), (source, (4, False, True, ()))]
