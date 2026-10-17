# Copyright (c) 2026 Relax Authors. All Rights Reserved.

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
    begin_ref = object()
    refs = {"recover_rollout_engines": recovery_ref, "get_rollout_engines_and_lock": engines_ref}
    manager = SimpleNamespace(
        begin_weight_update=SimpleNamespace(remote=Mock(return_value=begin_ref)),
        rollout_operation=SimpleNamespace(remote=Mock(side_effect=lambda name: refs[name])),
    )
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        offload_train=False,
        offload_rollout=False,
        use_fault_tolerance=True,
    )
    actor.inference_manager = manager

    class ReachedWeightSync(BaseException):
        pass

    def get(ref):
        if ref is begin_ref:
            return None
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

    operations = [call.args[0] for call in manager.rollout_operation.remote.call_args_list]
    assert operations.count("recover_rollout_engines") == 1
    assert all((*call.args, *call.kwargs.values()) == (process_group,) for call in get_rank.call_args_list)
    assert reached_weight_sync == ([] if fails else [0, 1])
    assert operations.count("get_rollout_engines_and_lock") == (0 if fails else 2)


@pytest.mark.parametrize("rank,actor_fwd_only", [(0, False), (1, False), (0, True)])
def test_fully_async_sync_hands_confirmed_recipients_to_the_manager(monkeypatch, rank, actor_fwd_only):
    synced = ["seed"]
    client = SimpleNamespace(
        update_weights_for_rollout=Mock(return_value=synced),
        init_process_groups_for_actor_fwd_ref=Mock(),
    )
    manager = SimpleNamespace(rollout_operation=SimpleNamespace(remote=Mock(return_value="confirm-ref")))
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(rollout_http_timeout=30)
    actor.checkpoint_engine_client = client
    actor.inference_manager = manager
    actor._weight_sync_lock = None
    get = Mock()
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(actor_module.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(actor_module, "run", lambda value: value)
    monkeypatch.setattr(actor_module, "print_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_module.ray, "get", get)

    update = getattr(actor_module.MegatronTrainRayActor.update_weights_fully_async, "__wrapped__", None)
    update = update or actor_module.MegatronTrainRayActor.update_weights_fully_async
    update(actor, 0, rollout_only=True, actor_fwd_only=actor_fwd_only)

    if rank == 0 and not actor_fwd_only:
        manager.rollout_operation.remote.assert_called_once_with("confirm_weight_sync", synced)
        get.assert_called_once_with("confirm-ref", timeout=30)
    else:
        manager.rollout_operation.remote.assert_not_called()
