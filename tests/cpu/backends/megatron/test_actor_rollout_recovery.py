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
