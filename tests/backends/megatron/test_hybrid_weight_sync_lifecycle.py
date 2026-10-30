# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.backends.megatron import actor as actor_module
from relax.components.actor import Actor


@pytest.fixture(params=[None, "elastic.yaml"])
def hybrid_actor(monkeypatch, request):
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.args = SimpleNamespace(
        hybrid=True,
        autoscaler_config=request.param,
        fully_async=True,
        loss_type="policy_loss",
        offload_train=True,
        offload_rollout=False,
        use_critic=False,
        debug_train_only=False,
        debug_rollout_only=False,
        distributed_timeout_minutes=7,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        keep_old_actor=False,
    )
    actor.role = "actor"
    actor._flush_resident_sft_parameters = Mock()
    actor._use_dcs_hybrid_weight_sync = True
    state = SimpleNamespace(awake=True, tp_group=object(), updating=False, events=[])

    def offload():
        assert state.awake
        state.awake = False
        state.events.append("offload")

    def reload():
        assert not state.awake, "must not resume an already resident TMS pool"
        state.awake = True
        state.events.append("reload")

    def destroy():
        state.tp_group = None
        state.events.append("destroy_groups")

    def reload_groups(**kwargs):
        assert state.awake
        assert kwargs["timeout_minutes"] == 7
        state.tp_group = object()
        state.events.append("reload_groups")

    @contextmanager
    def disable():
        state.updating = True
        try:
            yield
        finally:
            state.updating = False

    actor._train_state_offloader = SimpleNamespace(offload=offload, reload=reload, disable_during_update=disable)
    actor.set_rollout_manager = Mock()
    actor._advance_keep_old_actor_queue = Mock()
    monkeypatch.setattr(actor_module, "clear_memory", Mock())
    monkeypatch.setattr(actor_module, "print_memory", Mock())
    monkeypatch.setattr(actor_module, "destroy_process_groups", destroy)
    monkeypatch.setattr(actor_module, "reload_process_groups", reload_groups)
    monkeypatch.setattr(
        actor_module,
        "Envs",
        SimpleNamespace(
            RELAX_DEBUG_DSV4_PP_GROUP_WARMUP=False,
            RELAX_DEBUG_DSV4_CP_P2P_WARMUP=False,
        ),
    )
    return actor, state


@pytest.mark.parametrize("offload_train", [False, True])
@pytest.mark.parametrize("fail_sync", [False, True])
def test_hybrid_dcs_restores_explicit_rollout_offload(hybrid_actor, monkeypatch, offload_train, fail_sync):
    actor, state = hybrid_actor
    actor.args.offload_train = offload_train
    actor.args.offload_rollout = True
    actor.genrm_manager = None
    actor.rollout_manager = Mock()
    actor.rollout_manager.onload.remote.side_effect = lambda: state.events.append("onload_weights_and_kv")
    actor.rollout_manager.onload_kv.remote.side_effect = lambda: state.events.append("onload_kv")
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda *a, **kw: 0)
    monkeypatch.setattr(actor_module.dist, "barrier", lambda **kw: None)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: object())
    monkeypatch.setattr(actor_module.ray, "get", lambda refs: refs)
    monkeypatch.setattr(actor_module, "append_managed_opd_teacher_onload_handle", lambda *a: None)
    if offload_train:
        actor.sleep()
    state.events.clear()

    def push(*args, **kwargs):
        assert state.events[-1] == "onload_weights_and_kv"
        state.events.append("push")
        if fail_sync:
            raise RuntimeError("sync failed")

    actor.update_weights_fully_async = push
    if fail_sync:
        with pytest.raises(RuntimeError, match="sync failed"):
            actor.update_weights()
    else:
        actor.update_weights()
    expected = ["onload_weights_and_kv", "push"]
    if offload_train:
        expected = ["reload", "reload_groups", *expected, "destroy_groups", "offload"]
    assert state.events == expected
    actor.rollout_manager.onload_kv.remote.assert_not_called()


@pytest.mark.parametrize("offload_train", [False, True])
@pytest.mark.parametrize("fail_sync", [False, True])
def test_initial_hybrid_dcs_sync_restores_offloaded_state_and_tp_group(hybrid_actor, offload_train, fail_sync):
    actor, state = hybrid_actor
    actor.args.offload_train = offload_train
    if offload_train:
        actor.sleep()  # Same state as the end of Megatron initialization.
    state.events.clear()

    def push(*args, **kwargs):
        assert state.awake and state.tp_group is not None
        assert state.updating, "DCS conversion buffers must be outside TMS tracking"
        assert args == (0,) and kwargs == {"rollout_only": True}
        state.events.append("push")
        if fail_sync:
            raise RuntimeError("sync failed")

    actor.update_weights_fully_async = push
    config = SimpleNamespace(hybrid=True, fully_async=True, autoscaler_config="elastic.yaml", loss_type="policy_loss")
    service = SimpleNamespace(config=config, actor_model=actor)
    if fail_sync:
        with pytest.raises(RuntimeError, match="sync failed"):
            Actor.func_or_class.set_rollout_manager(service, object())
    else:
        Actor.func_or_class.set_rollout_manager(service, object())
    assert state.awake is not offload_train
    assert (state.tp_group is None) is offload_train
    assert not state.updating
    assert state.events == (
        ["reload", "reload_groups", "push", "destroy_groups", "offload"] if offload_train else ["push"]
    )
    assert actor._advance_keep_old_actor_queue.call_count == (0 if fail_sync else 1)


def test_hybrid_training_restores_offload_before_first_parallel_query(hybrid_actor, monkeypatch):
    actor, state = hybrid_actor
    actor.sleep()

    class StopBeforeTraining(Exception):
        pass

    def query(**kwargs):
        assert state.awake and state.tp_group is not None
        raise StopBeforeTraining

    monkeypatch.setattr(actor_module.mpu, "get_data_parallel_world_size", query)
    with pytest.raises(StopBeforeTraining):
        actor.train_hybrid(0)


@pytest.mark.parametrize("save_checkpoint", [False, True])
def test_consecutive_hybrid_steps_sync_after_offload_and_checkpoint(hybrid_actor, monkeypatch, save_checkpoint):
    actor, state = hybrid_actor
    vars(actor.args).update(
        use_rollout_routing_replay=False,
        multimodal_keys=None,
        advantage_estimator="grpo",
        rollout_data_postprocess_path=None,
        use_opd=False,
        compute_advantages_and_returns=False,
        use_routing_replay=False,
        ref_update_interval=None,
        num_rollout=10,
        save="checkpoint" if save_checkpoint else None,
        rotate_ckpt=False,
        save_interval=1,
    )
    actor._active_model_tag = "actor"
    actor.weights_backuper = SimpleNamespace(backup=Mock())
    actor.model = actor.optimizer = actor.opt_param_scheduler = actor.tokenizer = actor.flops_counter = None
    actor.rollout_data_postprocess = None
    actor.prof = SimpleNamespace(step=Mock())
    actor.all_consumed = lambda *_a, **_kw: False
    actor._get_data_from_transfer_queue = lambda *_a: ({"total_lengths": [2, 3]}, None)
    actor._hybrid_forward_subbatch = Mock()
    actor._wait_for_previous_eval = Mock()
    actor._check_services_health = lambda: (True, False)
    actor._run_step_evaluation = Mock()

    def checkpoint(*_a, **_kw):
        assert state.awake and state.tp_group is not None
        state.tp_group = None  # save_model destroys the Megatron groups after saving.
        state.events.append("checkpoint")

    def push(*_a, **_kw):
        assert state.awake and state.tp_group is not None and state.updating
        state.events.append("push")

    def query(**_kw):
        assert state.awake and state.tp_group is not None
        return 2  # TP=2, DP=2; the full training world would contain four ranks.

    actor.save_model = checkpoint
    actor.update_weights_fully_async = push
    monkeypatch.setattr(actor_module.mpu, "get_data_parallel_world_size", query)
    monkeypatch.setattr(actor_module.mpu, "get_data_parallel_group", lambda **_kw: state.tp_group)
    monkeypatch.setattr(
        actor_module.dist,
        "all_gather_object",
        lambda output, value, **_kw: output.__setitem__(slice(None), [value, value]),
    )
    monkeypatch.setattr(actor_module.dist, "barrier", Mock())
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: object())
    monkeypatch.setattr(
        actor_module,
        "build_rollout_minibatch_plan",
        lambda *_a: SimpleNamespace(mini_local_sample_request=2, num_rollout_minis=1),
    )
    monkeypatch.setattr(actor_module, "timer", lambda *_a: nullcontext())
    monkeypatch.setattr(actor_module, "inverse_timer", lambda *_a: nullcontext())
    monkeypatch.setattr(actor_module, "get_data_iterator", lambda *_a: (None, 1))
    monkeypatch.setattr(actor_module, "train", Mock())
    monkeypatch.setattr(actor_module, "log_rollout_data", Mock())
    monkeypatch.setattr(actor_module, "log_perf_data", Mock())
    monkeypatch.setattr(actor_module, "Timer", SimpleNamespace)
    monkeypatch.setattr(actor_module.train_dump_utils, "save_debug_train_data", Mock())
    monkeypatch.setattr(actor_module.tracking_utils, "flush_metrics", Mock())
    monkeypatch.setattr(actor_module, "compute_rollout_step", lambda *_a: 0)
    actor.sleep()
    state.events.clear()
    for step in range(2):
        actor.train_hybrid(step)
        assert not state.awake and state.tp_group is None
    assert state.events.count("push") == 2
    assert state.events.count("checkpoint") == (2 if save_checkpoint else 0)
    assert actor._advance_keep_old_actor_queue.call_count == 2
