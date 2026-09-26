# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""components/actor.py module-level helpers for SFT-aware partition naming."""

from argparse import Namespace


def _mk_actor_config(loss_type: str):
    return Namespace(
        loss_type=loss_type,
        kl_coef=0.0,
        use_kl_loss=False,
        opd_teacher_load=False,
        fully_async=False,
        colocate=False,
        num_rollout=1,
        save=None,
        save_interval=None,
        rotate_ckpt=False,
        num_critic_only_steps=0,
        start_rollout_id=0,
        tq_config={},
    )


def test_actor_helpers_emit_sft_partition_under_sft_loss_type():
    from relax.engine.sft.runtime import sft_partition_id, sft_task_name

    cfg = _mk_actor_config(loss_type="sft")
    cfg.start_rollout_id = 5

    assert sft_partition_id(cfg, 5) == "sft_5"
    assert sft_task_name(cfg, component="actor") == "sft_train"


def test_actor_helpers_emit_train_partition_under_rl():
    from relax.engine.sft.runtime import sft_partition_id, sft_task_name

    rl_cfg = _mk_actor_config(loss_type="policy_loss")
    rl_cfg.start_rollout_id = 5

    assert sft_partition_id(rl_cfg, 5) == "train_5"
    # NOTE: components/actor.py uses "train_actor" historically; preserve that for RL.
    assert sft_task_name(rl_cfg, component="actor") == "train_actor"


def test_actor_resume_uses_backend_step_over_auto_cold_start():
    from relax.components.actor import _resolve_start_rollout_id

    assert _resolve_start_rollout_id(0, 200, explicitly_set=False) == 200


def test_actor_cold_start_and_explicit_step_are_preserved():
    from relax.components.actor import _resolve_start_rollout_id

    assert _resolve_start_rollout_id(0, 1, explicitly_set=False) == 0
    assert _resolve_start_rollout_id(0, 200, explicitly_set=True) == 0
    assert _resolve_start_rollout_id(100, 200, explicitly_set=True) == 100
    assert _resolve_start_rollout_id(None, 200, explicitly_set=False) == 200
