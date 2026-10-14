# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""components/actor.py module-level helpers for SFT-aware partition naming."""

from argparse import Namespace
from unittest.mock import MagicMock

import pytest


def _mk_actor_config(loss_type: str):
    return Namespace(
        loss_type=loss_type,
        dpo_reference_free=False,
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


@pytest.mark.parametrize("loss_type", ["sft", "dpo", "rm"])
def test_actor_helpers_emit_sft_partition_for_offline_training(loss_type):
    from relax.engine.sft.runtime import sft_partition_id, sft_task_name

    cfg = _mk_actor_config(loss_type=loss_type)
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


@pytest.mark.parametrize(
    ("loss_type", "reference_free", "kl_coef", "use_kl_loss", "expected_ref"),
    [
        ("dpo", False, 0.0, False, True),
        ("dpo", True, 0.0, False, False),
        ("policy_loss", False, 0.1, False, True),
        ("policy_loss", False, 0.0, True, True),
        ("sft", False, 0.0, False, False),
    ],
)
def test_actor_initialization_preserves_reference_requirements(
    monkeypatch, loss_type, reference_free, kl_coef, use_kl_loss, expected_ref
):
    from relax.components import actor as actor_module

    config = _mk_actor_config(loss_type)
    config.dpo_reference_free = reference_free
    config.kl_coef = kl_coef
    config.use_kl_loss = use_kl_loss
    group = MagicMock()
    group.init_and_wait.return_value = [0]
    client = MagicMock()
    actor_class = actor_module.Actor.func_or_class.__mro__[1]
    # Ray Serve serializes the ingress class with its own globals.
    init_globals = actor_class.__init__.__globals__
    monkeypatch.setitem(init_globals, "allocate_train_group", lambda **_: group)
    monkeypatch.setitem(init_globals, "attach_tq_client", lambda *_args, **_kwargs: client)

    actor = actor_class(None, None, 1, config, "actor")

    group.init_and_wait.assert_called_once_with(config, role="actor", with_ref=expected_ref, with_opd_teacher=False)
    assert actor.data_system_client is client
