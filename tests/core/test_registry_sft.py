# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for SFT registry entries."""

import importlib

import pytest

from tests.support.args import build_relax_args


# `relax.core.registry` eagerly imports `relax.components.advantages`, which
# imports `megatron.core` at module level. Skip the whole module when megatron
# is unavailable (e.g. GitHub CI without GPU stack).
pytest.importorskip("megatron.core")


def _cfg(**kwargs):
    # Defaults only: callers may override any of these via kwargs without
    # triggering a duplicate-keyword error in the builder.
    merged = {
        "debug_rollout_only": False,
        "debug_train_only": False,
        "fully_async": False,
        "hybrid": False,
        "loss_type": "policy_loss",
        "advantage_estimator": "grpo",
    }
    merged.update(kwargs)
    return build_relax_args(**merged)


def _registry_module():
    return importlib.import_module("relax.core.registry")


@pytest.mark.parametrize("loss_type", ["sft", "dpo", "rm"])
def test_process_role_returns_sft_only_for_offline_training(loss_type):
    registry = _registry_module()

    roles = registry.process_role(_cfg(loss_type=loss_type))
    assert roles is registry.ROLES_SFT_ONLY
    assert {r.value for r in roles} == {"actor", "sft"}


def test_process_role_keeps_rl_path_unchanged():
    registry = _registry_module()

    roles = registry.process_role(_cfg(loss_type="policy_loss"))
    assert roles is registry.ROLES_COLOCATE


def test_process_role_debug_flags_take_precedence_over_sft():
    """debug_rollout_only / debug_train_only 仍优先于 sft（保留 RL 调试惯例）。"""
    registry = _registry_module()

    assert registry.process_role(_cfg(debug_rollout_only=True, loss_type="sft")) is registry.ROLES_ROLLOUT_ONLY
    assert registry.process_role(_cfg(debug_train_only=True, loss_type="sft")) is registry.ROLES_TRAIN_ONLY


def test_roles_main_enum_includes_sft():
    registry = _registry_module()

    assert registry.ROLES.sft.value == "sft"


def test_algos_sft_has_sft_class_wired():
    from relax.components.sft import SFT

    registry = _registry_module()

    assert registry.ALGOS["sft"][registry.ROLES.sft] is SFT


def test_algos_sft_actor_class_unchanged():
    from relax.components.actor import Actor

    registry = _registry_module()

    assert registry.ALGOS["sft"][registry.ROLES.actor] is Actor


def test_image_sft_selects_producer_without_mutating_text_registry():
    from relax.components.image_sft import ImageSFT
    from relax.components.sft import SFT

    registry = _registry_module()
    image = registry.resolve_algorithm(_cfg(loss_type="sft", generation_task="i2i"))
    text = registry.resolve_algorithm(_cfg(loss_type="sft"))
    assert image[registry.ROLES.sft] is ImageSFT
    assert text[registry.ROLES.sft] is SFT
    assert registry.ALGOS["sft"][registry.ROLES.sft] is SFT
