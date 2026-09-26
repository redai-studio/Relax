# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for SFT registry entries."""

import importlib
from types import SimpleNamespace

import pytest


# `relax.core.registry` eagerly imports `relax.components.advantages`, which
# imports `megatron.core` at module level. Skip the whole module when megatron
# is unavailable (e.g. GitHub CI without GPU stack).
pytest.importorskip("megatron.core")


def _cfg(**kwargs):
    defaults = dict(
        debug_rollout_only=False,
        debug_train_only=False,
        fully_async=False,
        hybrid=False,
        loss_type="policy_loss",
        advantage_estimator="grpo",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _registry_module():
    return importlib.import_module("relax.core.registry")


def test_algos_has_sft_entry():
    registry = _registry_module()

    assert "sft" in registry.ALGOS
    assert "actor" in registry.ALGOS["sft"]


def test_process_role_returns_sft_only_when_loss_type_sft():
    registry = _registry_module()

    roles = registry.process_role(_cfg(loss_type="sft"))
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


def test_roles_sft_only_includes_actor_and_sft():
    registry = _registry_module()

    assert {r.value for r in registry.ROLES_SFT_ONLY} == {"actor", "sft"}


def test_algos_sft_has_sft_class_wired():
    from relax.components.sft import SFT

    registry = _registry_module()

    assert registry.ALGOS["sft"][registry.ROLES.sft] is SFT


def test_algos_sft_actor_class_unchanged():
    from relax.components.actor import Actor

    registry = _registry_module()

    assert registry.ALGOS["sft"][registry.ROLES.actor] is Actor
