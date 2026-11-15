# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the RLOO registry entry."""

from types import SimpleNamespace

import pytest


# `relax.core.registry` eagerly imports `relax.components.advantages`, which
# imports `megatron.core` at module level. Skip when megatron is unavailable.
pytest.importorskip("megatron.core")

from relax.core.registry import ALGOS, ROLES, process_role  # noqa: E402


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


def test_algos_has_rloo_entry_same_roles_as_grpo():
    assert "rloo" in ALGOS
    assert "grpo" in ALGOS
    rloo_keys = set(ALGOS["rloo"].keys())
    grpo_keys = set(ALGOS["grpo"].keys())
    assert rloo_keys == grpo_keys, f"rloo roles {rloo_keys} != grpo roles {grpo_keys}"


def test_algos_rloo_has_no_critic():
    assert ROLES.critic not in ALGOS["rloo"]


def test_algos_rloo_role_classes_match_grpo():
    for role in ALGOS["grpo"]:
        assert ALGOS["rloo"][role] is ALGOS["grpo"][role], f"role {role} class mismatch"


def test_process_role_rloo_sync_is_colocate():
    from relax.core.registry import ROLES_COLOCATE

    roles = process_role(_cfg(advantage_estimator="rloo"))
    assert roles is ROLES_COLOCATE
