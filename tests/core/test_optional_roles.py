# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Optional role registration helpers."""

from argparse import Namespace

import pytest


def test_register_sft_rollout_adds_rollout_when_enabled():
    from relax.core.optional_roles import register_sft_rollout

    try:
        from relax.components.rollout import Rollout
        from relax.core.registry import ROLES
    except (ImportError, AssertionError) as exc:
        pytest.skip(f"relax.components.rollout unavailable: {exc}")

    config = Namespace(loss_type="sft", sft_predict_interval=10)
    algo: dict = {}

    extras = register_sft_rollout(config, algo)

    assert extras == [ROLES.rollout]
    assert algo[ROLES.rollout] is Rollout


def test_register_sft_rollout_noop_without_flag():
    from relax.core.optional_roles import register_sft_rollout

    config = Namespace(loss_type="sft", sft_predict_interval=None)
    algo: dict = {}

    assert register_sft_rollout(config, algo) == []
    assert algo == {}


@pytest.mark.parametrize("loss_type", ["policy_loss", "dpo", "rm"])
def test_register_sft_rollout_noop_for_non_sft_algorithms(loss_type):
    """Only ordinary SFT supports the generation prediction hook."""
    from relax.core.optional_roles import register_sft_rollout

    config = Namespace(loss_type=loss_type, sft_predict_interval=10)
    algo: dict = {}

    assert register_sft_rollout(config, algo) == []
    assert algo == {}
