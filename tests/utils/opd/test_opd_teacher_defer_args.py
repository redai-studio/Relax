# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""``--opd-teacher-defer``: the flag, and the colocate layouts it allows."""

from __future__ import annotations

import argparse
from argparse import Namespace

import pytest

from relax.utils.opd.opd_utils import add_opd_arguments, validate_managed_opd_teacher_colocate_args


def _colocate_args(*, rollout: int, teacher: int, actor: int = 8, **overrides) -> Namespace:
    values = dict(
        offload_train=None,
        offload_rollout=None,
        rollout_num_gpus=rollout,
        actor_num_gpus_per_node=actor,
        actor_num_nodes=1,
        use_critic=False,
        opd_teacher_defer=False,
        resource={"actor": [1, actor], "rollout": [1, rollout], "teacher": [1, teacher]},
    )
    values.update(overrides)
    return Namespace(**values)


def test_opd_teacher_defer_flag_is_off_by_default():
    parser = add_opd_arguments(argparse.ArgumentParser())

    assert parser.parse_args([]).opd_teacher_defer is False
    assert parser.parse_args(["--opd-teacher-defer"]).opd_teacher_defer is True


def test_opd_teacher_defer_accepts_shared_bundles():
    args = _colocate_args(rollout=8, teacher=8, opd_teacher_defer=True)

    validate_managed_opd_teacher_colocate_args(args)

    # Colocate defaults are still applied.
    assert (args.offload_train, args.offload_rollout) == (True, True)


def test_opd_teacher_shared_bundles_rejected_without_defer():
    with pytest.raises(ValueError, match="requires split bundles") as excinfo:
        validate_managed_opd_teacher_colocate_args(_colocate_args(rollout=8, teacher=8))

    # The error says how the layout could be made valid.
    assert "--opd-teacher-defer" in str(excinfo.value)


def test_opd_teacher_defer_keeps_split_layout_valid():
    validate_managed_opd_teacher_colocate_args(_colocate_args(rollout=4, teacher=4, opd_teacher_defer=True))
    validate_managed_opd_teacher_colocate_args(_colocate_args(rollout=4, teacher=4))


@pytest.mark.parametrize(("rollout", "teacher"), [(8, 4), (4, 8), (2, 2)])
def test_opd_teacher_defer_rejects_layouts_that_are_neither_split_nor_shared(rollout, teacher):
    with pytest.raises(ValueError, match="requires split bundles"):
        validate_managed_opd_teacher_colocate_args(
            _colocate_args(rollout=rollout, teacher=teacher, opd_teacher_defer=True)
        )


def test_opd_teacher_defer_counts_the_critic_towards_the_actor_total():
    args = _colocate_args(
        rollout=8,
        teacher=8,
        actor=4,
        opd_teacher_defer=True,
        use_critic=True,
        critic_num_gpus_per_node=4,
        critic_num_nodes=1,
    )

    validate_managed_opd_teacher_colocate_args(args)
