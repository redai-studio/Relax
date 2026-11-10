# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""OpdManager._raise_if_all_failed: raise only when every non-empty sample fails.

Refactored: the old module-level ``raise_if_all_teacher_fetches_failed(args,
samples, results)`` became the staticmethod
``OpdManager._raise_if_all_failed(samples, results)`` (no ``args``); a sample
is "eligible" purely by ``response_length > 0``.
"""

import pytest

from relax.engine.rollout.on_policy_distillation import OpdManager
from relax.utils.types import Sample


def test_raise_when_all_nonempty_failed() -> None:
    samples = [Sample(index=10, response_length=2), Sample(index=11, response_length=3)]
    with pytest.raises(RuntimeError, match="All OPD teacher fetches failed"):
        OpdManager._raise_if_all_failed(samples, [False, False])


def test_no_raise_on_partial_success() -> None:
    samples = [Sample(index=10, response_length=2), Sample(index=11, response_length=3)]
    OpdManager._raise_if_all_failed(samples, [False, True])  # must not raise


def test_no_raise_when_only_empty_samples() -> None:
    # response_length == 0 samples are not eligible, so an all-False result is fine.
    OpdManager._raise_if_all_failed([Sample(index=10, response_length=0)], [False])
