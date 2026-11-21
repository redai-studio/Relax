# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for consumed weight version source selection in the actor."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


try:
    from relax.backends.megatron import actor as actor_module
except (ImportError, AssertionError) as exc:
    pytest.skip(f"relax.backends.megatron.actor unavailable: {exc}", allow_module_level=True)


def test_consumed_weight_version_helper_selects_source_by_mode():
    # Colocate/hybrid: the tensor-backuper weight_updater owns the counter.
    assert actor_module._consumed_weight_version(SimpleNamespace(weight_version=7), None) == "7"
    # Pure fully-async: the DCS backend owns the counter the rollout engine
    # echoes back in meta_info; join must use the same sequence.
    engine_client = SimpleNamespace(backend=SimpleNamespace(weight_version=3))
    assert actor_module._consumed_weight_version(None, engine_client) == "3"
    # Offline SFT: no weight publisher exists; the consumption point is unknown.
    assert actor_module._consumed_weight_version(None, None) is None
    # Backend not yet initialized carries no version.
    assert actor_module._consumed_weight_version(None, SimpleNamespace(backend=SimpleNamespace())) is None
