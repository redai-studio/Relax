# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""controller.py training_loop guard for rollout-absent (SFT-only) mode."""

import pytest


# Importing relax.core.controller transitively pulls in CUDA-only deps
# (deep_gemm asserts on CUDA_HOME at import time). Skip the whole module
# on CPU-only envs — matches the pattern used in
# tests/backends/megatron/test_sft_train_data_fields.py.
try:
    from relax.core.controller import _needs_rollout_worker_setup  # noqa: F401
except (ImportError, AssertionError) as _exc:
    pytest.skip(f"relax.core.controller requires CUDA env: {_exc}", allow_module_level=True)


def test_needs_rollout_worker_setup_false_for_sft_serve_dict():
    """ROLES_SFT_ONLY = {sft, actor}; serve_dict has no 'rollout' entry."""
    from relax.core.controller import _needs_rollout_worker_setup
    from relax.core.registry import ROLES

    sft_serve_dict = {ROLES.sft: object(), ROLES.actor: object()}
    assert _needs_rollout_worker_setup(sft_serve_dict) is False


def test_needs_rollout_worker_setup_true_for_rl_serve_dict():
    from relax.core.controller import _needs_rollout_worker_setup
    from relax.core.registry import ROLES

    rl_serve_dict = {
        ROLES.rollout: object(),
        ROLES.actor: object(),
        ROLES.advantages: object(),
    }
    assert _needs_rollout_worker_setup(rl_serve_dict) is True
