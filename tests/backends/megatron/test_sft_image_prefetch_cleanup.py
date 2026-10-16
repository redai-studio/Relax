# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise resource ownership without importing Megatron or PyTorch."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


_ROOT = Path(__file__).resolve().parents[3]


def _load_node(path, name, *, owner=None):
    source = _ROOT / path
    nodes = ast.parse(source.read_text()).body
    if owner:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner).body
    node = next(node for node in nodes if getattr(node, "name", None) == name)
    module = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], node], type_ignores=[])
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def helper():
    cls = _load_node("relax/engine/sft/image_prefetch.py", "SFTImagePrefetch")
    return cls(SimpleNamespace(), Mock())


def test_shutdown_cleans_pool_created_by_finishing_fetch(helper):
    shutdown = _load_node(
        "relax/backends/megatron/actor.py", "_shutdown_sft_train_prefetch", owner="MegatronTrainRayActor"
    )
    pool = Mock()
    future = Mock()

    def finish_fetch(**kwargs):
        assert kwargs == {"wait": True, "cancel_futures": True}
        future.cancel.assert_called_once_with()
        # A fetch already running cannot be cancelled and may still create
        # the lazy processor pool and publish its image features while joined.
        helper.pool = pool
        helper.features[7] = [object()]

    executor = Mock()
    executor.shutdown.side_effect = finish_fetch
    actor = SimpleNamespace(
        _sft_train_prefetch=future,
        _sft_train_prefetch_rollout_id=7,
        _sft_train_prefetch_executor=executor,
        _sft_image_prefetch=helper,
    )
    shutdown(actor)
    shutdown(actor)

    pool.shutdown.assert_called_once_with(wait=True)
    assert helper.pool is None
    assert helper.features == {}
    assert actor._sft_train_prefetch is None
    assert actor._sft_train_prefetch_rollout_id is None
    assert actor._sft_train_prefetch_executor is None


def test_close_is_idempotent_and_can_release_later_eval_pool(helper):
    first, second = Mock(), Mock()
    helper.pool = first
    helper.close()
    helper.close()
    # Eval may rebuild synchronously after the last train prefetch shuts down.
    helper.pool = second
    helper.features[9] = [object()]
    helper.close()
    first.shutdown.assert_called_once_with(wait=True)
    second.shutdown.assert_called_once_with(wait=True)
    assert helper.features == {}


def test_discard_only_releases_the_completed_stale_batch(helper):
    current = [object()]
    helper.features = {4: [object()], 5: current}
    helper.discard(4)
    helper.discard(4)
    assert helper.features == {5: current}
