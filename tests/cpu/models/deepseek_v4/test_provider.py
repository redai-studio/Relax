# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regression for the DeepSeek-V4 indexer QAT / replay guard.

``install_indexer_qat`` checks ``config.relax_use_indexer_replay``, but that
flag is only set later by ``fill_indexer_replay``, so the check could be
bypassed on the normal initialization path. The provider now rejects the
combination from the launch argument before building the model.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from relax.models.deepseek_v4.provider import _validate_indexer_qat_replay, model_provider


pytestmark = pytest.mark.cpu


def _patch_provider_deps(monkeypatch, *, replay: bool):
    """Stub the provider's lazy imports and record the build order."""
    calls = []
    model = object()

    megatron = ModuleType("megatron")
    training = ModuleType("megatron.training")
    global_vars = ModuleType("megatron.training.global_vars")
    global_vars.get_args = lambda: SimpleNamespace(use_rollout_indexer_replay=replay, hf_checkpoint="ckpt")

    fp8_weights = ModuleType("relax.models.deepseek_v4.fp8_weights")
    fp8_weights.install_native_fp8_weights = lambda *_args, **_kwargs: calls.append("fp8") or ()
    indexer = ModuleType("relax.models.deepseek_v4.indexer")
    indexer.select_process_mode = lambda **_kwargs: calls.append("select_process_mode")
    indexer.install_indexer_qat = lambda *_args, **_kwargs: calls.append("qat") or ()
    qat = ModuleType("relax.models.deepseek_v4.qat")
    qat._fp4_mode = lambda: "native"
    qat.model_provider = lambda **_kwargs: calls.append("build") or model

    env = ModuleType("relax.utils.env")
    env.Envs = SimpleNamespace(RELAX_DSV4_FP4_INDEXER=True, RELAX_DSV4_FP4_BF16_SCORES=False)
    logging_utils = ModuleType("relax.utils.logging_utils")
    logging_utils.get_logger = lambda _name: SimpleNamespace(info=lambda *_args, **_kwargs: None)

    for name, module in {
        "megatron": megatron,
        "megatron.training": training,
        "megatron.training.global_vars": global_vars,
        "relax.models.deepseek_v4.fp8_weights": fp8_weights,
        "relax.models.deepseek_v4.indexer": indexer,
        "relax.models.deepseek_v4.qat": qat,
        "relax.utils.env": env,
        "relax.utils.logging_utils": logging_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls, model


def test_native_indexer_qat_rejects_rollout_indexer_replay():
    with pytest.raises(ValueError, match="cannot be combined with --use-rollout-indexer-replay"):
        _validate_indexer_qat_replay(enabled=True, use_rollout_indexer_replay=True)


@pytest.mark.parametrize(
    ("enabled", "replay"),
    [(True, False), (False, True), (False, False)],
)
def test_guard_allows_supported_combinations(enabled, replay):
    _validate_indexer_qat_replay(enabled=enabled, use_rollout_indexer_replay=replay)


def test_model_provider_rejects_before_building(monkeypatch):
    calls, _ = _patch_provider_deps(monkeypatch, replay=True)

    with pytest.raises(ValueError, match="cannot be combined with --use-rollout-indexer-replay"):
        model_provider()

    assert calls == []


def test_model_provider_builds_when_replay_is_disabled(monkeypatch):
    calls, model = _patch_provider_deps(monkeypatch, replay=False)

    assert model_provider() is model
    assert calls[0] == "select_process_mode"
    assert "build" in calls
