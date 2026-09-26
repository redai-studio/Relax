# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the ``--lora-scope`` guard on LoRA adapter mode.

SGLang hosts LoRA on language-model layers only, so a vision-tower adapter is
trained and then dropped on the way to the rollout engine without any log line.
The guard has to fail loud, and — just as importantly — must NOT fire on the
configurations that are actually safe (merge mode, explicit language scope), or
it would block legitimate runs.
"""

import argparse
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture()
def arguments_module(monkeypatch):
    """Import ``relax.utils.arguments`` with its heavy deps stubbed out."""
    router_pkg = ModuleType("sglang_router")
    launch_router = ModuleType("sglang_router.launch_router")
    launch_router.RouterArgs = object
    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch_router)

    sglang_arguments = ModuleType("relax.backends.sglang.arguments")
    sglang_arguments.sglang_parse_args = lambda: None
    sglang_arguments.validate_args = lambda args: args
    monkeypatch.setitem(sys.modules, "relax.backends.sglang.arguments", sglang_arguments)

    device = ModuleType("relax.utils.device")
    device.get_dist_backend = lambda: "gloo"
    monkeypatch.setitem(sys.modules, "relax.utils.device", device)

    eval_config = ModuleType("relax.utils.training.eval_config")
    eval_config.EvalDatasetConfig = dict
    eval_config.build_eval_dataset_configs = lambda args, datasets_config, defaults: []
    eval_config.build_named_prompt_data_configs = lambda values: []
    eval_config.ensure_dataset_list = lambda values: values or []
    monkeypatch.setitem(sys.modules, "relax.utils.training.eval_config", eval_config)

    sys.modules.pop("relax.utils.arguments", None)
    module = importlib.import_module("relax.utils.arguments")
    yield module
    sys.modules.pop("relax.utils.arguments", None)


def _stub_hf_config(monkeypatch, config):
    """Stub ``relax.utils.misc.get_hf_config``; ``config`` may be an
    exception."""
    misc = ModuleType("relax.utils.misc")

    def get_hf_config(_hf_checkpoint):
        if isinstance(config, Exception):
            raise config
        return config

    misc.get_hf_config = get_hf_config
    monkeypatch.setitem(sys.modules, "relax.utils.misc", misc)


def _args(**overrides):
    base = dict(
        lora_rank=32,
        lora_scope="all",
        lora_adapter_mode=True,
        lora_merge_mode=False,
        hf_checkpoint="/models/base",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


_VL_CONFIG = SimpleNamespace(vision_config=SimpleNamespace(depth=27))
_OMNI_CONFIG = SimpleNamespace(audio_config=SimpleNamespace(depth=32))
_TEXT_CONFIG = SimpleNamespace(hidden_size=4096)


class TestRejects:
    def test_vl_model_in_adapter_mode_with_scope_all(self, arguments_module, monkeypatch):
        _stub_hf_config(monkeypatch, _VL_CONFIG)
        with pytest.raises(ValueError, match="--lora-scope language"):
            arguments_module._validate_lora_vision_scope(_args())

    def test_omni_model_is_caught_via_audio_config(self, arguments_module, monkeypatch):
        """The audio encoder is a non-language region too (VISION_REGION_TOKENS
        lists ``audio``)."""
        _stub_hf_config(monkeypatch, _OMNI_CONFIG)
        with pytest.raises(ValueError, match="vision/audio encoder"):
            arguments_module._validate_lora_vision_scope(_args())

    def test_error_offers_the_merge_mode_escape_hatch(self, arguments_module, monkeypatch):
        _stub_hf_config(monkeypatch, _VL_CONFIG)
        with pytest.raises(ValueError, match="--lora-merge-mode"):
            arguments_module._validate_lora_vision_scope(_args())


class TestAllows:
    def test_explicit_language_scope(self, arguments_module, monkeypatch):
        _stub_hf_config(monkeypatch, _VL_CONFIG)
        arguments_module._validate_lora_vision_scope(_args(lora_scope="language"))

    def test_merge_mode_may_use_any_scope(self, arguments_module, monkeypatch):
        """Merge mode folds the adapter into the synced base weights, so the
        engine never sees LoRA and the vision tower is legitimately
        trainable."""
        _stub_hf_config(monkeypatch, _VL_CONFIG)
        arguments_module._validate_lora_vision_scope(_args(lora_adapter_mode=False, lora_merge_mode=True))

    def test_text_only_model(self, arguments_module, monkeypatch):
        _stub_hf_config(monkeypatch, _TEXT_CONFIG)
        arguments_module._validate_lora_vision_scope(_args())

    def test_unreadable_config_warns_instead_of_blocking(self, arguments_module, monkeypatch):
        """A best-effort probe must never turn a valid run into a hard failure;
        the injection-time backstop in model_provider still covers this
        case."""
        _stub_hf_config(monkeypatch, OSError("no such directory"))
        warnings: list[str] = []
        # Assert on the logger rather than captured stderr: loguru binds its sink at
        # import time, so capsys only sees the output when this file runs alone.
        monkeypatch.setattr(
            arguments_module,
            "logger",
            SimpleNamespace(warning=lambda msg, *a: warnings.append(msg % a if a else msg)),
        )
        arguments_module._validate_lora_vision_scope(_args())
        assert any("--lora-scope language" in w for w in warnings)

    def test_missing_hf_checkpoint_is_not_probed(self, arguments_module, monkeypatch):
        def boom(_):
            raise AssertionError("should not probe without an hf_checkpoint")

        misc = ModuleType("relax.utils.misc")
        misc.get_hf_config = boom
        monkeypatch.setitem(sys.modules, "relax.utils.misc", misc)
        arguments_module._validate_lora_vision_scope(_args(hf_checkpoint=None))
