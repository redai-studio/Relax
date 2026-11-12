# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise model routing without constructing a Ray/Megatron actor."""

import importlib.util
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from relax.utils.misc import resolve_is_vl_model


@pytest.mark.parametrize("config_name", ["processor_config.json", "preprocessor_config.json"])
@pytest.mark.parametrize("processor", [False, True])
@pytest.mark.parametrize("declared", [None, False, True])
def test_provider_declaration_overrides_checkpoint_metadata(tmp_path, config_name, processor, declared):
    if processor:
        (tmp_path / config_name).write_text("{}")
    provider = SimpleNamespace(relax_is_multimodal=declared)
    assert resolve_is_vl_model(provider, str(tmp_path)) is (processor if declared is None else declared)


@pytest.mark.parametrize("provider", [None, SimpleNamespace()])
def test_unknown_provider_without_checkpoint_is_text_only(provider):
    assert resolve_is_vl_model(provider, None) is False


def test_non_boolean_provider_declaration_fails_loudly(tmp_path):
    with pytest.raises(TypeError, match="must be bool or None"):
        resolve_is_vl_model(SimpleNamespace(relax_is_multimodal="false"), str(tmp_path))


@pytest.mark.parametrize("variant,expected", [("dense", False), ("dense_vl", True), ("moe", False), ("moe_vl", True)])
def test_gemma4_provider_declarations_preserve_in_place_conversion(monkeypatch, tmp_path, variant, expected):
    # The CI image need not ship Gemma4 Bridge classes. Load the production
    # provider subclasses against small dataclass bases to exercise the actual
    # declarations and Bridge's in-place class replacement without a model.
    @dataclass
    class BaseProvider:
        rotary_base: int = 10000

    text = ModuleType("megatron.bridge.models.gemma.gemma4_provider")
    vl = ModuleType("megatron.bridge.models.gemma_vl.gemma4_vl_provider")
    text.Gemma4DenseProvider = type("Gemma4DenseProvider", (BaseProvider,), {})
    text.Gemma4ModelProvider = type("Gemma4ModelProvider", (BaseProvider,), {})
    vl.Gemma4DenseVLProvider = type("Gemma4DenseVLProvider", (BaseProvider,), {})
    vl.Gemma4VLModelProvider = type("Gemma4VLModelProvider", (BaseProvider,), {})
    monkeypatch.setitem(sys.modules, text.__name__, text)
    monkeypatch.setitem(sys.modules, vl.__name__, vl)
    path = Path(__file__).parents[4] / "relax/models/gemma4/gemma4_provider.py"
    spec = importlib.util.spec_from_file_location("_gemma4_routing_provider", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    upstream = {
        "dense": text.Gemma4DenseProvider,
        "dense_vl": vl.Gemma4DenseVLProvider,
        "moe": text.Gemma4ModelProvider,
        "moe_vl": vl.Gemma4VLModelProvider,
    }[variant]
    provider = upstream(rotary_base=123)
    provider.extra_bridge_metadata = object()
    original_state = vars(provider).copy()
    provider.__class__ = module.RELAX_PROVIDERS[upstream]
    assert vars(provider) == original_state
    assert [field.name for field in fields(provider)] == ["rotary_base"]
    (tmp_path / "processor_config.json").write_text("{}")
    assert resolve_is_vl_model(provider, str(tmp_path)) is expected
