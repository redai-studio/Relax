# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check decoder lifetimes without loading Megatron or allocating GPU
memory."""

import ast
import gc
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def decoder_release(monkeypatch):
    import inspect

    weights = []
    live_weights_at_build = []

    class Qwen3VLTransformerBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            live_weights_at_build.append(sum(ref() is not None for ref in weights))
            self.weight = torch.nn.Parameter(torch.randn(2, 2))
            weights.append(weakref.ref(self.weight))
            # Exercise GC as well as reference counting.
            self.cycle = [self]

    class GPTModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.decoder = Qwen3VLTransformerBlock()

    class Qwen3VLGPTModel(GPTModel):
        def __init__(self, config):
            super().__init__(config)
            self.decoder = Qwen3VLTransformerBlock()

    bridge = ModuleType("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model")
    bridge.Qwen3VLGPTModel = Qwen3VLGPTModel
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    source = Path(__file__).resolve().parents[4] / "relax/backends/megatron/model_provider.py"
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_install_qwen38_decoder_release"
    )
    logger = Mock()
    namespace = {"Any": Any, "torch": torch, "inspect": inspect, "logger": logger}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    # CPU-only test: never initialize a CUDA context for diagnostics.
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    yield SimpleNamespace(
        install=namespace[function.name],
        base=GPTModel,
        model=Qwen3VLGPTModel,
        weights=weights,
        live_at_build=live_weights_at_build,
        logger=logger,
    )
    gc.collect()


def test_qwen38_decoder_release_precedes_rebuild_and_preserves_weights(decoder_release):
    case = decoder_release
    config = SimpleNamespace(ple_layer_ids=[2])
    torch.manual_seed(123)
    baseline = case.model(config)
    expected = baseline.state_dict()["decoder.weight"].clone()
    assert case.live_at_build == [0, 1]
    del baseline
    gc.collect()
    case.weights.clear()
    case.live_at_build.clear()

    case.install()
    installed = case.base.__init__
    case.install()
    assert case.base.__init__ is installed
    torch.manual_seed(123)
    model = case.model(config)

    assert case.live_at_build == [0, 0]
    assert case.weights[0]() is None
    assert case.weights[1]() is model.decoder.weight
    torch.testing.assert_close(model.state_dict()["decoder.weight"], expected, rtol=0, atol=0)
    # The replacement keeps the original parameter path for checkpoint loading.
    model.load_state_dict({"decoder.weight": torch.ones(2, 2)}, strict=True)
    torch.testing.assert_close(model.decoder.weight, torch.ones(2, 2))
    assert "old_decoder_alive=False" in case.logger.info.call_args.args[0]


def test_qwen38_decoder_release_preserves_other_gpt_models(decoder_release):
    case = decoder_release
    case.install()
    model = case.base(SimpleNamespace(ple_layer_ids=[2]))
    assert model.decoder.weight is case.weights[0]()
    case.logger.info.assert_not_called()


def test_qwen38_decoder_release_preserves_vl_without_ple(decoder_release):
    case = decoder_release
    case.install()
    model = case.model(SimpleNamespace(ple_layer_ids=[]))
    assert case.live_at_build == [0, 1]
    assert model.decoder.weight is case.weights[1]()
    case.logger.info.assert_not_called()


def test_qwen38_decoder_release_skips_changed_bridge_constructor(decoder_release):
    case = decoder_release

    def init_without_rebuild(self, config):
        case.base.__init__(self, config)

    case.model.__init__ = init_without_rebuild
    original = case.base.__init__
    case.install()
    assert case.base.__init__ is original
    case.logger.warning.assert_called_once()
    model = case.model(SimpleNamespace(ple_layer_ids=[2]))
    assert model.decoder.weight is case.weights[0]()
