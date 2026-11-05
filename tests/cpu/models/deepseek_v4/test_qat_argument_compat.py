# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check real QAT guards with simulated SDK type boundaries, not kernels."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def target(monkeypatch):
    class GroupedLinear:
        def __init__(self, config):
            self.config = config
            self._fp8_workspaces = {"existing": object()}

    for name in (
        "megatron",
        "megatron.core",
        "megatron.core.extensions",
        "megatron.core.extensions.transformer_engine",
    ):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
        parent, _, leaf = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], leaf, package)
    package.TEGroupedLinear = GroupedLinear
    package.TEQuantizationParams = object
    package.TEQuantizationRecipe = object
    path = Path(__file__).resolve().parents[4] / "relax/models/deepseek_v4/qat.py"
    spec = importlib.util.spec_from_file_location("_qat_configuration_cpu_target", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setenv("OPEN_TRAINING_INT4_FAKE_QAT_FLAG", "0")
    return module, GroupedLinear


def _config(**changes):
    values = dict(fp8="hybrid", fp8_recipe="blockwise", fp8_param=False)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "changes",
    [{}, {"use_transformer_engine_op_fuser": False, "moe_single_grouped_weight": False}],
    ids=["legacy-missing", "modern-explicit-false"],
)
def test_optional_features_allow_qat_to_reach_workspace_validation(target, changes):
    module, grouped = target
    owner = grouped(_config(**changes))
    with pytest.raises(RuntimeError, match="workspace cache must be empty"):
        module.install_routed_expert_qat(owner, module_name="experts.fc1")


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("use_transformer_engine_op_fuser", "TE op fuser bypasses"),
        ("moe_single_grouped_weight", "single_grouped_weight storage"),
    ],
)
def test_enabled_optional_features_are_rejected_before_install(target, field, message):
    module, grouped = target
    owner = grouped(_config(**{field: True}))
    with pytest.raises(ValueError, match=message):
        module.install_routed_expert_qat(owner, module_name="experts.fc1")
    assert not hasattr(owner, "_relax_mxfp4_qat")
