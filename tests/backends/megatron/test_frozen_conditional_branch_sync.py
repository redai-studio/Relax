# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Frozen vision can bypass extra DP synchronization without removing
wrappers."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_SOURCE = Path(__file__).resolve().parents[3] / "relax/backends/megatron/conditional_branch_sync.py"
_SPEC = importlib.util.spec_from_file_location("conditional_branch_sync_under_test", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = torch.nn.ModuleDict({"encoder": torch.nn.Linear(3, 3), "merger": torch.nn.Linear(3, 3)})
        self.calls = 0

    def forward(self, values):
        self.calls += 1
        return values + 1


@pytest.mark.parametrize("trainable_projection", [False, True])
def test_frozen_vision_sync_checks_projection_and_preserves_forward(monkeypatch, trainable_projection):
    model = _Model()
    monkeypatch.setattr(_MODULE, "_is_qwen3vl_with_conditional_vision", lambda _: True)
    monkeypatch.setattr(_MODULE, "_rank0_log_info", lambda _: None)
    _MODULE.install_conditional_branch_sync(SimpleNamespace(overlap_param_gather=True), model)
    wrapper = model.forward
    # Freeze after installation, matching the provider/name-based freeze order.
    for parameter in model.vision_model.parameters():
        parameter.requires_grad_(False)
    if trainable_projection:
        model.vision_model["merger"].weight.requires_grad_(True)
    _MODULE.disable_frozen_conditional_branch_sync(model)
    assert model.forward == wrapper
    assert model._relax_conditional_branch_sync_active is trainable_projection
    if not trainable_projection:
        monkeypatch.setattr(_MODULE, "_first_tensor_device", lambda *_: pytest.fail("extra sync path was entered"))
        values = torch.randn(5, 3)
        torch.testing.assert_close(model(values), values + 1)
        assert model.calls == 1


def test_frozen_vision_does_not_disable_unknown_adapter():
    model = _Model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model._relax_conditional_branch_sync_installed = True
    model._relax_conditional_branch_sync_active = True
    model._relax_conditional_branch_sync_specs = ("other_branch",)
    _MODULE.disable_frozen_conditional_branch_sync(model)
    assert model._relax_conditional_branch_sync_active
