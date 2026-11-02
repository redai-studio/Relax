# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


# Load this standalone wrapper without importing the Megatron backend's GPU patches.
_spec = importlib.util.spec_from_file_location(
    "conditional_branch_sync_under_test",
    Path(__file__).resolve().parents[3] / "relax/backends/megatron/conditional_branch_sync.py",
)
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("local_images", [False, True])
def test_mixed_vision_forward_preserves_output_and_backward(monkeypatch, paired, local_images):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_weight = torch.nn.Parameter(torch.tensor(2.0))
            self.pg_collection = SimpleNamespace(dp=object())
            self.mask = torch.tensor([[1.0, 0.0]])

        def forward(self, input_ids, **kwargs):
            output = input_ids * (self.vision_weight if local_images else 3)
            return (output, self.mask) if paired else output

    model = Model()
    monkeypatch.setattr(sync, "_is_qwen3vl_with_conditional_vision", lambda _: True)
    monkeypatch.setattr(sync, "_dp_any", lambda *args: True)
    dummy_calls = []

    def dummy(model, device):
        dummy_calls.append(device)
        return model.vision_weight.square()

    monkeypatch.setattr(sync, "_run_dummy_vision", dummy)
    monkeypatch.setenv("RELAX_DISABLE_VISION_BRANCH_SYNC", "0")
    sync.install_conditional_branch_sync(Namespace(overlap_grad_reduce=True), model)
    inputs = torch.tensor([[1.0, 2.0]], requires_grad=True)
    kwargs = {"image_grid_thw": torch.ones(1, 3)} if local_images else {}
    result = model(inputs, **kwargs)
    if paired:
        assert isinstance(result, tuple) and len(result) == 2
        output, mask = result
        assert mask is model.mask
        assert not mask.requires_grad
    else:
        output = result
    torch.testing.assert_close(output, inputs * (2 if local_images else 3))
    output.sum().backward()
    torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 2 if local_images else 3))
    assert model.vision_weight.grad is not None
    torch.testing.assert_close(model.vision_weight.grad, torch.tensor(3.0 if local_images else 0.0))
    assert len(dummy_calls) == (0 if local_images else 1)


def test_conditional_sync_rejects_unknown_output_contract():
    with pytest.raises(TypeError, match="tensor pair"):
        sync._attach_zero_dependency((torch.ones(1), None), torch.ones(1))


def test_text_only_opt_out_does_not_wrap_forward(monkeypatch):
    model = torch.nn.Linear(2, 2)
    original = model.forward
    monkeypatch.setenv("RELAX_DISABLE_VISION_BRANCH_SYNC", "1")
    monkeypatch.setattr(sync, "_is_qwen3vl_with_conditional_vision", lambda _: True)
    monkeypatch.setattr(sync, "_rank0_log_info", lambda _: None)
    sync.install_conditional_branch_sync(Namespace(overlap_grad_reduce=True), model)
    assert model.forward == original
