# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the adapter with real FP32 AdamW; native Megatron requires GPUs."""

import json
from types import SimpleNamespace

import pytest
import torch

from relax.utils.training.optimizer_step_capture import OptimizerStepCapture


class _OptimizerContract:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.optimizer = torch.optim.AdamW([self.parameter], lr=0.01)
        self.config = SimpleNamespace()
        self.prepare_calls = 0

    def get_parameters(self):
        return [self.parameter]

    def prepare_grads(self):
        self.prepare_calls += 1
        return False

    def step(self):
        self.prepare_grads()
        torch.nn.utils.clip_grad_norm_([self.parameter], 1.0)
        self.optimizer.step()


def test_capture_reads_unclipped_grad_and_actual_fp32_adamw_delta(tmp_path):
    optimizer = _OptimizerContract()
    optimizer.parameter.grad = torch.tensor([3.0, 4.0])
    before = optimizer.parameter.detach().clone()
    capture = OptimizerStepCapture(optimizer, tmp_path / "rank-0", group=None, device=torch.device("cpu"))
    with capture:
        optimizer.step()
    capture.finish({"loss": 1.0}, True)
    assert optimizer.prepare_calls == 1
    assert "prepare_grads" not in vars(optimizer)
    payload = torch.load(tmp_path / "rank-0/parameter-000000.pt", weights_only=True)
    torch.testing.assert_close(payload["gradient"], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(payload["initial"], before, atol=0, rtol=0)
    torch.testing.assert_close(payload["update"], optimizer.parameter - before, atol=0, rtol=0)
    assert json.loads((tmp_path / "rank-0/manifest.json").read_text())["elements"] == 2


@pytest.mark.parametrize("failure", ["missing_grad", "second_prepare", "skipped_prepare", "step_error"])
def test_capture_restores_prepare_method_on_failures(tmp_path, failure):
    optimizer = _OptimizerContract()
    optimizer.parameter.grad = None if failure == "missing_grad" else torch.ones(2)
    capture = OptimizerStepCapture(optimizer, tmp_path / "capture", group=None, device=torch.device("cpu"))
    with pytest.raises(RuntimeError):
        with capture:
            if failure != "skipped_prepare":
                optimizer.prepare_grads()
            if failure == "second_prepare":
                optimizer.prepare_grads()
            if failure == "step_error":
                raise RuntimeError("optimizer failed")
    assert "prepare_grads" not in vars(optimizer)


def test_capture_rejects_casting_bf16_values_into_fake_master_weights(tmp_path):
    optimizer = _OptimizerContract()
    optimizer.parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="initialization"):
        with OptimizerStepCapture(optimizer, tmp_path / "capture", group=None, device=torch.device("cpu")):
            pytest.fail("Unsupported optimizer must fail before stepping")


def test_capture_rejects_chunked_offload_before_restoring_master_references(tmp_path):
    optimizer = _OptimizerContract()
    optimizer.config.chunked_optimizer_state_offload = True
    with pytest.raises(RuntimeError, match="initialization"):
        with OptimizerStepCapture(optimizer, tmp_path / "capture", group=None, device=torch.device("cpu")):
            pytest.fail("Chunked offload requires a separately validated capture adapter")
    assert optimizer.prepare_calls == 0
