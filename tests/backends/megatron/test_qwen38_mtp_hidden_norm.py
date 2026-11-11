# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_image_validator_rejects_global_mtp_norm(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "validate_unified_runtime", root / "docker/scripts/validate_unified_runtime.py"
    )
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    source = tmp_path / "norm.py"
    source.write_text(
        "class GatedResidualMTPHiddenNorm:\n"
        "    def forward(self, x):\n"
        "        return x * (x.square().mean(-1, keepdim=True) + self.eps).rsqrt()\n"
    )
    with pytest.raises(ValueError, match="independent-stream"):
        validator.require_mtp_stream_norm(source)
    source.write_text(
        "class GatedResidualMTPHiddenNorm:\n"
        "    def forward(self, x):\n"
        "        return gated_residual_group_rmsnorm(x, self.weight, self.num_streams, self.eps)\n"
    )
    validator.require_mtp_stream_norm(source)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MCore/TE import requires a CUDA runtime")
def test_qwen38_mtp_hidden_norm_matches_reference_stream_norm(dtype):
    hc = pytest.importorskip("megatron.core.transformer.hyper_connection")
    if not hasattr(hc, "GatedResidualMTPHiddenNorm"):
        pytest.skip("Requires the Qwen3.8 Megatron runtime")
    config = SimpleNamespace(hidden_size=8, params_dtype=dtype, sequence_parallel=True)
    config.num_residual_streams = config.mhc_num_residual_streams = 4
    model = hc.GatedResidualMTPHiddenNorm(config, hidden_size=8, eps=1e-6)
    reference_gain = torch.nn.Parameter(torch.zeros(32, dtype=dtype))
    gain = torch.linspace(-0.2, 0.3, 32, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(gain)
        reference_gain.copy_(gain)
    # Unequal stream scales distinguish one n*H variance from n independent H variances.
    base = torch.arange(1, 9, dtype=dtype).reshape(1, 1, 1, 8)
    scales = torch.tensor([0.1, 1.0, 4.0, 10.0], dtype=dtype).reshape(1, 1, 4, 1)
    inputs = (base * scales).expand(3, 2, 4, 8).clone().requires_grad_()
    reference_inputs = inputs.detach().clone().requires_grad_()
    actual = model(inputs)
    # Independent PyTorch RMSNorm operator over each residual stream.
    expected = (
        torch.nn.functional.rms_norm(reference_inputs.float(), (8,), eps=1e-6)
        * (1.0 + reference_gain.float()).reshape(4, 8)
    ).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == inputs.shape
    assert model.weight.sequence_parallel
    global_norm = torch.nn.functional.rms_norm(inputs.flatten(-2).float(), (32,), eps=1e-6)
    global_norm = global_norm * (1.0 + model.weight.float())
    assert not torch.allclose(actual.flatten(-2).float(), global_norm.float(), atol=0.1, rtol=0.1)
    changed = inputs.detach().clone()
    changed[:, :, 1] *= 7
    torch.testing.assert_close(model(changed)[:, :, 0], actual[:, :, 0], rtol=0, atol=0)
    probe = torch.linspace(-1, 1, inputs.numel(), dtype=torch.float32).reshape_as(inputs)
    (actual.float() * probe).sum().backward()
    (expected.float() * probe).sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
    torch.testing.assert_close(model.weight.grad, reference_gain.grad, rtol=0, atol=0)
