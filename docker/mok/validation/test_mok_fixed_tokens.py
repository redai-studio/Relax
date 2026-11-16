"""CPU contracts for the adapter's padding around the CUDA-only MoK call."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.moe.megakernel.mok import backend


def make_adapter(target):
    module = backend.MoKMegakernel.__new__(backend.MoKMegakernel)
    torch.nn.Module.__init__(module)
    module.fixed_num_local_tokens = target
    module.topk = 2
    module.native_single_grouped_weights = False
    module._routed_fc1_parameter_names = []
    module._routed_fc2_parameter_names = []
    module.shared_output_gate_weight = None
    module.shared_fc1_weight = None
    module.shared_fc2_weight = None
    return module


@pytest.mark.parametrize("num_tokens", [3, 257, 512])
def test_padding_preserves_outputs_and_input_router_gradients(monkeypatch, num_tokens):
    module = make_adapter(512)
    x = torch.linspace(-1, 1, num_tokens * 4).reshape(num_tokens, 1, 4).requires_grad_()
    weights = torch.linspace(0.1, 0.9, num_tokens * 2).reshape(num_tokens, 2).requires_grad_()
    routes = torch.ones(num_tokens, 2, dtype=torch.bool)
    expert_ids = torch.zeros(num_tokens, 2, dtype=torch.int32)
    monkeypatch.setattr(backend, "routing_map_to_mok_inputs", lambda *_: (weights, expert_ids))
    captured = {}

    def run_mok(_module, padded_x, padded_weights, padded_experts, *_):
        assert padded_x.shape == (512, 4)
        assert padded_weights.shape == padded_experts.shape == (512, 2)
        assert torch.count_nonzero(padded_x[num_tokens:]) == 0
        assert torch.count_nonzero(padded_weights[num_tokens:]) == 0
        assert torch.all(padded_experts[num_tokens:] == -1)
        output = padded_x.square() * padded_weights.sum(-1, keepdim=True)
        output.retain_grad()
        captured["output"] = output
        return output

    monkeypatch.setattr(backend, "_MoKAutograd", SimpleNamespace(apply=run_mok))
    result = module(x, weights, routes)
    expected = x.square() * weights.sum(-1).view(num_tokens, 1, 1)
    torch.testing.assert_close(result, expected)
    upstream = torch.linspace(0.2, 1.2, result.numel()).reshape_as(result)
    result.backward(upstream)
    torch.testing.assert_close(x.grad, 2 * x.detach() * weights.detach().sum(-1).view(num_tokens, 1, 1) * upstream)
    expected_router_grad = (x.detach().square() * upstream).sum((1, 2))[:, None].expand(-1, 2)
    torch.testing.assert_close(weights.grad, expected_router_grad)
    assert torch.count_nonzero(captured["output"].grad[num_tokens:]) == 0


def test_oversize_microbatch_fails_before_routing(monkeypatch):
    module = make_adapter(512)
    monkeypatch.setattr(backend, "routing_map_to_mok_inputs", lambda *_: pytest.fail("routing must not run"))
    with pytest.raises(ValueError, match="513 exceeds fixed_num_local_tokens=512"):
        module(torch.zeros(513, 4), torch.zeros(513, 2), torch.zeros(513, 2, dtype=torch.bool))
