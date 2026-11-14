from types import SimpleNamespace

import pytest
import torch

from relax.components import advantages as module


@pytest.mark.parametrize("layout", ["list", "strided", "jagged"])
@pytest.mark.parametrize("use_rollout", [False, True])
def test_kl_accepts_batched_log_probs(monkeypatch, layout, use_rollout):
    values = [torch.tensor([-0.3, -0.6]), torch.tensor([-0.8, -0.2, -0.4])]
    references = [torch.tensor([-0.5, -0.7]), torch.tensor([-0.7, -0.3, -0.6])]
    if layout == "list":
        logs, refs = values, references
    else:
        kind = torch.jagged if layout == "jagged" else torch.strided
        logs = torch.nested.nested_tensor(values, layout=kind)
        refs = torch.nested.nested_tensor(references, layout=kind)
    # Exercise the service method and KL formula without starting Ray or compiling a CPU kernel.
    monkeypatch.setattr(module, "compute_approx_kl", module.compute_approx_kl._torchdynamo_orig_callable)
    monkeypatch.setattr(module, "compute_advantages_and_returns", lambda args, **data: (data["kl"], data["kl"]))
    service = SimpleNamespace(
        config=SimpleNamespace(use_rollout_logprobs=use_rollout, kl_coef=0.01, kl_loss_type="k2")
    )
    result = module.Advantages.func_or_class.compute_advantages_and_returns(
        service,
        {"rollout_log_probs" if use_rollout else "log_probs": logs, "ref_log_probs": refs},
    )
    for actual, log_prob, ref in zip(result["advantages"].unbind(), values, references, strict=True):
        torch.testing.assert_close(actual, (log_prob - ref).square() / 2)


def test_zero_kl_uses_rollout_template_without_reference(monkeypatch):
    logs = torch.nested.nested_tensor([torch.tensor([-0.3, -0.6]), torch.tensor([-0.8])], layout=torch.jagged)
    monkeypatch.setattr(module, "compute_advantages_and_returns", lambda args, **data: (data["kl"], data["kl"]))
    service = SimpleNamespace(config=SimpleNamespace(use_rollout_logprobs=False, kl_coef=0))
    result = module.Advantages.func_or_class.compute_advantages_and_returns(service, {"rollout_log_probs": logs})
    assert [value.numel() for value in result["advantages"].unbind()] == [2, 1]
    assert all(torch.count_nonzero(value) == 0 for value in result["advantages"].unbind())
