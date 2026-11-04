"""Torch SDPA parity must not silently depend on the optional FA2 package."""

import os

import torch

from relax.models.sensenova.adapter import configure_attention


def test_torch_sdpa_deterministic_policy_is_supported(monkeypatch):
    before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    monkeypatch.setenv("RELAX_ATTENTION_IMPLEMENTATION", "torch")
    monkeypatch.setenv("RELAX_DETERMINISTIC_ATTENTION", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        configure_attention("torch", deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert os.environ["RELAX_ATTENTION_IMPLEMENTATION"] == "torch"
        assert os.environ["RELAX_DETERMINISTIC_ATTENTION"] == "1"
    finally:
        torch.use_deterministic_algorithms(before[0])
        torch.backends.cudnn.deterministic = before[1]
        torch.backends.cudnn.benchmark = before[2]
