# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from relax.backends.fsdp_turbo.performance_ops.token_chunks import _moe
from relax.backends.fsdp_turbo.qwen4_experts import configure


@pytest.mark.parametrize("rank", [0, 1])
def test_chunked_moe_preserves_router_record_and_gradients(rank):
    torch.manual_seed(42)
    shared = nn.Linear(3, 3)
    shared_gate = nn.Linear(3, 1)
    router = nn.Linear(3, 2)
    records = []

    def gate(x):
        logits = router(x)
        records.append(logits)
        weights, indices = logits.softmax(-1).topk(1)
        return logits, weights, indices

    def experts(x, indices, weights):
        return x * weights * (indices + 1)

    x = torch.randn(2, 5, 3, requires_grad=True)
    logits = router(x.reshape(-1, 3))
    weights, indices = logits.softmax(-1).topk(1)
    expected = experts(x.reshape(-1, 3), indices, weights).reshape_as(x)
    expected = expected + shared(x) * shared_gate(x).sigmoid()
    model = SimpleNamespace(
        gate=gate,
        experts=experts,
        shared_expert=shared,
        shared_expert_gate=shared_gate,
        _relax_token_chunk_size=4,
        _relax_token_cp_rank=rank,
        _relax_token_cp_size=2,
    )
    actual = _moe(model, x)
    torch.testing.assert_close(actual, expected)
    assert len(records) == 1
    torch.testing.assert_close(records[0], logits)
    parameters = (x, *shared.parameters(), *shared_gate.parameters(), *router.parameters())
    actual_grads = torch.autograd.grad(actual.sum(), parameters, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_megamoe_rejects_ep1_before_loading_extension():
    config = SimpleNamespace(
        relax_performance=SimpleNamespace(moe_implementation="megamoe"),
        distributed=SimpleNamespace(expert_parallel_size=1),
    )
    with pytest.raises(ValueError, match="expert_parallel_size"):
        configure(config, sft=True)
