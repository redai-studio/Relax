# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check bounded TE Adam calls with the many-parameter shape of EP MoE
models."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA and Transformer Engine")
@pytest.mark.parametrize("parameter_count", [17, 5003])
def test_fused_adam_large_group_matches_small_groups(parameter_count):
    from transformer_engine.pytorch.optimizers import FusedAdam

    torch.manual_seed(1234)
    initial = torch.randn(parameter_count, 8, device="cuda", dtype=torch.bfloat16) * 0.1
    gradient = torch.randn(parameter_count, 8, device="cuda", dtype=torch.float32)
    actual = [torch.nn.Parameter(row.clone()) for row in initial]
    reference = [torch.nn.Parameter(row.clone()) for row in initial]
    options = dict(
        lr=1e-3,
        weight_decay=0.1,
        master_weights=True,
        use_decoupled_grad=True,
        store_param_remainders=True,
        exp_avg_dtype=torch.bfloat16,
        exp_avg_sq_dtype=torch.bfloat16,
    )
    optimizer = FusedAdam(actual, **options)
    small_groups = [{"params": reference[start : start + 1000]} for start in range(0, parameter_count, 1000)]
    reference_optimizer = FusedAdam(small_groups, **options)
    for step in range(3):
        for index, (parameter, expected) in enumerate(zip(actual, reference, strict=True)):
            parameter.decoupled_grad = gradient[index] * (step + 1)
            expected.decoupled_grad = parameter.decoupled_grad.clone()
        optimizer.step()
        reference_optimizer.step()
        torch.cuda.synchronize()
        torch.testing.assert_close(torch.stack(actual), torch.stack(reference), rtol=0, atol=0)
        for name in ("exp_avg", "exp_avg_sq", "master_param"):
            values = torch.stack([optimizer.state[p][name] for p in actual])
            expected = torch.stack([reference_optimizer.state[p][name] for p in reference])
            torch.testing.assert_close(values, expected, rtol=0, atol=0)
        for name in ("exp_avg", "exp_avg_sq"):
            values = torch.stack([optimizer._scales[p][name] for p in actual])
            expected = torch.stack([reference_optimizer._scales[p][name] for p in reference])
            torch.testing.assert_close(values, expected, rtol=0, atol=0)
        assert all(group["step"] == step + 1 for group in optimizer.param_groups)
        assert all(group["step"] == step + 1 for group in reference_optimizer.param_groups)
    assert not torch.equal(torch.stack(actual), initial)
