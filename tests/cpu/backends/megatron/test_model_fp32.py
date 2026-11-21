# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch


def test_model_precision_declarations_survive_megatron_bf16_wrapping():
    module = pytest.importorskip("megatron.core.transformer.module")
    if not hasattr(module, "mark_keep_in_fp32"):
        pytest.skip("Megatron Core does not support FP32 tensor marking")
    from megatron.core.transformer.moe.router import TopKRouter

    from relax.backends.megatron.model_provider import configure_model_fp32

    # Use Core's bias-maintenance method without its CUDA-only constructor.
    router = TopKRouter.__new__(TopKRouter)
    torch.nn.Module.__init__(router)
    router.weight = torch.nn.Parameter(torch.zeros(2))
    router.register_buffer("expert_bias", torch.tensor([0.123456, -0.654321]))
    router.register_buffer("local_tokens_per_expert", torch.zeros(2), persistent=False)

    declared = torch.nn.Linear(2, 2, dtype=torch.bfloat16, bias=False)
    declared._keep_in_float32_parameter_names = ("weight",)
    parameter = declared.weight
    parameter.tensor_model_parallel = True
    parameter.partition_dim = 0
    model = torch.nn.Sequential(declared, router)
    config = SimpleNamespace(bf16=True, fp16=False, virtual_pipeline_model_parallel_size=None)

    expected_weight = torch.tensor([[0.123456, 0.654321], [-0.123456, -0.654321]])
    expected_bias = router.expert_bias.clone()
    configure_model_fp32(model)
    wrapped = module.Float16Module(config, model)
    torch.testing.assert_close(router.expert_bias, expected_bias, atol=0, rtol=0)
    wrapped.load_state_dict({"0.weight": expected_weight, "1.weight": torch.zeros(2), "1.expert_bias": expected_bias})

    assert declared.weight is parameter and parameter.tensor_model_parallel and parameter.partition_dim == 0
    torch.testing.assert_close(declared.weight, expected_weight, atol=0, rtol=0)
    torch.testing.assert_close(router.expert_bias, expected_bias, atol=0, rtol=0)
    assert router.weight.dtype == torch.bfloat16
    assert torch.bfloat16 == router.local_tokens_per_expert.dtype
