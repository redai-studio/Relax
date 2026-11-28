# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from argparse import Namespace
from types import SimpleNamespace

import pytest

from relax.backends.fsdp_turbo.model_utils import build_turbo_model
from relax.backends.fsdp_turbo.performance_config import PerformanceConfig, validate_model_options


def test_non_qwen_model_rejects_explicit_backend():
    options = PerformanceConfig.from_dict({"rmsnorm_implementation": "triton"})
    config = SimpleNamespace(relax_performance=options)
    hf_config = SimpleNamespace(model_type="deepseek_v4", architectures=["DeepseekV4ForCausalLM"])
    with pytest.raises(ValueError, match="rmsnorm_implementation"):
        build_turbo_model(Namespace(), config, hf_config)


def test_other_models_keep_native_and_optimizer_options():
    validate_model_options(PerformanceConfig(), is_qwen4_exp=False)
    validate_model_options(PerformanceConfig(adamw_fused=True), is_qwen4_exp=False)


def test_omitted_options_preserve_native_qhc_forward():
    from torch import nn

    from relax.backends.fsdp_turbo.qwen4_norm_alternatives import install

    class Qwen4ExpTextGatedResidual(nn.Module):
        def forward(self, value):
            return value + 1

    module = Qwen4ExpTextGatedResidual()
    install(module, PerformanceConfig.from_dict({}))
    assert module(3) == 4
    assert module.forward.__func__ is Qwen4ExpTextGatedResidual.forward
