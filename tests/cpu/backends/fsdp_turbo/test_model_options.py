# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from argparse import Namespace
from types import SimpleNamespace

import pytest

from relax.backends.fsdp_turbo.model_utils import build_turbo_model
from relax.backends.fsdp_turbo.performance_config import PerformanceConfig, validate_model_options


def test_non_qwen_model_rejects_explicit_backend():
    options = PerformanceConfig.from_dict({"rmsnormgated_implementation": "triton"})
    config = SimpleNamespace(relax_performance=options)
    hf_config = SimpleNamespace(model_type="deepseek_v4", architectures=["DeepseekV4ForCausalLM"])
    with pytest.raises(ValueError, match="rmsnormgated_implementation"):
        build_turbo_model(Namespace(), config, hf_config)


def test_other_models_keep_native_and_optimizer_options():
    validate_model_options(PerformanceConfig(), is_qwen4_exp=False)
    validate_model_options(PerformanceConfig(adamw_fused=True), is_qwen4_exp=False)
