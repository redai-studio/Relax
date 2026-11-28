# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from dataclasses import replace

import pytest

from relax.backends.fsdp_turbo.performance_config import PerformanceConfig, validate_model_options


def test_non_qwen_model_rejects_explicit_backend():
    defaults = PerformanceConfig()
    selected = "vision_load_balance"
    options = replace(defaults, vision_load_balance=True)
    with pytest.raises(ValueError, match=selected):
        validate_model_options(options, is_qwen4_exp=False)
    validate_model_options(options, is_qwen4_exp=True)


def test_other_models_keep_native_and_optimizer_options():
    validate_model_options(PerformanceConfig(), is_qwen4_exp=False)
    validate_model_options(PerformanceConfig(adamw_fused=True), is_qwen4_exp=False)
