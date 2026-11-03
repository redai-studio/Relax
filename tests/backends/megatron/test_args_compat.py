# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Megatron 兼容层（args_compat）的单元测试。

这些用例只覆盖"字段改名 / 取值回退 / 能力探测"三类兼容语义，不依赖 GPU； 需要真实 Megatron 的用例用 ``importorskip``
优雅跳过，保持 CPU CI 可跑。
"""

import sys
from argparse import Namespace

import pytest

from relax.backends.megatron import args_compat


def test_resolve_aliased_field_prefers_canonical_name():
    name, value = args_compat.resolve_aliased_field(Namespace(norm_epsilon=1e-6), "norm_epsilon")
    assert (name, value) == ("norm_epsilon", 1e-6)


def test_resolve_aliased_field_falls_back_to_legacy_name():
    name, value = args_compat.resolve_aliased_field(Namespace(layernorm_epsilon=1e-5), "norm_epsilon")
    assert (name, value) == ("layernorm_epsilon", 1e-5)


def test_resolve_aliased_field_reports_missing_field():
    """两个名字都不存在说明 Megatron 版本不受支持，必须显式失败。"""
    with pytest.raises(AttributeError, match="neither 'norm_epsilon'"):
        args_compat.resolve_aliased_field(Namespace(), "norm_epsilon")


def test_rope_type_defaults_to_rope_for_dense_attention():
    args = Namespace(rope_type=None, multi_latent_attention=False)
    args_compat.set_rope_type_default(args)
    assert args.rope_type == "rope"


def test_rope_type_defaults_to_yarn_for_mla():
    args = Namespace(rope_type=None, multi_latent_attention=True)
    args_compat.set_rope_type_default(args)
    assert args.rope_type == "yarn"


def test_rope_type_explicit_value_is_preserved():
    args = Namespace(rope_type="rope", multi_latent_attention=True)
    args_compat.set_rope_type_default(args)
    assert args.rope_type == "rope"


def test_rope_type_absent_field_is_left_alone():
    args = Namespace(multi_latent_attention=False)
    args_compat.set_rope_type_default(args)
    assert not hasattr(args, "rope_type")


def _fp16_args(**overrides):
    values = {
        "fp16": True,
        "loss_scale": None,
        "initial_loss_scale": 32768.0,
        "min_loss_scale": 1.0,
        "use_precision_aware_optimizer": True,
        "store_param_remainders": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_validate_fp16_optimizer_accepts_valid_values():
    args_compat.validate_fp16_optimizer_args(_fp16_args())


@pytest.mark.parametrize("field", ["initial_loss_scale", "min_loss_scale"])
@pytest.mark.parametrize("value", [0, -1.0, float("nan"), float("inf"), True])
def test_validate_fp16_optimizer_rejects_non_positive_loss_scale(field, value):
    with pytest.raises(ValueError, match="must be a finite number greater than 0"):
        args_compat.validate_fp16_optimizer_args(_fp16_args(**{field: value}))


def test_validate_fp16_optimizer_rejects_inverted_loss_scale():
    with pytest.raises(ValueError, match="--min-loss-scale must be less than or equal"):
        args_compat.validate_fp16_optimizer_args(_fp16_args(initial_loss_scale=1.0, min_loss_scale=2.0))


def test_validate_fp16_optimizer_rejects_non_boolean_flag():
    with pytest.raises(ValueError, match="must resolve to a boolean"):
        args_compat.validate_fp16_optimizer_args(_fp16_args(store_param_remainders=1))


def test_alias_table_documents_its_source():
    """新增兼容项必须写明差异来源，避免变成无出处的兜底表。"""
    for canonical, (alias, note) in args_compat.FIELD_ALIASES.items():
        assert canonical and alias and note


def test_supports_dynamic_context_parallel_requires_megatron():
    pytest.importorskip("megatron.core")
    assert isinstance(args_compat.supports_dynamic_context_parallel(), bool)


def test_fill_missing_boolean_fields_adds_absent_switch():
    """NPU 侧 Megatron 没有声明该开关：补 False，而不是让直读抛 AttributeError。"""
    args = Namespace()

    args_compat.fill_missing_boolean_fields(args)

    assert args.dynamic_context_parallel is False


def test_fill_missing_boolean_fields_preserves_provided_value():
    args = Namespace(dynamic_context_parallel=True)

    args_compat.fill_missing_boolean_fields(args)

    assert args.dynamic_context_parallel is True


def test_fill_missing_boolean_fields_is_idempotent():
    args = Namespace()
    args_compat.fill_missing_boolean_fields(args)
    args_compat.fill_missing_boolean_fields(args)
    assert args.dynamic_context_parallel is False


def test_boolean_fallback_table_documents_its_reason():
    for name, (default, reason) in args_compat.MEGATRON_BOOLEAN_FALLBACKS.items():
        assert isinstance(default, bool), name
        assert reason, name


def test_uses_canonical_norm_epsilon_detects_modern_name():
    assert args_compat.uses_canonical_norm_epsilon(Namespace(norm_epsilon=1e-6)) is True


def test_uses_canonical_norm_epsilon_detects_legacy_name():
    assert args_compat.uses_canonical_norm_epsilon(Namespace(layernorm_epsilon=1e-5)) is False


def test_megatron_test_defaults_match_upstream(monkeypatch):
    """测试侧镜像的 Megatron 字段必须与上游默认值一致。"""
    pytest.importorskip("megatron.training.arguments")

    from megatron.training.arguments import parse_args as megatron_parse_args

    from tests.support.megatron_defaults import MEGATRON_TEST_DEFAULTS

    # 上游 parse_args 只接受 (extra_args_provider, ignore_unknown_args)，且会读取
    # sys.argv；这里显式隔离，避免 pytest 的命令行被当成 Megatron 参数。
    monkeypatch.setattr(sys, "argv", ["megatron"])
    upstream = megatron_parse_args(extra_args_provider=None)
    for name, value in MEGATRON_TEST_DEFAULTS.items():
        if not hasattr(upstream, name):
            # 该字段在当前 Megatron 版本由其他解析路径提供，镜像值仅用于构造测试对象。
            continue
        assert getattr(upstream, name) == value, name
