# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""HF 参数校验（``_hf_validate_args``）的比较范围。

这个函数运行在 Megatron 默认值派生与 Bridge provider 之前：未显式给出的字段仍是 ``None``，``--num-layers``
还允许减层训练覆盖。因此校验只应比较"此时已经有意义"的 字段，否则会拒绝原本合法的启动配置。本文件锁定该范围，避免后续重构悄悄扩大比较面。
"""

from argparse import Namespace

import pytest


pytest.importorskip("megatron.training.arguments")

from relax.backends.megatron import arguments as megatron_arguments  # noqa: E402


HF_CONFIG = Namespace(
    hidden_size=2560,
    num_attention_heads=32,
    num_hidden_layers=4,
    intermediate_size=10240,
    tie_word_embeddings=False,
    rope_theta=1_000_000,
    rms_norm_eps=1e-6,
)


def _legacy_args(**overrides):
    """固定镜像（MCore 4a4de865）的形态：只有 layernorm_epsilon。"""
    values = {
        "hidden_size": 2560,
        "num_attention_heads": 32,
        "num_layers": 2,  # 减层训练覆盖
        "ffn_hidden_size": None,  # 省略 --ffn-hidden-size，稍后由 Bridge 派生
        "untie_embeddings_and_output_weights": False,
        "rotary_base": 10000,
        "layernorm_epsilon": 1e-6,
        "moe_layer_freq": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _canonical_args(**overrides):
    """暴露 norm_epsilon 的 Megatron 版本。"""
    values = {
        "hidden_size": 2560,
        "num_attention_heads": 32,
        "num_layers": 4,
        "ffn_hidden_size": 10240,
        "untie_embeddings_and_output_weights": True,
        "rotary_base": 1_000_000,
        "norm_epsilon": 1e-6,
        "moe_layer_freq": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_legacy_epsilon_name_only_compares_epsilon():
    """未派生字段（ffn_hidden_size=None）与减层覆盖不应被判为不一致。"""
    megatron_arguments._hf_validate_args(_legacy_args(), HF_CONFIG)


def test_legacy_epsilon_mismatch_is_still_rejected():
    with pytest.raises(AssertionError, match="rms_norm_eps"):
        megatron_arguments._hf_validate_args(_legacy_args(layernorm_epsilon=1e-5), HF_CONFIG)


def test_canonical_epsilon_name_compares_the_model_table():
    """暴露 norm_epsilon 的版本仍按原语义逐项比较。"""
    megatron_arguments._hf_validate_args(_canonical_args(), HF_CONFIG)


def test_canonical_epsilon_name_reports_mismatched_field():
    with pytest.raises(AssertionError, match="hidden_size"):
        megatron_arguments._hf_validate_args(_canonical_args(hidden_size=4096), HF_CONFIG)


def test_missing_epsilon_field_is_an_explicit_error():
    """两个名字都不存在说明 Megatron 版本不受支持，必须显式失败。"""
    args = _legacy_args()
    del args.layernorm_epsilon
    with pytest.raises(AttributeError, match="neither 'norm_epsilon'"):
        megatron_arguments._hf_validate_args(args, HF_CONFIG)
