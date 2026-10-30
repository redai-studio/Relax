# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""``dynamic_context_parallel`` 缺失时的启动路径。

NPU 侧的 Megatron 分支没有声明 ``--dynamic-context-parallel``，解析出来的 namespace
里根本没有这个属性，而 Relax 的校验与业务代码会直接读取它。 本文件锁定兼容层的补全语义：只补缺失字段、已提供的值不动、启用时仍走能力校验。
"""

from argparse import Namespace

import pytest


pytest.importorskip("megatron.training.arguments")

from relax.backends.megatron import arguments as megatron_arguments  # noqa: E402


def _npu_shaped_args() -> Namespace:
    """复刻 NPU parser 的输出：没有 dynamic_context_parallel，其余字段齐备。"""
    return Namespace(
        context_parallel_size=1,
        use_dynamic_batch_size=False,
        variable_seq_lengths=None,
        moe_token_dispatcher_type="alltoall",
        pipeline_model_parallel_size=1,
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
    )


@pytest.fixture()
def without_upstream_validation(monkeypatch):
    """跳过上游 validate_args 与设备探测，只保留 Relax 自己的分支逻辑。"""
    monkeypatch.setattr(megatron_arguments, "_megatron_validate_args", lambda args: None)
    monkeypatch.setattr(megatron_arguments.device_utils, "is_available", lambda: True)


def test_validate_args_fills_missing_dynamic_context_parallel(without_upstream_validation):
    """回归：字段缺失时原先会在 `args.dynamic_context_parallel` 抛 AttributeError。"""
    args = _npu_shaped_args()
    assert not hasattr(args, "dynamic_context_parallel")

    megatron_arguments.validate_args(args)

    assert args.dynamic_context_parallel is False


def test_validate_args_keeps_explicit_value(without_upstream_validation, monkeypatch):
    """已显式提供的值不被覆盖；启用时仍要求当前 Megatron 支持该能力。"""
    monkeypatch.setattr(
        megatron_arguments,
        "supports_dynamic_context_parallel",
        lambda: False,
    )
    args = _npu_shaped_args()
    args.dynamic_context_parallel = True

    with pytest.raises(AssertionError, match="dynamic_context_parallel is not supported"):
        megatron_arguments.validate_args(args)
