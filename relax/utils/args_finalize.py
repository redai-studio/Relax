# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""参数解析后的统一收尾阶段。

问题背景：Relax 的参数来自三个独立 parser（megatron/slime、sglang server、
sglang teacher），由 ``_parse_args_impl`` 逐段合并进同一个 ``Namespace``。合并是
**有条件**的（``--debug-train-only``、``--load-debug-rollout-data``、i2i SFT 会跳过
sglang 解析），于是下游代码长期靠 ``getattr(args, "xxx", 默认值)`` 补齐：
既重复维护了默认值，也把"字段根本没生成"掩盖成了"用了默认值"。

本模块把这部分工作收敛到解析之后、校验之前的**唯一收尾阶段**，只做两件事：

1. ``complete_optional_subparser_defaults``：把被跳过的子 parser 的声明默认值补进
   命名空间。取值仍然只有一个来源——子 parser 的声明，不在这里另写常量。
2. ``seed_runtime_derived``：为"解析期无法确定、运行期才被写入"的派生属性给出
   初值，使读侧可以直接 ``args.xxx``，不必再逐点兜底。

边界：本模块不产生任何新的默认值语义。Relax 自有参数的默认值仍然只在参数定义处
声明；Megatron 字段差异见 ``relax/backends/megatron/args_compat.py``。
"""

import argparse
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# 解析期无法确定、由运行期组件写入的派生属性。
# 每一项的第二个值说明"谁在什么阶段写入"，避免把常量误当参数。
RUNTIME_DERIVED_DEFAULTS: dict[str, tuple[Any, str]] = {
    "is_vl_model": (
        False,
        "MegatronTrainRayActor 加载 HF config 后按 processor/preprocessor 能力覆写（backends/megatron/actor.py）",
    ),
    "uses_unsplit_forward": (
        False,
        "model provider 在确认 bridge 产出的模型需要未切分前向时置 True（backends/megatron/model_provider.py）",
    ),
    "sft_image_preprocess_on_rank": (
        False,
        "SFT 图像预处理开关；当前无写入方，恒为 False，读取点统一走 args 直接访问",
    ),
    "use_critic": (
        False,
        "由 validate_algorithm_args 按算法 spec 的 needs_critic 覆写；解析期先给 False 初值",
    ),
}


def _sglang_default_attributes() -> dict[str, Any]:
    """从 sglang 子 parser 的声明里取出默认值。

    sglang 是可选依赖（CPU CI 与部分部署不安装），依赖缺失时返回空字典：这些字段 本来也不会被使用，静默跳过比让整个解析失败更合适。
    """
    try:
        from relax.backends.sglang import arguments as sglang_arguments
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        logger.debug("sglang argument definitions unavailable, skip namespace completion: %s", exc)
        return {}

    # 测试用替身可能只提供解析入口而不提供声明 API；没有声明就没有可补的默认值。
    add_sglang_arguments = getattr(sglang_arguments, "add_sglang_arguments", None)
    if add_sglang_arguments is None:
        return {}

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_sglang_arguments(parser)
    return dict(vars(parser.parse_args([])))


def complete_optional_subparser_defaults(args: Any) -> None:
    """补齐被跳过的子 parser 字段，保证命名空间与执行路径无关。

    只填 **缺失** 的字段，已存在的值一律不覆盖，因此对"已经解析过 sglang"的路径 完全无副作用。
    """
    existing = vars(args)
    for name, value in _sglang_default_attributes().items():
        if name not in existing:
            setattr(args, name, value)


def seed_runtime_derived(args: Any) -> None:
    """为运行期派生属性给出初值，缺失才写入。"""
    existing = vars(args)
    for name, (default, _) in RUNTIME_DERIVED_DEFAULTS.items():
        if name not in existing:
            setattr(args, name, default)


def finalize_args(args: Any) -> Any:
    """解析完成后的统一收尾。幂等，可重复调用。

    Args:
        args: 合并完成的参数对象。

    Returns:
        同一个参数对象，便于链式调用。
    """
    complete_optional_subparser_defaults(args)
    seed_runtime_derived(args)
    return args
