# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Megatron 参数兼容逻辑的唯一入口。

本模块只处理**上游 Megatron 字段**带来的差异，共三类：

1. **字段改名**：同一语义在 Megatron 不同版本使用不同参数名，例如
   ``norm_epsilon``（新）与 ``layernorm_epsilon``（旧）。
2. **字段缺失**：旧版本 Megatron 没有该参数，Relax 需要按版本能力决定是否使用，
   例如 ``dynamic_context_parallel``（要求 ``mpu.initialize_model_parallel``
   接受同名形参）。
3. **取值回退**：字段存在但缺省值需要按 Relax 语义推导，例如 ``rope_type``、
   FP16 optimizer 相关选项。

不适用范围（重要）：

- **Relax 自有参数**（在 ``relax/utils/arguments.py``、
  ``relax/backends/{megatron,fsdp,sglang}/arguments.py`` 中声明）永远直接读取
  ``args.xxx``，不得经由本模块补默认值，也不得为了"稳妥"在这里再写一份默认值。
  该边界由 ``tests/utils/test_no_getattr_defaults.py`` 静态守卫。
- HF 模型 config 字段探测（``text_config``、``thinker_config`` 等）属于模型结构问题，
  与命令行参数兼容无关。
- 运行时补丁（optimizer monkey patch 等）见 ``relax/backends/megatron/compat.py``。

新增条目时必须同时补齐三件事：适用条件、差异来源（Megatron 版本/字段名）、
对应测试。不要在本模块之外再新增 Megatron 版本判断。
"""

import inspect
import math
from functools import lru_cache
from typing import Any


# 字段改名表：Relax 代码使用的规范名 -> (上游旧名, 差异来源说明)
FIELD_ALIASES: dict[str, tuple[str, str]] = {
    "norm_epsilon": (
        "layernorm_epsilon",
        "Megatron-Core 将 layernorm_epsilon 更名为 norm_epsilon；旧版镜像仍只有旧名。",
    ),
}

# FP16 optimizer 在旧版 Megatron 中随 --fp16 隐式启用，新版改为独立开关。
# 这里记录旧版行为，用于字段为 None（未显式配置）时的回退。
FP16_OPTIMIZER_FALLBACKS: dict[str, Any] = {
    "initial_loss_scale": 32768.0,
    "min_loss_scale": 1.0,
    "use_precision_aware_optimizer": True,
    "store_param_remainders": False,
}

# 仅这两项的取值参与 loss scale 合法性校验（见 validate_fp16_optimizer_args）。
DYNAMIC_LOSS_SCALE_FIELDS: tuple[str, ...] = ("initial_loss_scale", "min_loss_scale")

# 布尔开关：某些 Megatron 分支（例如 NPU 侧的 MindSpeed）没有声明这些参数，
# 解析后字段根本不存在，而 Relax 会在校验与业务代码里直接读取。
# 语义：未声明 == 未启用 == False；一旦启用，仍由能力校验决定是否支持。
MEGATRON_BOOLEAN_FALLBACKS: dict[str, tuple[bool, str]] = {
    "dynamic_context_parallel": (
        False,
        "NPU 侧 Megatron 未声明该开关；未启用即等价于 False，启用时由 "
        "_validate_dynamic_context_parallel 做 npu/mpu 能力校验。",
    ),
}


def resolve_aliased_field(args: Any, name: str) -> tuple[str, Any]:
    """按 ``FIELD_ALIASES`` 解析字段改名。

    Args:
        args: 已完成 Megatron 解析的参数对象。
        name: Relax 代码使用的规范字段名。

    Returns:
        ``(实际存在的字段名, 取值)``。

    Raises:
        AttributeError: 规范名与别名都不存在。这意味着当前 Megatron 版本两个名字都
            不支持，属于必须显式暴露的环境问题，不做静默兜底。
    """
    alias, _ = FIELD_ALIASES[name]
    if hasattr(args, name):
        return name, getattr(args, name)
    if hasattr(args, alias):
        return alias, getattr(args, alias)
    raise AttributeError(f"neither {name!r} nor its Megatron alias {alias!r} exists on the parsed args")


@lru_cache(maxsize=1)
def supports_dynamic_context_parallel() -> bool:
    """当前 Megatron 版本是否支持 ``dynamic_context_parallel``。

    差异来源：Megatron-Core 在 ``mpu.initialize_model_parallel`` 中加入
    ``dynamic_context_parallel`` 形参后，Relax 才能把该开关传下去。用签名探测而不是 版本号比较，可以同时覆盖 dev
    分支与回移植版本。
    """
    from megatron.core import mpu

    return "dynamic_context_parallel" in inspect.signature(mpu.initialize_model_parallel).parameters


def fill_missing_boolean_fields(args: Any) -> None:
    """在首次校验与直读之前，为上游未声明的布尔开关补 ``False``。

    只补 **缺失** 的字段，已提供的值（包括显式 ``True``）一律保留，因此可以重复调用。 与
    ``resolve_aliased_field`` 的区别：这里处理的是"上游根本没有这个参数"，而不是改名。
    """
    for name, (default, _) in MEGATRON_BOOLEAN_FALLBACKS.items():
        if not hasattr(args, name):
            setattr(args, name, default)


def uses_canonical_norm_epsilon(args: Any) -> bool:
    """当前 Megatron 是否以规范名 ``norm_epsilon`` 暴露 RMSNorm epsilon。

    差异来源：Megatron-Core 把 ``layernorm_epsilon`` 更名为 ``norm_epsilon``；固定镜像 （MCore ``4a4de865``）只有旧名。
    HF 校验的比较范围依赖这一判定，见
    ``relax/backends/megatron/arguments.py::_hf_validate_args``。
    """
    return hasattr(args, "norm_epsilon")


def set_rope_type_default(args: Any) -> None:
    """按 Relax 语义补 ``rope_type`` 的缺省值。

    差异来源：``rope_type`` 在 Megatron 中缺省为 ``None``，需按是否 MLA 推导为 yarn /
    rope。字段本身不存在的版本不做任何写入。
    """
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"


def _optimizer_option(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def _format_optimizer_fallback(name: str, value: object) -> str:
    option = _optimizer_option(name)
    if isinstance(value, bool):
        return option if value else f"--no-{option[2:]}"
    return f"{option} {value!r}"


def validate_fp16_optimizer_args(args: Any) -> None:
    """校验 FP16 optimizer 相关字段（仅在校验阶段调用）。"""
    if getattr(args, "loss_scale", None) is None:
        for name in DYNAMIC_LOSS_SCALE_FIELDS:
            value = getattr(args, name)
            option = _optimizer_option(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{option} must be a finite number greater than 0, got {value!r}.")

        if args.min_loss_scale > args.initial_loss_scale:
            raise ValueError(
                "--min-loss-scale must be less than or equal to --initial-loss-scale, "
                f"got {args.min_loss_scale!r} > {args.initial_loss_scale!r}."
            )

    for name, fallback in FP16_OPTIMIZER_FALLBACKS.items():
        if not isinstance(fallback, bool):
            continue
        value = getattr(args, name)
        if not isinstance(value, bool):
            option = _optimizer_option(name)
            raise ValueError(f"{option} must resolve to a boolean, got {value!r}.")


def resolve_optimizer_precision_args(args: Any, logger: Any = None) -> Any:
    """在 Megatron 校验前解析 precision optimizer 参数。

    旧版 Megatron 在 ``--fp16`` 下隐式启用 precision-aware optimizer，新版需要显式 开关。字段为
    ``None`` 时按 ``FP16_OPTIMIZER_FALLBACKS``（fp16 路径）或 ``OptimizerConfig``
    原生默认值（bf16 路径）补齐，后者保证与上游一致。
    """
    from megatron.core.optimizer import OptimizerConfig

    native_defaults = None if args.fp16 else OptimizerConfig()
    missing = []
    for name, fp16_fallback in FP16_OPTIMIZER_FALLBACKS.items():
        if args.fp16 and getattr(args, "loss_scale", None) is not None and name in DYNAMIC_LOSS_SCALE_FIELDS:
            continue
        if getattr(args, name, None) is None:
            fallback = fp16_fallback if args.fp16 else getattr(native_defaults, name)
            setattr(args, name, fallback)
            if args.fp16:
                missing.append(name)

    if not args.fp16:
        return args

    validate_fp16_optimizer_args(args)
    if missing and logger is not None:
        fallback_text = ", ".join(_format_optimizer_fallback(name, getattr(args, name)) for name in missing)
        logger.warning(
            "FP16 optimizer options were not explicitly configured; using Relax compatibility fallbacks: %s. "
            "Pass the active FP16 optimizer options explicitly to silence this warning.",
            fallback_text,
        )
    return args
