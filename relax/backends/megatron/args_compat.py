# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Single entry point for Megatron parameter compatibility logic.

This module only handles differences caused by **upstream Megatron fields**, in
three categories:

1. **Renamed fields**: the same concept uses different parameter names across
   Megatron versions, e.g. ``norm_epsilon`` (new) vs ``layernorm_epsilon`` (old).
2. **Missing fields**: older Megatron versions lack the parameter, and Relax
   must decide whether to use it based on version capability, e.g.
   ``dynamic_context_parallel`` (requires ``mpu.initialize_model_parallel`` to
   accept a parameter of the same name).
3. **Value fallbacks**: the field exists but its default must be derived from
   Relax semantics, e.g. ``rope_type`` and the FP16 optimizer options.

Out of scope (important):

- **Relax-owned parameters** (declared in ``relax/utils/arguments.py`` and
  ``relax/backends/{megatron,fsdp,sglang}/arguments.py``) are always read
  directly via ``args.xxx``; they must never be default-filled through this
  module, and no duplicate default table may be written here "just in case".
- Probing HF model config fields (``text_config``, ``thinker_config``, etc.) is
  a model-structure concern, unrelated to CLI argument compatibility.
- Runtime patches (optimizer monkey patches, etc.) live in
  ``relax/backends/megatron/compat.py``.

Every new entry must come with three things: its applicability condition, the
source of the difference (Megatron version / field name), and a matching test.
Do not add Megatron version checks outside this module.
"""

import inspect
import math
from functools import lru_cache
from typing import Any


# Field rename table: canonical name used by Relax code -> (upstream old name, source note)
FIELD_ALIASES: dict[str, tuple[str, str]] = {
    "norm_epsilon": (
        "layernorm_epsilon",
        "Megatron-Core renamed layernorm_epsilon to norm_epsilon; older mirrors only carry the old name.",
    ),
}

# In older Megatron versions the precision-aware optimizer was enabled implicitly
# by --fp16; newer versions use a dedicated switch. The old behaviour is recorded
# here as the fallback for fields left None (not explicitly configured).
FP16_OPTIMIZER_FALLBACKS: dict[str, Any] = {
    "initial_loss_scale": 32768.0,
    "min_loss_scale": 1.0,
    "use_precision_aware_optimizer": True,
    "store_param_remainders": False,
}

# Only these two values participate in loss-scale validity checks
# (see validate_fp16_optimizer_args).
DYNAMIC_LOSS_SCALE_FIELDS: tuple[str, ...] = ("initial_loss_scale", "min_loss_scale")

# Boolean switches: some Megatron branches (e.g. the NPU-side MindSpeed build) do
# not declare these parameters, so the field does not exist at all after parsing,
# while Relax reads it directly during validation and business logic.
# Semantics: undeclared == not enabled == False; once enabled, capability checks
# still decide whether the feature is supported.
MEGATRON_BOOLEAN_FALLBACKS: dict[str, tuple[bool, str]] = {
    "dynamic_context_parallel": (
        False,
        "NPU-side Megatron does not declare this switch; absent means disabled "
        "(False), and enabling it goes through the npu/mpu capability check in "
        "_validate_dynamic_context_parallel.",
    ),
}


def resolve_aliased_field(args: Any, name: str) -> tuple[str, Any]:
    """Resolve a renamed field according to ``FIELD_ALIASES``.

    Args:
        args: parameter object that has already been through Megatron parsing.
        name: canonical field name used by Relax code.

    Returns:
        ``(name_that_exists, value)``.

    Raises:
        AttributeError: neither the canonical name nor the alias exists. This
            means the current Megatron version supports neither name, which is
            an environment problem that must surface instead of being silently
            default-filled.
    """
    alias, _ = FIELD_ALIASES[name]
    if hasattr(args, name):
        return name, getattr(args, name)
    if hasattr(args, alias):
        return alias, getattr(args, alias)
    raise AttributeError(f"neither {name!r} nor its Megatron alias {alias!r} exists on the parsed args")


@lru_cache(maxsize=1)
def supports_dynamic_context_parallel() -> bool:
    """Whether the current Megatron version supports
    ``dynamic_context_parallel``.

    Source of the difference: Relax can only pass the switch through once
    Megatron-Core added the ``dynamic_context_parallel`` parameter to
    ``mpu.initialize_model_parallel``. Probing the signature instead of
    comparing version numbers covers both dev branches and backport releases.
    """
    from megatron.core import mpu

    return "dynamic_context_parallel" in inspect.signature(mpu.initialize_model_parallel).parameters


def fill_missing_boolean_fields(args: Any) -> None:
    """Fill ``False`` for boolean switches the upstream parser never declared.

    Only **missing** fields are written; any value already present (including
    an explicit ``True``) is preserved, so the call is repeatable. Unlike
    ``resolve_aliased_field``, this handles "upstream has no such parameter at
    all" rather than a rename.
    """
    for name, (default, _) in MEGATRON_BOOLEAN_FALLBACKS.items():
        if not hasattr(args, name):
            setattr(args, name, default)


def uses_canonical_norm_epsilon(args: Any) -> bool:
    """Whether the current Megatron exposes RMSNorm epsilon as
    ``norm_epsilon``.

    Source of the difference: Megatron-Core renamed ``layernorm_epsilon`` to
    ``norm_epsilon``; the pinned mirror (MCore ``4a4de865``) only has the old
    name. The comparison scope of the HF validation depends on this predicate,
    see ``relax/backends/megatron/arguments.py::_hf_validate_args``.
    """
    return hasattr(args, "norm_epsilon")


def set_rope_type_default(args: Any) -> None:
    """Fill the ``rope_type`` default according to Relax semantics.

    Source of the difference: ``rope_type`` defaults to ``None`` in Megatron
    and must be derived to yarn / rope based on whether MLA is used. Versions
    where the field itself does not exist are left untouched.
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
    """Validate the FP16 optimizer fields (called during validation only)."""
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
    """Resolve precision optimizer parameters before Megatron validation.

    Older Megatron versions enabled the precision-aware optimizer implicitly
    under ``--fp16``; newer versions need the explicit switch. Fields left
    ``None`` are filled from ``FP16_OPTIMIZER_FALLBACKS`` (fp16 path) or from
    the native ``OptimizerConfig`` defaults (bf16 path), the latter keeping
    parity with upstream.
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
