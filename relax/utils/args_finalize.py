# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Single finalization stage applied after argument parsing.

Background: Relax arguments come from three independent parsers (megatron/slime,
sglang server, sglang teacher), merged segment by segment into one
``Namespace`` by ``_parse_args_impl``. The merge is **conditional**
(``--debug-train-only``, ``--load-debug-rollout-data`` and i2i SFT skip the
sglang parsing), so downstream code has long been filling gaps with
``getattr(args, "xxx", default)``: defaults get maintained twice, and "the
field was never generated" gets masked as "the default was used".

This module consolidates that work into the **single finalization stage**
between parsing and validation. It does exactly two things:

1. ``complete_optional_subparser_defaults``: fill the declared defaults of
   sub-parsers that a given launch path skipped. The values still have a
   single source — the sub-parser's own declarations; no constants are added
   here.
2. ``seed_runtime_derived``: give initial values to derived attributes that
   cannot be determined at parse time and are written at runtime, so readers
   can use ``args.xxx`` directly instead of patching over missing fields.

Boundary: this module introduces no new default semantics. Defaults of
Relax-owned parameters are still declared only at their parameter definitions;
Megatron field differences live in ``relax/backends/megatron/args_compat.py``.
"""

import argparse
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Derived attributes that cannot be determined at parse time and are written by
# runtime components. The second value of each entry documents "who writes it at
# which stage", so constants are not mistaken for parameters.
RUNTIME_DERIVED_DEFAULTS: dict[str, tuple[Any, str]] = {
    "is_vl_model": (
        False,
        "overwritten by MegatronTrainRayActor according to processor/preprocessor capability after loading the HF config (backends/megatron/actor.py)",
    ),
    "uses_unsplit_forward": (
        False,
        "set True by the model provider once the bridge-produced model is confirmed to need an unsplit forward (backends/megatron/model_provider.py)",
    ),
    "sft_image_preprocess_on_rank": (
        False,
        "SFT image preprocessing switch; currently no writer, always False, read sites use direct args access",
    ),
    "use_critic": (
        False,
        "overwritten by validate_algorithm_args based on the algorithm spec's needs_critic; seeded to False at parse time",
    ),
}


def _sglang_default_attributes() -> dict[str, Any]:
    """Take the declared defaults out of the sglang sub-parser.

    sglang is an optional dependency (CPU CI and some deployments do not
    install it); when it is missing this returns an empty dict: those fields
    would not be used anyway, and skipping silently is preferable to failing
    the whole parse.
    """
    try:
        from relax.backends.sglang import arguments as sglang_arguments
    except ImportError as exc:  # pragma: no cover - optional dependency missing
        logger.debug("sglang argument definitions unavailable, skip namespace completion: %s", exc)
        return {}

    # A test double may only provide the parse entry without the declaration
    # API; without declarations there is nothing to fill in.
    add_sglang_arguments = getattr(sglang_arguments, "add_sglang_arguments", None)
    if add_sglang_arguments is None:
        return {}

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_sglang_arguments(parser)
    return dict(vars(parser.parse_args([])))


def complete_optional_subparser_defaults(args: Any) -> None:
    """Fill fields of skipped sub-parsers so the namespace is launch-path
    independent.

    Only **missing** fields are filled and existing values are never
    overwritten, so paths that already parsed sglang are completely unaffected.
    """
    existing = vars(args)
    for name, value in _sglang_default_attributes().items():
        if name not in existing:
            setattr(args, name, value)


def seed_runtime_derived(args: Any) -> None:
    """Seed initial values for runtime-derived attributes; only missing ones
    are written."""
    existing = vars(args)
    for name, (default, _) in RUNTIME_DERIVED_DEFAULTS.items():
        if name not in existing:
            setattr(args, name, default)


def finalize_args(args: Any) -> Any:
    """Unified finalization after parsing. Idempotent and safe to call
    repeatedly.

    Args:
        args: the fully merged parameter object.

    Returns:
        The same parameter object, for convenient chaining.
    """
    complete_optional_subparser_defaults(args)
    seed_runtime_derived(args)
    return args
