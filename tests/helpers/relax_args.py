# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Declaration-driven ``argparse.Namespace`` factory for tests.

Production code parses arguments through :func:`relax.utils.arguments.parse_args`,
which extends the Megatron parser with Relax's own arguments and then runs
validations that require a concrete training configuration (batch sizes,
checkpoint paths, ...).  Tests historically rebuilt a hand-maintained subset of
the parsed namespace, which silently drops newly declared defaults and derived
fields.

This module collects the parser defaults from the **real** parser registration
(no hand-maintained default table):

* the Megatron parser is constructed with Relax's extra-args provider and parses
  an empty command line with unknown arguments ignored, so every registered
  ``type=`` converter runs exactly once;
* the flow-control arguments produced by the production pre-parser are merged
  the same way production merges them.

Post-parse validation deliberately does not run.  The small set of derived
fields that are deterministic without a model, a dataset or a GPU are applied
by :func:`apply_safe_derivations`, using the same assignments production
makes (including forcing ``use_critic`` off for offline objectives). Fields
produced by worker/controller initialization (``is_vl_model``,
``num_rollout_per_epoch``, ...) stay absent and must be supplied explicitly by
the tests that exercise them.
"""

from __future__ import annotations

import argparse
import copy
import sys
from typing import Any


# Flow-control keys the production pre-parser owns and copies into the main
# namespace after Phase 2 parsing (``loss_type``/``generation_task`` stay with
# the main parser); keep this list identical to _parse_args_impl.
_PRE_PARSER_KEYS = (
    "train_backend",
    "debug_rollout_only",
    "debug_train_only",
    "load_debug_rollout_data",
    "skip_hf_validate",
)

_DEFAULTS: argparse.Namespace | None = None

# Fields apply_safe_derivations produces. Input overrides (e.g.
# ``advantage_estimator``) are applied first so a derivation reads the test's
# value; explicitly overriding one of the derived fields itself wins afterwards.
_DERIVED_FIELDS = ("model_source", "_start_rollout_id_explicit", "lora_target_modules", "use_critic")


def _collect_defaults() -> argparse.Namespace:
    """Parse an empty command line through the real parser registration
    once."""
    global _DEFAULTS
    if _DEFAULTS is not None:
        return _DEFAULTS

    saved_argv = sys.argv
    sys.argv = ["relax-tests"]
    try:
        from megatron.training.arguments import parse_args as megatron_parse_args

        from relax.utils.arguments import _pre_parse_mode, get_slime_extra_args_provider

        args = megatron_parse_args(
            extra_args_provider=get_slime_extra_args_provider(),
            ignore_unknown_args=True,
        )
        pre = _pre_parse_mode()
        for key in _PRE_PARSER_KEYS:
            setattr(args, key, getattr(pre, key))
    finally:
        sys.argv = saved_argv

    _DEFAULTS = args
    return _DEFAULTS


def apply_safe_derivations(args: argparse.Namespace) -> argparse.Namespace:
    """Apply the post-parse assignments that are deterministic without I/O.

    Mirrors the corresponding lines in
    ``relax.utils.arguments._parse_args_impl`` and ``slime_validate_args``.
    Data- and model-dependent derivations are intentionally excluded; callers
    set them per test.
    """
    # Driver-only descriptor; worker namespaces serialize without it and tests
    # that exercise model-source resolution provide their own ModelSource.
    args.model_source = None

    args._start_rollout_id_explicit = args.start_rollout_id is not None

    # Same Megatron fallback production applies before validation; fsdp keeps
    # None so the generative adapter can supply diffusers module names.
    if getattr(args, "lora_target_modules", None) is None and args.train_backend != "fsdp":
        args.lora_target_modules = ["linear_qkv", "linear_proj"]

    from relax.algorithms import get_algorithm
    from relax.engine.sft.runtime import is_offline_mode

    # Mirror production ordering: validate_algorithm_args first assigns the
    # algorithm spec's needs_critic, then slime_validate_args forces no critic
    # for offline objectives (SFT/DPO/RM spawn no critic role).
    args.use_critic = get_algorithm(args.advantage_estimator).needs_critic and not is_offline_mode(args)
    return args


def make_raw_args(**overrides: Any) -> argparse.Namespace:
    """A fresh namespace of parser defaults without post-parse derivations.

    Use this for tests of parsing/initialization stages that run before derived
    fields exist: a missing field then fails loudly instead of being silently
    defaulted.
    """
    args = copy.deepcopy(_collect_defaults())
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_args(**overrides: Any) -> argparse.Namespace:
    """A fresh parsed namespace with deterministic derivations applied.

    Only the fields passed as keyword arguments differ from the parser
    defaults. Input overrides are applied before the deterministic derivations
    so, e.g., ``advantage_estimator="ppo"`` yields ``use_critic=True``;
    explicitly overriding a derived field itself wins. An explicit
    ``None``/``False``/``0``/``[]`` input keeps its meaning instead of being
    replaced by a default.
    """
    args = copy.deepcopy(_collect_defaults())
    input_overrides = {k: v for k, v in overrides.items() if k not in _DERIVED_FIELDS}
    derived_overrides = {k: v for k, v in overrides.items() if k in _DERIVED_FIELDS}
    for key, value in input_overrides.items():
        setattr(args, key, value)
    apply_safe_derivations(args)
    for key, value in derived_overrides.items():
        setattr(args, key, value)
    return args


def reset_defaults_cache() -> None:
    """Drop the cached parsed namespace (tests that stub parser modules)."""
    global _DEFAULTS
    _DEFAULTS = None


__all__ = ["apply_safe_derivations", "make_args", "make_raw_args", "reset_defaults_cache"]
