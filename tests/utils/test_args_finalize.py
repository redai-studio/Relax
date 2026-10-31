# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the argument finalization stage (``args_finalize``).

The finalization stage is the middle segment of "declarations -> finalize ->
validate". This file verifies its two contracts:

1. Only **missing** fields are filled and existing values are never
   overwritten (idempotent);
2. When the sub-parser is unavailable (optional dependency missing, or the
   test double lacks the declaration API), it skips safely without blocking
   parsing.
"""

import sys
from argparse import Namespace
from types import ModuleType

from relax.utils import args_finalize


def _install_sglang_stub(monkeypatch, **attributes):
    """Replace ``relax.backends.sglang.arguments`` with a test double.

    Patch both ``sys.modules`` and the package attribute: ``from package import
    name`` checks the package attribute first, so patching ``sys.modules``
    alone is not guaranteed to take effect.
    """
    stub = ModuleType("relax.backends.sglang.arguments")
    for name, value in attributes.items():
        setattr(stub, name, value)
    monkeypatch.setitem(sys.modules, "relax.backends.sglang.arguments", stub)

    package = sys.modules.get("relax.backends.sglang")
    if package is not None:
        monkeypatch.setattr(package, "arguments", stub, raising=False)
    return stub


def test_seed_runtime_derived_fills_missing_only():
    args = Namespace(is_vl_model=True)
    args_finalize.seed_runtime_derived(args)

    assert args.is_vl_model is True, "existing values must not be overwritten"
    assert args.uses_unsplit_forward is False
    assert args.use_critic is False
    assert set(args_finalize.RUNTIME_DERIVED_DEFAULTS) <= set(vars(args))


def test_finalize_args_is_idempotent():
    args = Namespace()
    args_finalize.finalize_args(args)
    snapshot = dict(vars(args))
    args_finalize.finalize_args(args)
    assert vars(args) == snapshot


def test_finalize_args_only_fills_missing_subparser_fields(monkeypatch):
    def add_sglang_arguments(parser):
        parser.add_argument("--sglang-dp-size", type=int, default=1)
        parser.add_argument("--sglang-pp-size", type=int, default=2)
        return parser

    _install_sglang_stub(monkeypatch, add_sglang_arguments=add_sglang_arguments)
    args = Namespace(sglang_dp_size=8)

    args_finalize.finalize_args(args)

    assert args.sglang_dp_size == 8, "explicitly parsed values must not be overwritten by declared defaults"
    assert args.sglang_pp_size == 2


def test_finalize_args_skips_stub_without_declaration_api(monkeypatch):
    """When the double only provides the parse entry, skip completion instead
    of raising AttributeError."""
    _install_sglang_stub(monkeypatch, sglang_parse_args=lambda: None, validate_args=lambda args: args)

    args = Namespace()
    args_finalize.finalize_args(args)

    # Keep only the runtime-derived attributes, without any sglang fields.
    assert set(vars(args)) == set(args_finalize.RUNTIME_DERIVED_DEFAULTS)
