# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unified parameter-object builder for the test side.

Design goals (matching the acceptance criteria of task No.13):

- **A single source of defaults**: reuse the real parameter declarations
  registered via ``get_slime_extra_args_provider()``; the namespace produced by
  ``parse_args([])`` already carries the declared defaults, so tests never
  hand-copy a default table.
- **Declare only the diff**: a test overrides the fields it cares about via
  ``build_relax_args(**overrides)``; everything else keeps its declared value,
  so adding a new parameter cannot silently break fixtures.
- **Expose instead of hide**: overriding an undeclared parameter name fails
  immediately, instead of being silently swallowed by
  ``getattr(args, ..., default)`` like before.
"""

import argparse
import copy
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from tests.support.megatron_defaults import MEGATRON_TEST_DEFAULTS


_IMPORT_GUARD_READY = False
_BASE_ARGS: dict[str, Any] | None = None


def _ensure_router_stub() -> None:
    """Ensure RouterArgs has the add_cli_args signature.

    Other tests (e.g. test_arguments_spec_driven) replace RouterArgs with
    ``object`` to isolate dependencies. This builder reuses the same module
    object, so a callable stub must be restored before validation. The real
    sglang_router is unaffected.
    """
    module = sys.modules.get("sglang_router.launch_router")
    if module is None:
        return
    router_args = getattr(module, "RouterArgs", None)
    if not hasattr(router_args, "add_cli_args"):
        module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)


def _install_optional_dependency_stubs() -> None:
    """Make the argument-declaration modules importable without sglang.

    CPU CI does not install sglang/sglang_router, but ``relax.utils.arguments``
    references them at module import time. Only minimal stubs are installed:
    the parameter declarations themselves come from Relax, and the stubs never
    affect any default value. When the real dependencies exist, nothing is
    replaced.
    """
    global _IMPORT_GUARD_READY
    _ensure_router_stub()
    if _IMPORT_GUARD_READY:
        return

    try:
        import sglang_router.launch_router  # noqa: F401
    except ImportError:
        router_pkg = ModuleType("sglang_router")
        launch_router = ModuleType("sglang_router.launch_router")
        # add_cli_args is provided by RouterArgs; the stub only needs the call signature.
        launch_router.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
        sys.modules.setdefault("sglang_router", router_pkg)
        sys.modules.setdefault("sglang_router.launch_router", launch_router)

    try:
        import relax.backends.sglang.arguments as sglang_arguments_module
    except ImportError:
        sglang_arguments_module = ModuleType("relax.backends.sglang.arguments")
        sglang_arguments_module.sglang_parse_args = lambda: None
        sglang_arguments_module.validate_args = lambda args: args
        sys.modules["relax.backends.sglang.arguments"] = sglang_arguments_module

    # Other tests may swap in a simpler stub; restore the entries the builder needs.
    if not hasattr(sglang_arguments_module, "add_sglang_arguments"):
        sglang_arguments_module.add_sglang_arguments = lambda parser: parser
    if not hasattr(sglang_arguments_module, "sglang_parse_args"):
        sglang_arguments_module.sglang_parse_args = lambda: None

    _IMPORT_GUARD_READY = True


def build_relax_argparser() -> argparse.ArgumentParser:
    """Build the Relax argument parser that matches production (without
    Megatron)."""
    _install_optional_dependency_stubs()

    from relax.utils import arguments as relax_arguments

    # Other tests may replace RouterArgs with object to isolate dependencies;
    # the module-level name was bound at import time, so fix it in place to
    # keep the builder independent of test execution order.
    if not hasattr(relax_arguments.RouterArgs, "add_cli_args"):
        relax_arguments.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)

    get_slime_extra_args_provider = relax_arguments.get_slime_extra_args_provider

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    get_slime_extra_args_provider()(parser)
    return parser


def _base_attributes() -> dict[str, Any]:
    """Produce a full attribute table in production parsing order: main parser.

    -> pre-parse -> finalize.
    """
    global _BASE_ARGS
    if _BASE_ARGS is not None:
        return _BASE_ARGS

    _install_optional_dependency_stubs()
    from relax.utils.args_finalize import finalize_args
    from relax.utils.arguments import build_pre_parse_parser

    args = build_relax_argparser().parse_args([])
    existing = vars(args)
    for name, value in vars(build_pre_parse_parser().parse_args([])).items():
        existing.setdefault(name, value)
    finalize_args(args)
    _BASE_ARGS = dict(existing)
    return _BASE_ARGS


def declared_parameter_names() -> set[str]:
    """Return every usable parameter name (single source: declarations +
    finalize-filled fields)."""
    return set(_base_attributes()) | set(MEGATRON_TEST_DEFAULTS)


def declared_parameter_defaults() -> dict[str, Any]:
    """Return the declared-defaults dict, for the contract test and
    before/after comparison."""
    return copy.deepcopy(_base_attributes())


def build_relax_args(**overrides: Any) -> argparse.Namespace:
    """Build a fully populated parameter object; tests declare only their own
    diff.

    Args:
        **overrides: parameter name -> value. Names must come from the
            declarations, the finalize-filled fields, or
            ``MEGATRON_TEST_DEFAULTS``; a typo raises ``AttributeError``
            (intentional: refuse to swallow mistakes silently).

    Returns:
        An ``argparse.Namespace`` isomorphic to the production parse result.
    """
    base = copy.deepcopy(_base_attributes())
    known = set(base) | set(MEGATRON_TEST_DEFAULTS)
    # sglang / router fields are declared by the sglang sub-parser; environments
    # without sglang installed (CPU CI) cannot see those declarations, so names
    # with these prefixes are allowed through and their values must be given
    # explicitly by the test.
    unknown = sorted(name for name in set(overrides) - known if not name.startswith(("sglang_", "router_")))
    if unknown:
        raise AttributeError(f"unknown Relax argument(s) {unknown}; declared names come from relax/utils/arguments.py")
    args = argparse.Namespace(**{**copy.deepcopy(MEGATRON_TEST_DEFAULTS), **base})
    for name, value in overrides.items():
        setattr(args, name, value)
    return args
