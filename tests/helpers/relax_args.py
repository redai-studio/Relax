# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build incomplete test args from Relax argparse defaults.

Defaults come from a real empty ``parse_args([])`` of slime /
``_pre_parse_mode`` registrations (so ``type=`` converters run), not a
hand-maintained table. Callers only pass overrides for fields under test.

Runtime-only fields (``model_source``, ``is_vl_model``, ...) and derived role
switches (``use_critic``) come from the same production helpers that parse /
validate call, so the factory never keeps its own placeholder table.

Optional import stubs used while collecting defaults are scoped and restored
so later tests can still import real SGLang modules.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Iterator


_DEFAULTS: dict[str, Any] | None = None
# Production helpers captured while ``relax.utils.arguments`` is importable
# under the scoped stubs; refreshed whenever ``_DEFAULTS`` is recollected.
_RUNTIME_HOOKS: dict[str, Callable[..., Any]] = {}


def _ensure_module(name: str, **attrs: Any) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        for key, value in attrs.items():
            if not hasattr(existing, key):
                setattr(existing, key, value)
        return existing
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


@contextmanager
def _temporary_optional_import_stubs() -> Iterator[None]:
    """Install SGLang stubs only while collecting argparse defaults."""

    class _RouterArgs:
        @staticmethod
        def add_cli_args(parser, **_kwargs):
            return parser

        @staticmethod
        def from_cli_args(args):
            return args

    watched = (
        "sglang_router",
        "sglang_router.launch_router",
        "relax.backends.sglang.arguments",
        "relax.utils.arguments",
    )
    previous = {name: sys.modules.get(name) for name in watched}
    utils_pkg = sys.modules.get("relax.utils")
    # ``importlib.import_module`` also binds ``relax.utils.arguments`` as a package
    # attribute; popping ``sys.modules`` alone leaves that cache in place.
    _missing = object()
    previous_utils_arguments_attr = getattr(utils_pkg, "arguments", _missing) if utils_pkg is not None else _missing
    # Only fabricate modules that are truly absent so we never shadow a real install.
    created: set[str] = set()

    def _stub_if_missing(name: str, **attrs: Any) -> None:
        if name in sys.modules:
            return
        _ensure_module(name, **attrs)
        created.add(name)

    # Existing test fixtures may leave ``RouterArgs = object`` on an already-imported
    # arguments module; temporarily adapt those so default collection stays independent.
    router_args_restores: list[tuple[Any, str, Any]] = []

    def _adapt_router_args(target: Any, attr: str = "RouterArgs") -> None:
        if target is None or not hasattr(target, attr):
            return
        current = getattr(target, attr)
        if callable(getattr(current, "add_cli_args", None)):
            return
        router_args_restores.append((target, attr, current))
        setattr(target, attr, _RouterArgs)

    try:
        _stub_if_missing("sglang_router")
        _stub_if_missing("sglang_router.launch_router", RouterArgs=_RouterArgs)
        _stub_if_missing(
            "relax.backends.sglang.arguments",
            sglang_parse_args=lambda: None,
            validate_args=lambda args: args,
            add_sglang_arguments=lambda parser: parser,
            ServerArgs=object,
        )
        _adapt_router_args(sys.modules.get("sglang_router.launch_router"))
        _adapt_router_args(sys.modules.get("relax.utils.arguments"))
        yield
    finally:
        for target, attr, old in reversed(router_args_restores):
            setattr(target, attr, old)
        for name in created:
            sys.modules.pop(name, None)
        for name, mod in previous.items():
            if mod is not None:
                sys.modules[name] = mod
            elif name in created:
                sys.modules.pop(name, None)
        # If we imported arguments under stubs, drop the cached module so a later
        # import rebinds against real optional deps when available.
        if previous["relax.utils.arguments"] is None:
            sys.modules.pop("relax.utils.arguments", None)
            utils_pkg = sys.modules.get("relax.utils")
            if utils_pkg is not None:
                if previous_utils_arguments_attr is _missing:
                    if hasattr(utils_pkg, "arguments"):
                        delattr(utils_pkg, "arguments")
                else:
                    setattr(utils_pkg, "arguments", previous_utils_arguments_attr)


def slime_arg_defaults() -> dict[str, Any]:
    """Return a deep copy of declared slime / pre-parse argument defaults."""
    global _DEFAULTS
    if _DEFAULTS is None:
        with _temporary_optional_import_stubs():
            module = importlib.import_module("relax.utils.arguments")
            parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
            module.get_slime_extra_args_provider()(parser)
            # parse_args applies type= converters (e.g. json.loads default "{}").
            defaults: dict[str, Any] = dict(vars(parser.parse_args([])))
            defaults.pop("help", None)

            argv = sys.argv
            sys.argv = [argv[0] if argv else "relax"]
            try:
                pre = module._pre_parse_mode()
            finally:
                sys.argv = argv
            for key, value in vars(pre).items():
                defaults.setdefault(key, value)

            _RUNTIME_HOOKS.clear()
            _RUNTIME_HOOKS["init_runtime_args"] = module.init_runtime_args
            _RUNTIME_HOOKS["derive_use_critic"] = module.derive_use_critic
            _DEFAULTS = defaults
    return copy.deepcopy(_DEFAULTS)


def make_args(**overrides: Any) -> SimpleNamespace:
    """Namespace as produced by parse: declared defaults, runtime-only fields,
    then ``overrides`` on top.

    ``use_critic`` is derived from the final values with the production helper
    unless the caller overrides it explicitly.
    """
    args = SimpleNamespace(**slime_arg_defaults())
    _RUNTIME_HOOKS["init_runtime_args"](args)
    for key, value in overrides.items():
        setattr(args, key, value)
    if "use_critic" not in overrides:
        args.use_critic = _RUNTIME_HOOKS["derive_use_critic"](args)
    return args
