# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reviewed ordinary-loader capabilities, owned by the SGLang adapter.

Unknown functions, caches, quantization and storage transforms retain ordinary
FULL input. This is not a model selector or another synchronization protocol.
Inherited reviewed loaders reuse the same capability without size/name flags.
"""

from __future__ import annotations

from typing import Any

from relax.utils.sparse_tensor import storage_identity

from .delta_model_loaders import stock_loader_identity, stock_requires_full_input


def full_loader_inputs(model: Any) -> frozenset[str]:
    """Aliased named inputs must preserve ordinary FULL's last-writer order."""
    aliases = {}
    # The reviewed SG loaders build params_dict from named_parameters(), which
    # omits duplicate names of the exact same Parameter. Distinct Parameters
    # sharing storage remain separate ordinary loader inputs.
    for name, parameter in model.named_parameters():
        aliases.setdefault(storage_identity(parameter), set()).add(name)
    return frozenset(name for names in aliases.values() if len(names) > 1 for name in names)


def declared_copy_layout(model: Any) -> tuple | None:
    """Optional model-owned contract for static, independent copy-only inputs.

    The declaration belongs to the actual dispatcher function. Its callable
    returns (immutable layout epoch, frozenset of input names). For each
    declared input the only allowed effect is a synchronous view copy to a
    disjoint destination (or a rank-local no-op). It must bump the epoch on any
    routing/geometry change and may not retain the input. Invocation-level
    hooks still execute. Returning None declines the optimization.
    """
    function = getattr(model.load_weights, "__func__", None)
    declare = getattr(function, "relax_copy_layout", None)
    if declare is None or getattr(model.load_weights, "__self__", None) is not model:
        return None
    layout = declare(model)
    if layout is None:
        return None
    epoch, names = layout
    if type(epoch) not in (int, str) or type(names) is not frozenset or any(type(n) is not str for n in names):
        raise ValueError("Invalid copy-only loader layout declaration")
    return function, declare, epoch, names


def requires_full_input(model: Any, name: str) -> bool:
    declared = declared_copy_layout(model)
    if declared is not None:
        return name not in declared[-1]
    return stock_requires_full_input(model, name)


def sparse_loader_identity(model: Any) -> tuple | None:
    loader = getattr(model, "load_weights", None)
    function = getattr(loader, "__func__", None)
    if function is None or getattr(loader, "__self__", None) is not model:
        return None
    declared = declared_copy_layout(model)
    if declared is not None:
        return declared
    return stock_loader_identity(model, function, full_loader_inputs(model))
