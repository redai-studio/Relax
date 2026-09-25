# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for the A7 blocker: ``config.timers`` and Megatron's
``remove_non_pickleables`` walk.

Megatron's bridge converter runs
``remove_non_pickleables(config, max_depth=3)`` on the model config before
broadcasting it to the other pipeline ranks
(``megatron/bridge/models/conversion/utils.py``, reached from
``MegatronTrainRayActor.train_async -> update_weights_fully_async``). That walk
does ``copy.copy(obj)`` and then ``setattr`` for every attribute in
``vars(obj)``; with the enabled shim in ``config.timers`` it recursed into the
frozen ``StragglerConfig`` and raised
``dataclasses.FrozenInstanceError: cannot assign to field 'enabled'`` at step 0
(job ``mbTivCsZz9YY23ft``).

The regression has four parts, and only the last needs the Megatron checkout:

1. the shim exposes no instance ``__dict__``, which is what keeps the walk from
   recursing into it at all;
2. ``__reduce__`` yields an inert copy that still answers the timer interface;
3. a **local** walker that mirrors upstream ``remove_non_pickleables``
   (``copy.copy`` + ``setattr`` over ``vars()``) proves the frozen-config crash
   happened with the pre-fix shape, and cannot happen with the current
   slotted shim -- this needs no Megatron stack at all, so the invariant is
   enforced on the CPU CI where the import guard below silently skips;
4. the real upstream ``remove_non_pickleables`` accepts a real
   ``TransformerConfig`` with ``get_straggler_timers()`` installed, and the live
   shim still records intervals afterwards.

Part 4 loads the single upstream module by file because the CPU test venv has no
complete Megatron bridge stack (importing ``megatron.bridge`` pulls
``transformer_engine``). It is skipped, with the reason, when no Megatron-LM
checkout is configured through ``MEGATRON_PATH``/``MEGATRON``.
"""

import copy
import dataclasses
import functools
import importlib.util
import os
import pickle
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

import relax.utils.straggler as straggler
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.megatron_timer_shim import NullTimerSink, StragglerTimers


# The CPU venv ships a flashinfer/flashinfer-cubin version pair that fails an
# import-time consistency check; disabling the check is required to import any
# ``megatron.core`` module there. Set before the first megatron import.
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

_UTILS_RELATIVE_PATH = "megatron/bridge/models/conversion/utils.py"


def _megatron_root() -> Optional[str]:
    """Return a Megatron-LM checkout that carries the bridge converter."""
    for variable in ("MEGATRON_PATH", "MEGATRON"):
        root = os.environ.get(variable)
        if root and (Path(root) / _UTILS_RELATIVE_PATH).is_file():
            return root
    try:
        import megatron
    except Exception:
        return None
    root = str(Path(megatron.__file__).resolve().parents[1])
    if (Path(root) / _UTILS_RELATIVE_PATH).is_file():
        return root
    return None


def _load_remove_non_pickleables() -> Optional[Callable[..., Any]]:
    """Load the exact upstream function, or ``None`` when unavailable."""
    root = _megatron_root()
    if root is None:
        return None
    if root not in sys.path:
        sys.path.insert(0, root)
    path = Path(root) / _UTILS_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("_task11_bridge_conversion_utils", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.remove_non_pickleables


def _local_remove_non_pickleables(obj: Any, max_depth: int = 3, current_depth: int = 0) -> Any:
    """Faithful, dependency-free copy of upstream ``remove_non_pickleables``.

    Mirrors ``megatron/bridge/models/conversion/utils.py`` (Megatron core
    0.19.0): depth guard, ``None`` passthrough, ``callable`` removal for
    functions/methods/partials, then the ``copy.copy`` + ``setattr`` walk over
    ``vars()`` that triggers the frozen-dataclass crash, followed by list/tuple/
    dict handling. The ProcessGroup special case is omitted because the test
    objects cannot reach it.
    """
    if current_depth >= max_depth:
        return obj
    if obj is None:
        return obj
    if callable(obj):
        if isinstance(obj, type):
            return obj
        if isinstance(obj, (types.FunctionType, types.MethodType, functools.partial)) or hasattr(obj, "__self__"):
            return None
    if hasattr(obj, "__dict__"):
        cleaned = copy.copy(obj)
        for attr_name in list(vars(cleaned).keys()):
            value = getattr(cleaned, attr_name)
            setattr(cleaned, attr_name, _local_remove_non_pickleables(value, max_depth, current_depth + 1))
        return cleaned
    if isinstance(obj, list):
        return [_local_remove_non_pickleables(item, max_depth, current_depth + 1) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_local_remove_non_pickleables(item, max_depth, current_depth + 1) for item in obj)
    if isinstance(obj, dict):
        return {key: _local_remove_non_pickleables(value, max_depth, current_depth + 1) for key, value in obj.items()}
    return obj


class _PreFixTimers:
    """The shape of ``StragglerTimers`` *before* commit c6d425f: no ``__slots__``.

    It keeps the frozen ``StragglerConfig`` reachable through ``__dict__``,
    exactly like the object that made the bridge converter raise
    ``FrozenInstanceError`` at step 0.
    """

    def __init__(self, config: StragglerConfig) -> None:
        self._config = config
        self._timers: dict = {}
        self._clock = time.perf_counter

    def __call__(self, name: str, log_level: Optional[int] = None) -> Any:
        return None


class _BroadcastConfig:
    """Minimal stand-in for the ``TransformerConfig`` the walk is applied to."""

    def __init__(self, timers: Any) -> None:
        self.timers = timers
        self.some_callable = lambda: None  # noqa: E731 - upstream removes functions
        self.nested = [1, (2, {"k": 3})]


@pytest.fixture(autouse=True)
def _reset_profiler() -> Any:
    straggler.reset_straggler_state_for_tests()
    yield
    straggler.reset_straggler_state_for_tests()


def test_shim_has_no_instance_dict() -> None:
    """The walk only recurses into objects that have ``__dict__``."""
    timers = StragglerTimers(StragglerConfig(enabled=True), sink=NullTimerSink())

    assert not hasattr(timers, "__dict__")
    assert timers._config.enabled is True


def test_pickling_the_shim_yields_an_inert_but_functional_copy() -> None:
    """A broadcast copy must not carry the readout thread or the socket."""
    timers = StragglerTimers(StragglerConfig(enabled=True), sink=NullTimerSink())

    restored = pickle.loads(pickle.dumps(timers))

    assert isinstance(restored, StragglerTimers)
    assert restored is not timers
    # The reconstruct target is disabled and sinkless: it records nothing.
    assert restored._config.enabled is False
    assert isinstance(restored._sink, NullTimerSink)
    # ...but it still answers every call shape Megatron uses.
    handle = restored("forward-compute", log_level=2)
    handle.start()
    handle.stop()
    assert restored.stats()["intervals"] == 1


def test_reduce_is_explicitly_inert_and_functional() -> None:
    """``__reduce__`` is the broadcast path and must not carry live state."""
    timers = StragglerTimers(StragglerConfig(enabled=True), sink=NullTimerSink())

    reconstructor, args = timers.__reduce__()
    restored = reconstructor(*args)

    assert isinstance(restored, StragglerTimers)
    assert restored._config.enabled is False
    assert isinstance(restored._sink, NullTimerSink)
    assert restored.__reduce__()[0] is reconstructor
    handle = restored("forward-compute", log_level=2)
    handle.start()
    handle.stop()
    assert restored.stats()["intervals"] == 1


def test_local_walker_reproduces_the_pre_fix_frozen_config_crash() -> None:
    """The original A7 crash is reproduced without any Megatron checkout.

    This is the counterexample that proves the walker is faithful: the pre-fix
    shape (instance ``__dict__`` holding a frozen ``StragglerConfig``) raises
    exactly the upstream error, so the passing assertions below are meaningful.
    """
    config = _BroadcastConfig(_PreFixTimers(StragglerConfig(enabled=True)))

    with pytest.raises(dataclasses.FrozenInstanceError):
        _local_remove_non_pickleables(config, max_depth=3)


def test_local_walker_leaves_the_current_shim_usable() -> None:
    """The current slotted shim is skipped by the walk and survives broadcast."""
    timers = StragglerTimers(StragglerConfig(enabled=True), sink=NullTimerSink())
    config = _BroadcastConfig(timers)

    cleaned = _local_remove_non_pickleables(config, max_depth=3)

    # No instance __dict__ -> the walk returns the object untouched instead of
    # recursing into the frozen config.
    assert cleaned.timers is timers
    assert cleaned.some_callable is None  # the walk still does its normal job

    # broadcast_obj_from_pp_rank immediately pickles the cleaned config.
    restored = pickle.loads(pickle.dumps(cleaned))
    assert isinstance(restored.timers, StragglerTimers)
    assert restored.timers._config.enabled is False

    handle = timers("forward-compute", log_level=2)
    before = timers.stats()["intervals"]
    handle.start()
    handle.stop()
    assert timers.stats()["intervals"] == before + 1


def test_remove_non_pickleables_leaves_enabled_timers_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end A7 regression: the real walk plus the real broadcast."""
    remove_non_pickleables = _load_remove_non_pickleables()
    if remove_non_pickleables is None:
        pytest.skip(
            "Megatron-LM checkout not available; set MEGATRON_PATH (or MEGATRON) to the "
            "checkout that contains megatron/bridge/models/conversion/utils.py"
        )
    try:
        from megatron.core.transformer.transformer_config import TransformerConfig
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"megatron.core is not importable in this environment: {exc}")

    monkeypatch.setenv("RELAX_STRAGGLER_ENABLE", "1")
    timers = straggler.get_straggler_timers()
    assert isinstance(timers, StragglerTimers)

    config = TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=8)
    config.timers = timers

    # This is the exact call megatron_to_hf makes; before the fix it raised
    # FrozenInstanceError on the frozen StragglerConfig reachable from timers.
    cleaned = remove_non_pickleables(config, max_depth=3)

    assert isinstance(cleaned.timers, StragglerTimers)

    # The walk is immediately followed by broadcast_obj_from_pp_rank, which
    # pickles the cleaned config: that must succeed too.
    restored = pickle.loads(pickle.dumps(cleaned))

    assert isinstance(restored.timers, StragglerTimers)

    # The live shim still records intervals after the walk.
    handle = timers("forward-compute", log_level=2)
    before = timers.stats()["intervals"]
    handle.start()
    handle.stop()

    assert timers.stats()["intervals"] == before + 1
