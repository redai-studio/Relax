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

The regression has three parts, and only the third needs the Megatron checkout:

1. the shim exposes no instance ``__dict__``, which is what keeps the walk from
   recursing into it at all;
2. pickling the shim (the broadcast that immediately follows the walk) yields an
   inert copy that still answers the timer interface;
3. the real upstream ``remove_non_pickleables`` accepts a real
   ``TransformerConfig`` with ``get_straggler_timers()`` installed, and the live
   shim still records intervals afterwards.

Part 3 loads the single upstream module by file because the CPU test venv has no
complete Megatron bridge stack (importing ``megatron.bridge`` pulls
``transformer_engine``). It is skipped, with the reason, when no Megatron-LM
checkout is configured through ``MEGATRON_PATH``/``MEGATRON``.
"""

import importlib.util
import os
import pickle
import sys
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
