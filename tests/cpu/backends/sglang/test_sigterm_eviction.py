# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Signal-safety regression tests for graceful elastic eviction."""

import signal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


pytest.importorskip("sglang.srt.server_args")

from relax.backends.sglang.sglang_engine import SGLangEngine


def test_sigterm_handler_only_publishes_eviction_intent():
    engine = SimpleNamespace(_evicted=MagicMock())
    captured = {}

    def _capture(_signal_number, handler):
        captured["handler"] = handler

    with (
        patch("relax.backends.sglang.sglang_engine.signal.getsignal", return_value=signal.SIG_DFL),
        patch("relax.backends.sglang.sglang_engine.signal.signal", side_effect=_capture),
        patch("relax.backends.sglang.sglang_engine.time.sleep") as sleep,
        patch.object(SGLangEngine, "unregister_from_router") as unregister,
    ):
        SGLangEngine._register_sigterm_handler(engine)
        captured["handler"](signal.SIGTERM, None)

    engine._evicted.set.assert_called_once_with()
    sleep.assert_not_called()
    unregister.assert_not_called()
