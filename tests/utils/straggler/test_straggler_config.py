# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the straggler profiler configuration knobs."""

import pytest

from relax.utils.env import Envs
from relax.utils.straggler.config import StragglerConfig


def test_config_disabled_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RELAX_STRAGGLER_ENABLE", raising=False)

    config = StragglerConfig.from_env()

    assert config.enabled is False
    assert config.timer_log_level == 2
    assert config.clamped == {}


@pytest.mark.parametrize("raw", ["1", "true", "True", "yes", "on"])
def test_config_enabled_accepts_truthy_spellings(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("RELAX_STRAGGLER_ENABLE", raw)

    assert StragglerConfig.from_env().enabled is True


def test_config_disabled_for_falsy_spelling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAX_STRAGGLER_ENABLE", "0")

    assert StragglerConfig.from_env().enabled is False


def test_config_clamps_out_of_range_values_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAX_STRAGGLER_ENABLE", "1")
    monkeypatch.setenv("RELAX_STRAGGLER_TIMER_LOG_LEVEL", "7")
    monkeypatch.setenv("RELAX_STRAGGLER_EVENT_POOL", "0")
    monkeypatch.setenv("RELAX_STRAGGLER_PERSIST_WINDOWS", "-3")
    monkeypatch.setenv("RELAX_STRAGGLER_WORK_TOLERANCE", "-1.0")
    monkeypatch.setenv("RELAX_STRAGGLER_WINDOW_S", "0")

    config = StragglerConfig.from_env()

    assert config.timer_log_level == 2
    assert config.event_pool == 512
    assert config.persist_windows == 3
    assert config.work_tolerance == 0.05
    assert config.window_seconds == 5.0
    assert set(config.clamped) == {
        "timer_log_level",
        "event_pool",
        "persist_windows",
        "work_tolerance",
        "window_seconds",
    }


def test_config_keeps_valid_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAX_STRAGGLER_ENABLE", "1")
    monkeypatch.setenv("RELAX_STRAGGLER_TIMER_LOG_LEVEL", "1")
    monkeypatch.setenv("RELAX_STRAGGLER_EVENT_POOL", "64")
    monkeypatch.setenv("RELAX_STRAGGLER_OUTPUT_DIR", "/tmp/straggler")
    monkeypatch.setenv("RELAX_STRAGGLER_COLLECTOR_ADDR", "127.0.0.1:39999")

    config = StragglerConfig.from_env()

    assert config.timer_log_level == 1
    assert config.event_pool == 64
    assert config.output_dir == "/tmp/straggler"
    assert config.collector_addr == "127.0.0.1:39999"
    assert config.clamped == {}


def test_config_zero_tolerance_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero tolerance is a deliberate "any difference counts" setting."""
    monkeypatch.setenv("RELAX_STRAGGLER_WORK_TOLERANCE", "0")

    assert StragglerConfig.from_env().work_tolerance == 0.0


def test_env_properties_are_declared_on_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The knobs must live in the central registry, not in raw os.environ."""
    monkeypatch.setenv("RELAX_STRAGGLER_EVENT_POOL", "128")

    assert Envs.RELAX_STRAGGLER_EVENT_POOL == 128
