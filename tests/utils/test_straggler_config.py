# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.runtime import attach_timers, on_rollout_end, reset_runtime_for_tests


def test_from_args_defaults_off():
    cfg = StragglerConfig.from_args(SimpleNamespace())
    assert cfg.enabled is False
    assert cfg.interval == 10
    assert cfg.persist_windows == 3


def test_from_args_reads_cli_fields():
    args = SimpleNamespace(
        straggler_analysis=True,
        straggler_interval=5,
        straggler_relative_threshold=0.2,
        straggler_absolute_ms_threshold=8.0,
        straggler_persist_windows=2,
        straggler_enable_module_stages=True,
    )
    cfg = StragglerConfig.from_args(args)
    assert cfg.enabled is True
    assert cfg.interval == 5
    assert cfg.relative_threshold == 0.2
    assert cfg.absolute_ms_threshold == 8.0
    assert cfg.persist_windows == 2
    assert cfg.enable_module_stages is True


def test_attach_timers_disabled_keeps_none():
    class C:
        timers = "sentinel"

    config = C()
    attach_timers(config, SimpleNamespace(straggler_analysis=False), role="actor")
    assert config.timers is None


def test_attach_timers_skips_non_actor():
    class C:
        timers = "sentinel"

    config = C()
    args = SimpleNamespace(
        straggler_analysis=True,
        straggler_interval=10,
        straggler_enable_module_stages=False,
    )
    attach_timers(config, args, role="critic")
    assert config.timers is None


def test_on_rollout_end_noop_when_disabled():
    reset_runtime_for_tests()
    on_rollout_end(SimpleNamespace(straggler_analysis=False), rollout_id=1)
