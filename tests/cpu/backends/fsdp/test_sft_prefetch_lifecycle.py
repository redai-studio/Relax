# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Training/checkpoint failures must stop scheduling speculative CPU reads."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.backends.fsdp.sft import FSDPSFTTrainer, _report_sft_metrics


@pytest.mark.parametrize("operation,internal,args", [("train", "_train_step", (0,)), ("save", "_save_checkpoint", ())])
def test_sft_failure_closes_prefetch_and_preserves_error(operation, internal, args):
    trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
    trainer.prefetch = Mock()
    error = RuntimeError("training failed")
    setattr(trainer, internal, Mock(side_effect=error))
    with pytest.raises(RuntimeError) as raised:
        getattr(trainer, operation)(*args)
    assert raised.value is error
    trainer.prefetch.close.assert_called_once()


def test_sft_success_keeps_prefetch_for_next_step():
    trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
    trainer.prefetch = Mock()
    trainer._train_step = Mock()
    trainer.train(0)
    trainer._train_step.assert_called_once_with(0)
    trainer.prefetch.close.assert_not_called()


@pytest.mark.parametrize("operation,internal,args", [("train", "_train_step", (0,)), ("save", "_save_checkpoint", ())])
def test_cleanup_interruption_preserves_original_training_error(operation, internal, args):
    trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
    error = RuntimeError("original failure")
    setattr(trainer, internal, Mock(side_effect=error))
    trainer.close = Mock(side_effect=KeyboardInterrupt("cleanup interrupted"))
    with pytest.raises(RuntimeError) as raised:
        getattr(trainer, operation)(*args)
    assert raised.value is error
    trainer.close.assert_called_once()


def test_sft_close_without_prefetch():
    trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
    trainer.prefetch = None
    trainer.close()


def test_condition_lookahead_uses_ready_future_samples_without_advancing_cursor():
    trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
    trainer.prefetch = Mock()
    future_data = [{"prompt": "next"}, {"prompt": "later"}]
    trainer.prefetch.ready_samples.return_value = [(1, 0, 3, future_data[0]), (2, 0, 9, future_data[1])]
    trainer.conditioning = SimpleNamespace(prefetch_enabled=True, prefetch_depth=2, prefetch_many=Mock())
    describe = Mock(side_effect=lambda data: data["prompt"])
    trainer.actor = SimpleNamespace(model=SimpleNamespace(conditioning_prefetch_spec=describe))
    trainer._prefetch_conditioning()
    trainer.prefetch.ready_samples.assert_called_once_with(2)
    trainer.prefetch.get.assert_not_called()
    trainer.conditioning.prefetch_many.assert_called_once_with(["next", "later"])


def test_condition_lookahead_error_discards_window_and_preserves_online_fallback():
    trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
    trainer.prefetch = Mock()
    trainer.prefetch.ready_samples.return_value = [(1, 0, 3, {})]
    trainer.conditioning = SimpleNamespace(prefetch_enabled=True, prefetch_depth=1, prefetch_many=Mock())
    describe = Mock(side_effect=ValueError("bad speculative input"))
    trainer.actor = SimpleNamespace(model=SimpleNamespace(conditioning_prefetch_spec=describe))
    trainer._prefetch_conditioning()
    trainer.conditioning.prefetch_many.assert_called_once_with([])


def test_condition_lookahead_reuses_only_retained_cursor_hints_and_stays_bounded():
    trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
    trainer.prefetch = Mock()
    trainer.conditioning = SimpleNamespace(prefetch_enabled=True, prefetch_depth=2, prefetch_many=Mock())
    describe = Mock(side_effect=lambda data: dict(data))
    trainer.actor = SimpleNamespace(model=SimpleNamespace(conditioning_prefetch_spec=describe))
    first, second, third = {"prompt": "a"}, {"prompt": "b"}, {"prompt": "a"}
    trainer.prefetch.ready_samples.return_value = [(1, 0, 3, first), (2, 0, 9, second)]
    trainer._prefetch_conditioning()
    trainer._prefetch_conditioning()
    assert describe.call_count == 2
    trainer.prefetch.ready_samples.return_value = [(2, 0, 9, second), (3, 0, 3, third)]
    trainer._prefetch_conditioning()
    # A repeated sample at a different cursor is described independently.
    assert describe.call_count == 3
    assert set(trainer._conditioning_hints) == {(2, 0, 9), (3, 0, 3)}
    assert trainer.conditioning.prefetch_many.call_args.args[0] == [second, third]
    trainer.prefetch.ready_samples.return_value = []
    trainer._prefetch_conditioning()
    assert trainer._conditioning_hints == {}


@pytest.mark.parametrize("outcome", ["disabled", "error", "close"])
def test_condition_lookahead_drops_retained_hints_on_exit(outcome):
    trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
    trainer.prefetch = Mock()
    trainer.prefetch.ready_samples.return_value = [(2, 0, 9, {})]
    trainer._conditioning_hints = {(1, 0, 3): object()}
    trainer.conditioning = SimpleNamespace(
        prefetch_enabled=outcome != "disabled", prefetch_depth=2, prefetch_many=Mock(), close=Mock()
    )
    trainer.step = 1
    describe = Mock(side_effect=ValueError("bad speculative input"))
    trainer.actor = SimpleNamespace(model=SimpleNamespace(conditioning_prefetch_spec=describe))
    trainer.close() if outcome == "close" else trainer._prefetch_conditioning()
    assert trainer._conditioning_hints == {}


def test_sft_metrics_flush_exact_completed_step(monkeypatch):
    tracking = ModuleType("relax.utils.tracking_utils")
    calls = []
    tracking.log = lambda args, values, step_key: calls.append(("log", values[step_key]))
    tracking.flush_metrics = lambda args, step: calls.append(("flush", step))
    monkeypatch.setitem(sys.modules, tracking.__name__, tracking)
    _report_sft_metrics(Mock(), {"train/step": 7, "perf/train_seconds": 1.0})
    assert calls == [("log", 7), ("flush", 7)]


@pytest.mark.parametrize("raises", [False, True])
def test_sft_metric_flush_failure_does_not_fail_update(monkeypatch, raises):
    tracking = ModuleType("relax.utils.tracking_utils")
    tracking.log = Mock()
    tracking.flush_metrics = Mock(
        side_effect=RuntimeError("metrics unavailable") if raises else None, return_value=False
    )
    monkeypatch.setitem(sys.modules, tracking.__name__, tracking)
    _report_sft_metrics(Mock(), {"train/step": 7})
    tracking.flush_metrics.assert_called_once()


def test_sft_metrics_flush_even_when_logging_fails(monkeypatch):
    tracking = ModuleType("relax.utils.tracking_utils")
    error = RuntimeError("logging failed")
    tracking.log = Mock(side_effect=error)
    tracking.flush_metrics = Mock()
    monkeypatch.setitem(sys.modules, tracking.__name__, tracking)
    with pytest.raises(RuntimeError) as raised:
        _report_sft_metrics(Mock(), {"train/step": 7})
    assert raised.value is error
    tracking.flush_metrics.assert_called_once()


@pytest.mark.parametrize("task", ["i2i_sft", "t2i_sft"])
def test_sft_save_requires_complete_rng_state(monkeypatch, task):
    from relax.backends.fsdp import sft

    trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
    trainer.step, trainer.rank = 10, 0
    trainer.task = task
    trainer.args = SimpleNamespace(save="/checkpoint")
    trainer.actor = Mock()
    trainer.actor._trainable_mode = "full"
    trainer.scheduler = Mock()
    trainer.contract = {}
    save = Mock()
    monkeypatch.setattr(sft.checkpoint, "save_checkpoint", save)
    trainer._save_checkpoint()
    assert save.call_args.args[1] == task
    assert save.call_args.kwargs["require_rng_state"] is True
