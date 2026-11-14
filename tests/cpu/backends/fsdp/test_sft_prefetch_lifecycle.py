# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Training/checkpoint failures must stop scheduling speculative CPU reads."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from relax.backends.fsdp.sft import FSDPSFTTrainer, _report_sft_metrics
from relax.models.condition_cache import ConditionFeatureCache


@pytest.fixture
def cached_trainer():
    trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
    cache = ConditionFeatureCache(1024)
    cache.put("condition", {"embedding": torch.ones(4)})
    trainer.actor = SimpleNamespace(model=SimpleNamespace(close_condition_cache=cache.close))
    trainer.prefetch = Mock()
    return trainer, cache


@pytest.mark.parametrize("operation,internal,args", [("train", "_train_step", (0,)), ("save", "_save_checkpoint", ())])
def test_sft_failure_closes_prefetch_and_preserves_error(operation, internal, args, cached_trainer):
    trainer, cache = cached_trainer
    error = RuntimeError("training failed")
    setattr(trainer, internal, Mock(side_effect=error))
    with pytest.raises(RuntimeError) as raised:
        getattr(trainer, operation)(*args)
    assert raised.value is error
    trainer.prefetch.close.assert_called_once()
    assert cache.stats()["bytes"] == 0


def test_sft_success_keeps_prefetch_for_next_step(cached_trainer):
    trainer, cache = cached_trainer
    trainer._train_step = Mock()
    trainer.train(0)
    trainer._train_step.assert_called_once_with(0)
    trainer.prefetch.close.assert_not_called()
    assert cache.stats()["bytes"] > 0


def test_sft_close_without_prefetch(cached_trainer):
    trainer, cache = cached_trainer
    trainer.prefetch = None
    trainer.close()
    assert cache.stats()["bytes"] == 0


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


def test_sft_save_requires_complete_rng_state(monkeypatch):
    from relax.backends.fsdp import sft

    trainer = FSDPSFTTrainer.__new__(FSDPSFTTrainer)
    trainer.step, trainer.rank = 10, 0
    trainer.args = SimpleNamespace(save="/checkpoint")
    trainer.actor = Mock()
    trainer.actor._trainable_mode = "full"
    trainer.scheduler = Mock()
    trainer.contract = {}
    save = Mock()
    monkeypatch.setattr(sft.checkpoint, "save_checkpoint", save)
    trainer._save_checkpoint()
    assert save.call_args.kwargs["require_rng_state"] is True
