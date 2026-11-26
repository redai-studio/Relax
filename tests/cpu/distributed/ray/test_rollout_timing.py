# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

from relax.distributed.ray import rollout as rollout_logging


def test_rollout_emits_compact_timing_without_dropping_tracking_metrics(monkeypatch):
    args = Namespace(custom_rollout_log_function_path=None, load_debug_rollout_data=False)
    log, tracking = Mock(), Mock()
    metrics = {f"perf_detail/noise/{i}": i for i in range(100)}
    metrics.update({"perf/rollout_time": 12.3456, "perf/reward_time/critical": 5.0})
    monkeypatch.setattr(rollout_logging, "logger", log)
    monkeypatch.setattr(rollout_logging, "save_rollout_result_jsonl", Mock())
    monkeypatch.setattr(rollout_logging, "compute_rollout_step", lambda args, step: step)
    monkeypatch.setattr(rollout_logging, "tracking_utils", SimpleNamespace(log=tracking, flush_metrics=Mock()))
    rollout_logging._log_rollout_data(2, args, [], metrics, 12.3456)
    compact = [call.args[0] for call in log.info.call_args_list if call.args[0].startswith("rollout timing ")]
    assert len(compact) == 1, "no compact timing output was emitted alongside the large metric dump"
    assert "critical" in compact[0] and "12.35" in compact[0]
    assert "noise" not in compact[0]
    assert tracking.call_args.args[1].items() >= metrics.items()
