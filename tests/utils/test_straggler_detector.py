# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.detector import JudgeState, RankWindow, judge, metrics_from
from relax.utils.straggler.stages import stage_of


def _row(rank, pp, fwd, tokens=100.0, gc=0.0, sync=10.0, recv=1.0):
    return RankWindow(
        rank=rank,
        pp=pp,
        stages_ms={"fwd": fwd, "bwd": fwd, "optim": 1.0, "dp_grad_sync": sync, "pp_recv": recv, "pp_send": 1.0},
        tokens=tokens,
        gc_ms=gc,
    )


def test_timer_names_map_to_coarse_stages():
    assert stage_of("forward-compute") == "fwd"
    assert stage_of("backward-compute") == "bwd"
    assert stage_of("forward-send-backward-recv") == "pp_recv"
    assert stage_of("all-grads-sync") == "dp_grad_sync"
    assert stage_of("optimizer-inner-step") == "optim"
    assert stage_of("forward-backward") is None
    assert stage_of("self-attention") is None
    assert stage_of("self-attention", module_stages=True) == "attention"


def test_slow_device_needs_three_windows():
    cfg = StragglerConfig(persist_windows=3, relative_threshold=0.10, absolute_ms_threshold=5.0)
    state = JudgeState()
    rows = [_row(0, 0, 10.0), _row(1, 0, 40.0)]
    first, state = judge(rows, cfg, state)
    second, state = judge(rows, cfg, state)
    third, state = judge(rows, cfg, state)
    assert first == []
    assert second == []
    assert len(third) == 1
    assert third[0].rank == 1
    assert third[0].reason == "slow_device"


def test_extra_tokens_are_data_imbalance():
    cfg = StragglerConfig(persist_windows=1, relative_threshold=0.10, absolute_ms_threshold=1.0)
    rows = [_row(0, 0, 10.0, tokens=100), _row(1, 0, 40.0, tokens=400)]
    alerts, _ = judge(rows, cfg, JudgeState())
    assert alerts[0].reason == "data_imbalance"


def test_short_grad_sync_is_late_arrival():
    cfg = StragglerConfig(persist_windows=1, relative_threshold=0.10, absolute_ms_threshold=5.0)
    rows = [_row(0, 0, 10.0, sync=20.0), _row(1, 0, 10.0, sync=20.0), _row(2, 0, 10.0, sync=2.0)]
    alerts, _ = judge(rows, cfg, JudgeState())
    assert [(item.rank, item.reason) for item in alerts] == [(2, "late_arrival")]


def test_single_rank_in_a_stage_is_not_flagged():
    cfg = StragglerConfig(persist_windows=1)
    alerts, _ = judge([_row(0, 0, 100.0)], cfg, JudgeState())
    assert alerts == []


def test_metrics_keep_upstream_wait_out_of_flagged():
    cfg = StragglerConfig(persist_windows=1, relative_threshold=0.10, absolute_ms_threshold=1.0)
    rows = [_row(0, 0, 10.0, recv=1.0), _row(1, 0, 10.0, recv=50.0)]
    alerts, _ = judge(rows, cfg, JudgeState())
    metrics = metrics_from(rows, alerts, dropped=0)
    assert alerts[0].reason == "upstream_wait"
    assert metrics["straggler/flagged/count"] == 0.0
    assert "straggler/fwd/max_rank" in metrics
