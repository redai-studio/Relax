# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the straggler window analysis (pure numpy)."""

import numpy as np
import pytest

from relax.utils.straggler.detector import (
    REASON_CODE,
    DetectorConfig,
    DetectorState,
    RankMeta,
    _leave_one_out_mad,
    _leave_one_out_median,
    _relative_and_z,
    analyze_window,
)
from tests.utils.straggler_helpers import meta as _meta
from tests.utils.straggler_helpers import row as _row


def _healthy_row(steps=10, fwd=100.0, bwd=200.0, optim=20.0, tokens=8000.0, dp_grad_sync=5.0, **extra):
    return _row(steps=steps, fwd=fwd, bwd=bwd, optim=optim, tokens=tokens, dp_grad_sync=dp_grad_sync, **extra)


def _analyze_once(table, meta, **config):
    """Analyze one window with a fresh state, flagging on the first window."""
    return analyze_window(table, meta, DetectorConfig(**{"persist_windows": 1, **config}), DetectorState())


def _reasons(report):
    return [(alert.rank, alert.reason) for alert in report.alerts]


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 9])
def test_leave_one_out_median_matches_brute_force(n):
    rng = np.random.default_rng(n)
    for values in (rng.random(n) * 100.0, np.repeat(7.0, n), np.arange(n, dtype=float)):
        expected = np.array([np.median(np.delete(values, i)) for i in range(n)])
        assert _leave_one_out_median(values) == pytest.approx(expected)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 9, 64])
def test_leave_one_out_mad_matches_brute_force(n):
    rng = np.random.default_rng(n)
    ties = rng.integers(0, 3, n).astype(float)
    for values in (rng.random(n) * 100.0, ties, np.repeat(7.0, n), np.eye(1, n).ravel()):
        med = _leave_one_out_median(values)
        expected = [np.median(np.abs(np.delete(values, i) - med[i])) for i in range(n)]
        assert _leave_one_out_mad(values, med) == pytest.approx(expected)


def test_relative_and_z_matches_brute_force_definition():
    values = np.array([100.0, 101.0, 99.0, 130.0, 100.5, 0.0])
    rel, z, med = _relative_and_z(values)
    for i, value in enumerate(values):
        peers = np.delete(values, i)
        peer_med = np.median(peers)
        mad = np.median(np.abs(peers - peer_med))
        assert med[i] == pytest.approx(peer_med)
        assert rel[i] == pytest.approx(min(value / peer_med - 1.0, 100.0))
        assert z[i] == pytest.approx((value - peer_med) / (1.4826 * mad))
    rel, z, med = _relative_and_z(np.array([5.0]))
    assert rel.tolist() == [0.0] and z.tolist() == [0.0] and med.tolist() == [5.0]


def test_healthy_window_raises_no_flag_and_reports_per_step_values():
    table = [_healthy_row() for _ in range(8)]
    report = analyze_window(table, _meta(8), DetectorConfig(), DetectorState())
    assert report.metrics["straggler/flagged/count"] == 0
    assert report.metrics["straggler/flagged/rank"] == -1
    assert report.metrics["straggler/self/median_ms"] == pytest.approx(320.0)
    assert report.metrics["straggler/fwd/median_ms"] == pytest.approx(100.0)
    assert report.metrics["straggler/pp_stage_imbalance"] == 1.0
    assert report.alerts == []
    assert len(report.rows) == 8 and report.rows[3]["rank"] == 3


def test_slow_device_flagged_only_after_persist_windows():
    config = DetectorConfig(persist_windows=3)
    state = DetectorState()
    table = [_healthy_row() for _ in range(8)]
    table[5] = _healthy_row(fwd=130.0, bwd=260.0)  # fwd/bwd 30% slower, same tokens; self = 410 vs 320
    for window in range(3):
        report = analyze_window(table, _meta(8), config, state)
        if window < 2:
            assert report.metrics["straggler/flagged/count"] == 0
            assert report.metrics["straggler/self/max_rank"] == 5
            assert report.metrics["straggler/self/spread"] == pytest.approx(410.0 / 320.0 - 1.0, abs=1e-6)
    assert report.metrics["straggler/flagged/count"] == 1
    assert report.metrics["straggler/flagged/rank"] == 5
    assert report.metrics["straggler/flagged/reason"] == REASON_CODE["slow_device"]
    assert [alert.rank for alert in report.alerts] == [5]
    assert report.alerts[0].new is True
    assert "slow_device" in report.alerts[0].message
    # Same state again: still flagged but no longer "new".
    report = analyze_window(table, _meta(8), config, state)
    assert report.alerts[0].new is False


def test_flag_clears_when_rank_recovers():
    config = DetectorConfig(persist_windows=1)
    state = DetectorState()
    slow = [_healthy_row() for _ in range(4)]
    slow[2] = _healthy_row(fwd=150.0, bwd=300.0)
    assert analyze_window(slow, _meta(4), config, state).metrics["straggler/flagged/count"] == 1
    healthy = [_healthy_row() for _ in range(4)]
    report = analyze_window(healthy, _meta(4), config, state)
    assert report.metrics["straggler/flagged/count"] == 0
    assert state.consecutive == {} and state.active == {}


def test_token_heavy_rank_is_data_imbalance_not_slow_device():
    table = [_healthy_row() for _ in range(8)]
    # 40% more tokens and proportionally more compute: ms/ktok unchanged.
    table[1] = _healthy_row(fwd=140.0, bwd=280.0, optim=20.0, tokens=11200.0)
    report = _analyze_once(table, _meta(8))
    assert report.metrics["straggler/flagged/rank"] == 1
    assert report.metrics["straggler/flagged/reason"] == REASON_CODE["data_imbalance"]
    assert report.metrics["straggler/tokens/spread"] == pytest.approx(0.4, abs=1e-6)
    assert report.metrics["straggler/self_per_ktok/spread"] == pytest.approx(0.0, abs=0.02)


def test_gc_heavy_rank_is_cpu_bound():
    table = [_healthy_row() for _ in range(4)]
    table[3] = _healthy_row(fwd=130.0, bwd=260.0, gc=40.0)
    report = _analyze_once(table, _meta(4))
    assert report.metrics["straggler/flagged/reason"] == REASON_CODE["cpu_bound"]
    assert report.metrics["straggler/gc/max_ms"] == pytest.approx(40.0)


def test_pp_groups_are_compared_separately_and_stage_imbalance_reported():
    # PP=2, 4 ranks per stage; stage 1 is uniformly 50% heavier (uneven split),
    # which must not flag anyone.
    table = [_healthy_row() for _ in range(4)] + [_healthy_row(fwd=150.0, bwd=300.0, optim=30.0) for _ in range(4)]
    report = _analyze_once(table, _meta(8, pp_size=2))
    assert report.metrics["straggler/flagged/count"] == 0
    assert report.metrics["straggler/pp_stage_imbalance"] == pytest.approx(1.5)


def test_downstream_stage_waiting_on_slow_upstream_is_not_flagged_as_slow():
    table = [_healthy_row() for _ in range(4)] + [_healthy_row(fwd=150.0, bwd=300.0, optim=30.0) for _ in range(4)]
    table[6] = _healthy_row(fwd=150.0, bwd=300.0, optim=30.0, pp_recv=80.0)  # waits much more than its stage peers
    report = _analyze_once(table, _meta(8, pp_size=2))
    assert report.metrics["straggler/flagged/count"] == 0
    assert report.metrics["straggler/waiting/count"] == 1
    assert _reasons(report) == [(6, "upstream_wait")]
    # wait/* points at the rank with the largest excess waiting over its stage peers.
    assert report.metrics["straggler/wait/max_rank"] == 6
    assert report.metrics["straggler/wait/max_ms"] == pytest.approx(85.0)
    assert report.metrics["straggler/wait/spread"] == pytest.approx(85.0 / 5.0 - 1.0)


def test_upstream_wait_needs_persist_windows_and_never_feeds_the_culprit_streak():
    config = DetectorConfig(persist_windows=3)
    state = DetectorState()
    table = [_healthy_row() for _ in range(4)] + [_healthy_row(fwd=150.0, bwd=300.0, optim=30.0) for _ in range(4)]
    table[6] = _healthy_row(fwd=150.0, bwd=300.0, optim=30.0, pp_recv=80.0)
    for window in range(3):
        report = analyze_window(table, _meta(8, pp_size=2), config, state)
        if window < 2:
            assert report.alerts == [], "a single noisy window must not raise upstream_wait"
            assert report.metrics["straggler/waiting/count"] == 0
    assert _reasons(report) == [(6, "upstream_wait")]
    assert "for 3 windows" in report.alerts[0].message
    assert report.metrics["straggler/flagged/count"] == 0
    assert state.consecutive == {}, "victim windows must not count towards flagging a culprit"


def test_two_rank_group_uses_leave_one_out_reference():
    table = [_healthy_row(), _healthy_row(fwd=115.0, bwd=230.0)]  # self 365 vs 320 = 14% slower than its only peer
    report = _analyze_once(table, _meta(2), rel_threshold=0.10)
    assert report.metrics["straggler/flagged/rank"] == 1
    assert report.metrics["straggler/self/spread"] == pytest.approx(365.0 / 320.0 - 1.0, abs=1e-6)


def test_wait_below_absolute_guard_is_not_reported():
    table = [_healthy_row() for _ in range(4)]
    table[2] = _healthy_row(pp_recv=2.0)  # peers 0 -> relative excess is huge, but 2 ms of 320 ms is noise
    report = _analyze_once(table, _meta(4), wait_abs_frac=0.05)
    assert report.metrics["straggler/waiting/count"] == 0
    assert report.alerts == []
    assert report.metrics["straggler/pp_recv/max_rank"] == 2
    assert report.metrics["straggler/pp_recv/spread"] == 100.0  # capped


def test_late_arriver_is_the_rank_with_the_shortest_grad_sync():
    # Observed on a real 8xA800 SFT run: rank 0's kernels are as fast as everyone
    # else's, but it reaches the DP reduce-scatter ~1 s late, so the 7 peers show a
    # ~1 s grad-sync bracket while rank 0's bracket is just the 20 ms transfer.
    config = DetectorConfig(persist_windows=2)
    state = DetectorState()
    table = [_healthy_row(dp_grad_sync=1020.0) for _ in range(8)]
    table[0] = _healthy_row(dp_grad_sync=20.0)
    first = analyze_window(table, _meta(8), config, state)
    assert first.metrics["straggler/flagged/count"] == 0  # persistence
    assert first.metrics["straggler/late/max_rank"] == 0
    assert first.metrics["straggler/late/max_ms"] == pytest.approx(1000.0)
    assert first.metrics["straggler/late/peer_idle_ms"] == pytest.approx(1020.0)
    assert first.metrics["straggler/self/spread"] == pytest.approx(0.0)  # compute itself is balanced
    report = analyze_window(table, _meta(8), config, state)
    assert report.metrics["straggler/flagged/count"] == 1
    assert report.metrics["straggler/flagged/rank"] == 0
    assert report.metrics["straggler/flagged/reason"] == REASON_CODE["late_arrival"]
    assert [(alert.rank, alert.reason, alert.new) for alert in report.alerts] == [(0, "late_arrival", True)]
    assert "1000.0 ms/step after its DP peers" in report.alerts[0].message
    assert report.rows[0]["late_ms"] == pytest.approx(1000.0)
    assert report.rows[1]["late_ms"] == pytest.approx(0.0)


def test_late_arrival_is_measured_within_grad_sync_groups_only():
    # Two PP stages; stage 1 has a longer grad-sync for everyone (bigger layers), which
    # must not make stage-0 ranks look "late". Only rank 5 is late within its own group.
    table = [_healthy_row(dp_grad_sync=30.0) for _ in range(4)] + [_healthy_row(dp_grad_sync=400.0) for _ in range(4)]
    table[5] = _healthy_row(dp_grad_sync=40.0)
    report = _analyze_once(table, _meta(8, pp_size=2))
    assert _reasons(report) == [(5, "late_arrival")]
    assert report.metrics["straggler/late/max_rank"] == 5
    assert report.metrics["straggler/late/max_ms"] == pytest.approx(360.0)


def test_late_arrival_groups_split_by_tp_but_not_by_cp():
    # TP=2, DP=4: each TP rank reduce-scatters with the 3 other ranks of the same TP index.
    table = [_healthy_row(dp_grad_sync=300.0) for _ in range(8)]
    table[2] = _healthy_row(dp_grad_sync=20.0)  # tp=0 group: ranks 0,2,4,6
    report = _analyze_once(table, _meta(8, tp_size=2))
    assert _reasons(report) == [(2, "late_arrival")]
    assert report.rows[3]["late_ms"] == 0.0  # tp=1 ranks are not in that collective

    # CP ranks are inside the DP x CP grad-sync collective, so a late rank in one
    # CP slice is still detected against peers from the other slice.
    cp_meta = [RankMeta(rank=r, dp=r // 2, tp=0, pp=0, cp=r % 2) for r in range(8)]
    table = [_healthy_row(dp_grad_sync=300.0) for _ in range(8)]
    table[3] = _healthy_row(dp_grad_sync=20.0)
    assert _reasons(_analyze_once(table, cp_meta)) == [(3, "late_arrival")]


def test_small_or_uniform_grad_sync_is_not_late_arrival():
    uniform = [_healthy_row(dp_grad_sync=500.0) for _ in range(8)]  # slow link, nobody late
    assert _analyze_once(uniform, _meta(8), wait_abs_frac=0.05).alerts == []
    tiny = [_healthy_row(dp_grad_sync=12.0) for _ in range(8)]
    tiny[3] = _healthy_row(dp_grad_sync=4.0)  # 8 ms of a 320 ms step is noise
    report = _analyze_once(tiny, _meta(8), wait_abs_frac=0.05)
    assert report.alerts == []
    assert report.metrics["straggler/late/max_rank"] == 3  # still reported, just not flagged
    assert report.metrics["straggler/flagged/count"] == 0


def test_slow_compute_takes_precedence_over_late_arrival_classification():
    # A rank whose kernels are slow naturally also arrives late; report the root cause.
    table = [_healthy_row(dp_grad_sync=200.0) for _ in range(4)]
    table[1] = _healthy_row(fwd=150.0, bwd=300.0, dp_grad_sync=20.0)
    assert _reasons(_analyze_once(table, _meta(4))) == [(1, "slow_device")]


def test_missing_tokens_disables_token_metrics():
    table = [_row(steps=5, fwd=10.0, bwd=20.0) for _ in range(2)]
    report = analyze_window(table, _meta(2), DetectorConfig(), DetectorState())
    assert "straggler/tokens/median" not in report.metrics
    assert "straggler/self_per_ktok/median_ms" not in report.metrics
    assert report.metrics["straggler/self/median_ms"] == pytest.approx(30.0)


def test_overhead_is_averaged_per_step_and_dropped_events_are_totalled():
    table = [_row(steps=4, fwd=1.0, overhead=0.5, dropped=2.0) for _ in range(3)]
    report = analyze_window(table, _meta(3), DetectorConfig(), DetectorState())
    assert report.metrics["straggler/self_overhead_ms"] == pytest.approx(0.5)
    assert report.metrics["straggler/dropped_events"] == pytest.approx(24.0)


def test_table_and_meta_size_mismatch_raises():
    with pytest.raises(ValueError):
        analyze_window([_healthy_row()], _meta(2), DetectorConfig(), DetectorState())
