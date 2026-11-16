# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the autoscaler monitor TUI data layer.

The Textual widgets/rendering need a real terminal and are verified visually on
a cluster; here we cover the pure data-accumulation path that feeds the charts
(``MetricsHistory``), including the engine-count history added for the "engine
count over time" chart.
"""

from relax.utils.autoscaler.monitor import (
    AutoscalerSnapshot,
    MetricsHistory,
    _categorize_scale_error,
    _history_rows,
    _render_chart,
    _status_symbol,
)


def test_metrics_history_tracks_engine_count():
    """MetricsHistory must record the current engine count per point (so the
    TUI can chart how the engine count fluctuates over time), and clear() must
    drop it alongside the other series."""
    h = MetricsHistory(max_points=3)
    h.add_point(throughput=100.0, token_usage=0.1, running_reqs=5, queue_reqs=0, timestamp="t0", engines=1)
    h.add_point(throughput=120.0, token_usage=0.2, running_reqs=8, queue_reqs=2, timestamp="t1", engines=2)
    assert list(h.engines) == [1.0, 2.0]
    # honors max_points ring buffer like the other series
    h.add_point(throughput=90.0, token_usage=0.05, running_reqs=3, queue_reqs=0, timestamp="t2", engines=3)
    h.add_point(throughput=80.0, token_usage=0.04, running_reqs=2, queue_reqs=0, timestamp="t3", engines=1)
    assert list(h.engines) == [2.0, 3.0, 1.0]
    h.clear()
    assert list(h.engines) == []


def test_render_chart_renders_engine_series():
    """The chart renderer produces a titled chart for a small integer engine
    series without error."""
    out = _render_chart(
        [1.0, 2.0, 1.0, 3.0, 2.0],
        width=20,
        height=5,
        color="magenta",
        title="Engines",
        unit="",
        fmt=".0f",
        y_min=0.0,
    )
    assert isinstance(out, str)
    assert "Engines" in out


def _yaxis_labels(chart: str) -> list[str]:
    """Extract the y-axis tick labels (text after the trailing ``¦`` on chart
    body rows) from a rendered chart."""
    labels: list[str] = []
    for line in chart.splitlines():
        if not line.startswith("¦"):
            continue
        parts = line.rsplit("¦", 1)
        if len(parts) == 2 and parts[1].strip():
            labels.append(parts[1].strip())
    return labels


def test_render_chart_integer_yaxis_has_no_fractional_ticks():
    """Engine count is an integer quantity, so with integer_yaxis=True the
    y-axis tick labels must be whole numbers (no 0.2 / 0.8 fractional ticks
    even when the value range is small, e.g. a constant 1 engine with
    y_min=0)."""
    out = _render_chart(
        [1.0, 1.0, 1.0],
        width=20,
        height=6,
        color="magenta",
        title="Engines",
        unit="",
        fmt=".0f",
        y_min=0.0,
        integer_yaxis=True,
    )
    labels = _yaxis_labels(out)
    assert labels, "expected some y-axis tick labels"
    for lbl in labels:
        assert "." not in lbl, f"integer y-axis should not emit a fractional tick: {lbl!r}"


def test_status_symbol_three_states():
    """Status maps to ✓ / ✗ / ○ / - : terminal-success, failure, neutral
    no-op, and in-progress respectively (case-insensitive)."""
    # success terminals
    assert _status_symbol("ACTIVE") == ("✓", "green")
    assert _status_symbol("COMPLETED") == ("✓", "green")
    assert _status_symbol("partial") == ("✓", "green")
    # failure / rejection terminals
    assert _status_symbol("FAILED") == ("✗", "red")
    assert _status_symbol("CANCELLED") == ("✗", "red")
    assert _status_symbol("CONFLICT") == ("✗", "red")
    # neutral no-op (not a failure)
    assert _status_symbol("NOOP") == ("○", "yellow")
    assert _status_symbol("noop") == ("○", "yellow")
    # in-progress (non-terminal)
    assert _status_symbol("PENDING") == ("-", "dim")
    assert _status_symbol("CREATING") == ("-", "dim")
    assert _status_symbol("DRAINING") == ("-", "dim")
    assert _status_symbol("") == ("-", "dim")
    assert _status_symbol(None) == ("-", "dim")


def test_categorize_scale_error_buckets():
    """String-path buckets for sources without structured categories (scale-in,
    autoscaler no-op/conflict).

    Scale-out categories are handled structurally
    (see test_categorize_scale_error_prefers_structured_categories), so no
    NCCL/precheck/transport text guessing remains here.
    """
    assert _categorize_scale_error("request timeout after 300s") == "timeout"
    assert _categorize_scale_error("", status="CONFLICT") == "conflict/rejected"
    assert _categorize_scale_error("request rejected by router") == "conflict/rejected"
    # NOOP is neutral (not conflict/reject), its own bucket
    assert _categorize_scale_error("scale_out NOOP: already at target") == "no scaling needed (no-op)"
    # fallback keeps the raw text (truncated) under "other"
    other = _categorize_scale_error("some brand new unmapped failure")
    assert other.startswith("other")
    assert "unmapped failure" in other
    # empty -> neutral dash
    assert _categorize_scale_error("", "", "") == "-"


def test_categorize_scale_error_prefers_structured_categories():
    """When failure_categories (source-classified) are present, they win over
    error_message string guessing for the disambiguated buckets."""
    # Structured category short-circuits, even with an unhelpful error_message.
    assert (
        _categorize_scale_error("some opaque text", categories=["NCCL_PRECHECK_TRANSPORT_MISMATCH"])
        == "NCCL transport mismatch"
    )
    assert _categorize_scale_error("", categories=["NCCL_PRECHECK_FAIL"]) == "precheck failed"
    assert _categorize_scale_error("", categories=["PROVISION_TIMEOUT"]) == "elastic node provision timeout"
    # Most-specific wins when several categories are present.
    assert (
        _categorize_scale_error("", categories=["PROVISION_TIMEOUT", "NCCL_PRECHECK_TRANSPORT_MISMATCH"])
        == "NCCL transport mismatch"
    )
    # Every source-classified category is bucketed from the enum (never silently
    # dropped to "other"): a structured category wins over the error_message text.
    assert _categorize_scale_error("request rejected", categories=["WEIGHT_SYNC_FAILED"]) == "weight sync failed"
    assert _categorize_scale_error("some text", categories=["HEALTH_CHECK_FAILED"]) == "health check failed"
    # No structured category -> fall back to the string path.
    assert _categorize_scale_error("request rejected", categories=[]) == "conflict/rejected"
    assert _categorize_scale_error("", categories=[]) == "-"


def test_history_rows_merges_pending_and_marks_status():
    """History rows merge completed records (/scale_history) with in-flight
    pending_requests (/status) so a decision shows on *initiation*; each row
    carries the tri-state symbol and failures show the categorized reason."""
    snap = AutoscalerSnapshot(
        history={
            "history": [
                {
                    "request_id": "done1",
                    "action": "scale_out",
                    "status": "ACTIVE",
                    "triggered_at": 100.0,
                    "from_engines": 2,
                    "to_engines": 3,
                    "reason": "high token usage",
                },
                {
                    "request_id": "fail1",
                    "action": "scale_out",
                    "status": "FAILED",
                    "triggered_at": 90.0,
                    "from_engines": 2,
                    "to_engines": 4,
                    "reason": "queue depth",
                    "error_message": "timeout waiting for PG ready",
                },
            ]
        },
        status={
            "pending_requests": [
                {
                    "request_id": "inflight1",
                    "action": "scale_out",
                    "status": "CREATING",
                    "triggered_at": 200.0,
                    "from_engines": 3,
                    "to_engines": 5,
                    "reason": "still provisioning",
                },
                # duplicate request_id already terminal in history -> deduped
                {
                    "request_id": "done1",
                    "action": "scale_out",
                    "status": "PENDING",
                    "triggered_at": 100.0,
                },
            ]
        },
    )
    rows = _history_rows(snap)
    # newest-first: the in-flight one (triggered_at=200) leads
    assert rows[0][0] and rows[0][2] == "scale_out"
    # dedup: done1 appears once (2 history + 1 unique pending = 3 rows)
    assert len(rows) == 3
    # in-flight row shows "-" symbol
    assert "-" in rows[0][1]
    # find the failed row -> ✗ and categorized provision-timeout reason
    fail_rows = [r for r in rows if "✗" in r[1]]
    assert fail_rows, "expected a failed row with ✗"
    assert "provision" in fail_rows[0][4] or "timeout" in fail_rows[0][4]
    # success row -> ✓
    assert any("✓" in r[1] for r in rows)


def test_history_reason_leads_with_category_and_keeps_long_detail():
    """The Reason cell must lead with the classified category (so it survives
    truncation) and keep the detail up to the 300-char cap — not chop it at the
    old 77 so the key info is lost to an ellipsis."""
    from relax.utils.autoscaler.monitor import _HISTORY_REASON_MAXLEN

    long_detail = "NCCL precheck FAIL: nccl_error " + ("x" * 200)
    snap = AutoscalerSnapshot(
        history={
            "history": [
                {
                    "request_id": "f1",
                    "action": "scale_out",
                    "status": "FAILED",
                    "triggered_at": 90.0,
                    "error_message": long_detail,
                    "failure_categories": ["NCCL_PRECHECK_FAIL"],
                }
            ]
        },
        status={"pending_requests": []},
    )
    reason = _history_rows(snap)[0][4]
    # Category leads (never the part lost to truncation).
    assert reason.startswith("precheck failed")
    # Detail preserved well past the old 77-char cap.
    assert len(reason) > 100
    assert "nccl_error" in reason
    # Still bounded (not unbounded / not a wrapped traceback).
    assert len(reason) <= _HISTORY_REASON_MAXLEN
