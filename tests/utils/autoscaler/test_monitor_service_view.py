# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the autoscaler monitor's service selection (Task 4).

``--service genrm`` must show the GenRM runtime's own state: the client sends
``service=`` query params where the backend filters (/conditions,
/scale_history), projects the aggregated /status payload onto the selected
service client-side, resolves per-service policy overrides onto the config
view, and renders ``unknown`` (never zeros, never the rollout alias) when the
selected service is not configured.
"""

import asyncio

import pytest

from relax.utils.autoscaler.monitor import (
    AutoscalerApiClient,
    AutoscalerSnapshot,
    _build_metrics_rows,
    _render_status_panel,
    _select_service_view,
    build_arg_parser,
)


_STATUS = {
    "enabled": True,
    "running": True,
    "current_engines": 4,
    "min_engines": 4,
    "max_engines": 8,
    "pending_requests": [{"request_id": "r1"}],
    "recent_metrics": {"num_engines": 4, "total_throughput": 100.0},
    "total_scale_operations": 7,
    "config": {
        "evaluation_interval_secs": 5.0,
        "scale_out_policy": {"token_usage_threshold": 0.8, "queue_depth_per_engine": 10},
        "scale_in_policy": {"token_usage_threshold": 0.05},
        "service_policies": {
            "genrm": {
                "scale_out_policy": {"token_usage_threshold": 0.3},
                "scale_in_policy": {"token_usage_threshold": 0.02},
            }
        },
    },
    "services": {
        "rollout": {"current_engines": 4},
        "genrm": {
            "current_engines": 2,
            "min_engines": 1,
            "max_engines": 2,
            "pending_requests": [],
            "recent_metrics": {"num_engines": 2, "total_throughput": 55.5},
            "total_scale_operations": 3,
        },
    },
}


def test_genrm_view_overlays_service_fields():
    view = _select_service_view(_STATUS, "genrm")
    assert view["current_engines"] == 2
    assert view["min_engines"] == 1
    assert view["max_engines"] == 2
    assert view["pending_requests"] == []
    assert view["recent_metrics"]["total_throughput"] == 55.5
    assert view["total_scale_operations"] == 3
    # Global fields stay visible.
    assert view["enabled"] is True and view["running"] is True


def test_genrm_view_resolves_policy_overrides():
    view = _select_service_view(_STATUS, "genrm")
    policy = view["config"]["scale_out_policy"]
    assert policy["token_usage_threshold"] == 0.3  # genrm override
    assert policy["queue_depth_per_engine"] == 10  # inherited global
    assert view["config"]["scale_in_policy"]["token_usage_threshold"] == 0.02


def test_rollout_view_is_passthrough():
    view = _select_service_view(_STATUS, "rollout")
    assert view is _STATUS


def test_missing_service_is_flagged_not_zeroed():
    status = dict(_STATUS)
    status["services"] = {"rollout": {"current_engines": 4}}
    view = _select_service_view(status, "genrm")
    assert view["selected_service_missing"] == "genrm"
    rows = _build_metrics_rows(AutoscalerSnapshot(status=view), None)
    by_label = {r.label: r for r in rows}
    assert "unknown" in by_label["Engines"].current
    assert "unknown" in by_label["  Min/Max"].current
    assert "unknown" in by_label["Throughput"].current
    # The rollout alias' numbers must not leak into the genrm view.
    assert "4" not in by_label["Engines"].current
    assert "100" not in by_label["Throughput"].current


def test_status_panel_shows_missing_service_banner():
    view = _select_service_view(dict(_STATUS, services={}), "genrm")
    text = _render_status_panel(AutoscalerSnapshot(status=view), None)
    assert "not configured" in text


def test_metrics_rows_use_selected_service_numbers():
    view = _select_service_view(_STATUS, "genrm")
    rows = _build_metrics_rows(AutoscalerSnapshot(status=view), None)
    by_label = {r.label: r for r in rows}
    assert "2" in by_label["Engines"].current  # genrm engines, not rollout's 4


def test_fetch_all_sends_service_query_params():
    client = AutoscalerApiClient("http://127.0.0.1:8000/autoscaler_genrm", service="genrm")
    paths = []

    async def fake_get(path):
        paths.append(path)
        return {}

    client._get_json = fake_get
    asyncio.run(client.fetch_all())
    assert paths == [
        "/status",
        "/conditions?service=genrm",
        "/scale_history?limit=10&service=genrm",
    ]


def test_fetch_all_rollout_keeps_default_params():
    client = AutoscalerApiClient("http://127.0.0.1:8000/autoscaler", service="rollout")
    paths = []

    async def fake_get(path):
        paths.append(path)
        return {}

    client._get_json = fake_get
    asyncio.run(client.fetch_all())
    assert paths == [
        "/status",
        "/conditions?service=rollout",
        "/scale_history?limit=10&service=rollout",
    ]


def test_arg_parser_rejects_unknown_service():
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--service", "critic"])
    args = build_arg_parser().parse_args(["--service", "genrm"])
    assert args.service == "genrm"
    assert build_arg_parser().parse_args([]).service == "rollout"
