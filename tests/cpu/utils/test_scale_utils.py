# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the scale-out failure classification helpers."""

from relax.utils.scale_utils import (
    PrecheckProbeCategory,
    ScaleOutFailure,
    ScaleOutFailureCategory,
    _aggregate_scale_out_reasons,
    _scale_out_failure_categories,
)


def test_message_label_only_for_known_categories_drops_detail():
    # Known (non-precheck) categories must never surface raw exception detail.
    f = ScaleOutFailure(ScaleOutFailureCategory.ENGINE_INIT_FAILED, "boom host=192.0.2.1 sk-secret")
    assert f.message() == "engine init/provision failed"
    assert "192.0.2.1" not in f.message()
    assert "sk-secret" not in f.message()


def test_message_surfaces_safe_precheck_token():
    f = ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_TRANSPORT_MISMATCH, "wrong_type")
    assert f.message() == "NCCL transport mismatch: wrong_type"


def test_message_surfaces_readable_precheck_first_line():
    f = ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_FAIL, "DistBackendError: wrong type 3 != 4")
    # Readable first line surfaced verbatim so the user sees the real NCCL error.
    assert f.message() == "NCCL precheck FAIL: DistBackendError: wrong type 3 != 4"


def test_message_precheck_first_line_only_no_traceback():
    f = ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_FAIL, "boom\nTraceback (most recent call last):\n  ...")
    assert f.message() == "NCCL precheck FAIL: boom"
    assert "Traceback" not in f.message()


def test_message_unknown_keeps_first_line_truncated():
    f = ScaleOutFailure(ScaleOutFailureCategory.UNKNOWN, "line one\nTraceback...\nsecret")
    assert f.message() == "line one"
    assert "Traceback" not in f.message()


def test_str_and_format_delegate_to_message():
    f = ScaleOutFailure(ScaleOutFailureCategory.WEIGHT_SYNC_FAILED, "only 1/2 synced")
    assert str(f) == "weight sync failed"
    assert f"{f}" == "weight sync failed"
    assert format(f) == "weight sync failed"


def test_aggregate_dedupes_by_category():
    reasons = [
        ScaleOutFailure(ScaleOutFailureCategory.WEIGHT_SYNC_FAILED, "a"),
        ScaleOutFailure(ScaleOutFailureCategory.WEIGHT_SYNC_FAILED, "b"),
        ScaleOutFailure(ScaleOutFailureCategory.HEALTH_CHECK_FAILED),
    ]
    assert _aggregate_scale_out_reasons(reasons) == "weight sync failed; health check failed"


def test_aggregate_caps_items_and_scrubs():
    secret = "sk-supersecret-token-abcdef0123456789"
    traceback_blob = "Traceback (most recent call last):\n  File x\n" + ("A" * 4000)
    reasons = [
        ScaleOutFailure(ScaleOutFailureCategory.ENGINE_INIT_FAILED, f"{traceback_blob} {secret}"),
        ScaleOutFailure(ScaleOutFailureCategory.WEIGHT_SYNC_FAILED),
        ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_TRANSPORT_MISMATCH, "wrong_type"),
        ScaleOutFailure(ScaleOutFailureCategory.HEALTH_CHECK_FAILED),
        ScaleOutFailure(ScaleOutFailureCategory.DCS_REGISTRATION_FAILED),
        ScaleOutFailure(ScaleOutFailureCategory.ROUTER_REGISTRATION_FAILED),
    ]
    out = _aggregate_scale_out_reasons(reasons)
    assert "Traceback" not in out
    assert secret not in out
    assert "AAAA" not in out
    assert "…and 3 more" in out  # 6 distinct categories, cap 3
    assert "weight sync failed" in out


def test_scale_out_failure_categories_returns_deduped_names():
    reasons = [
        ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_TRANSPORT_MISMATCH, "wrong_type"),
        ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_TRANSPORT_MISMATCH, "other"),
        ScaleOutFailure(ScaleOutFailureCategory.WEIGHT_SYNC_FAILED),
    ]
    assert _scale_out_failure_categories(reasons) == [
        "NCCL_PRECHECK_TRANSPORT_MISMATCH",
        "WEIGHT_SYNC_FAILED",
    ]


def test_aggregate_and_categories_truncate_by_priority_not_arrival():
    """When more distinct failures exist than the cap, truncation keeps the
    highest-priority root cause (transport mismatch) even if it arrived last --
    it must never be silently dropped behind lower-priority provisioning
    noise."""
    # 4 distinct categories, cap 3; transport mismatch arrives LAST.
    reasons = [
        ScaleOutFailure(ScaleOutFailureCategory.ENGINE_INIT_FAILED),
        ScaleOutFailure(ScaleOutFailureCategory.DCS_REGISTRATION_FAILED),
        ScaleOutFailure(ScaleOutFailureCategory.ROUTER_REGISTRATION_FAILED),
        ScaleOutFailure(ScaleOutFailureCategory.NCCL_PRECHECK_TRANSPORT_MISMATCH, "wrong_type"),
    ]
    names = _scale_out_failure_categories(reasons, max_items=3)
    assert names[0] == "NCCL_PRECHECK_TRANSPORT_MISMATCH"
    assert "NCCL_PRECHECK_TRANSPORT_MISMATCH" in names
    out = _aggregate_scale_out_reasons(reasons, max_items=3)
    assert "NCCL transport mismatch" in out
    assert "…and 1 more" in out


def test_empty_inputs():
    assert _aggregate_scale_out_reasons(None) == ""
    assert _aggregate_scale_out_reasons([]) == ""
    assert _scale_out_failure_categories(None) == []


# ---------------------------------------------------------------------------
# PrecheckProbeCategory: the structured probe wire protocol.
# ---------------------------------------------------------------------------


def test_probe_category_from_wire_roundtrips_and_defaults():
    assert PrecheckProbeCategory.from_wire("probe_failed") is PrecheckProbeCategory.PROBE_FAILED
    assert PrecheckProbeCategory.from_wire("launch_transient") is PrecheckProbeCategory.LAUNCH_TRANSIENT
    # Unknown / None wire values never guess: they collapse to INVALID_RESULT.
    assert PrecheckProbeCategory.from_wire(None) is PrecheckProbeCategory.INVALID_RESULT
    assert PrecheckProbeCategory.from_wire("some_new_value") is PrecheckProbeCategory.INVALID_RESULT


def test_probe_category_semantics():
    # Only a probe that never produced output is retried; a probe that ran and
    # failed is deterministic and fails closed.
    assert PrecheckProbeCategory.LAUNCH_TRANSIENT.retryable
    assert not PrecheckProbeCategory.PROBE_FAILED.retryable
    assert not PrecheckProbeCategory.TIMEOUT.retryable
    # Topology pre-checks skip (let the real multi-node sync handle it).
    assert PrecheckProbeCategory.UNSUPPORTED_TOPOLOGY.is_skip
    assert PrecheckProbeCategory.UNSUPPORTED_NODE_RANK.is_skip
    assert not PrecheckProbeCategory.PROBE_FAILED.is_skip
