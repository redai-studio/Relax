# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU tests for scale request states and their serialized API responses."""

import pickle
from types import SimpleNamespace

import pytest

from relax.distributed.ray import rollout_scale_types
from relax.distributed.ray.rollout_scale_types import (
    ScaleInRequest,
    ScaleInStatus,
    ScaleOutMode,
    ScaleOutRequest,
    ScaleOutStatus,
)


# ============================== ScaleOutStatus ==============================


class TestScaleOutStatus:
    def test_all_values_present(self):
        expected = {
            "PENDING",
            "CREATING",
            "CONNECTING",
            "HEALTH_CHECKING",
            "WEIGHT_SYNCING",
            "READY",
            "ACTIVE",
            "PARTIAL",
            "FAILED",
            "REMOVING",
            "CANCELLED",
        }
        assert {s.value for s in ScaleOutStatus} == expected

    def test_string_enum(self):
        assert ScaleOutStatus.PENDING == "PENDING"
        assert isinstance(ScaleOutStatus.PENDING, str)


class TestScaleOutMode:
    def test_modes(self):
        assert ScaleOutMode.RAY_NATIVE == "ray_native"
        assert ScaleOutMode.EXTERNAL == "external"


# ============================== ScaleOutRequest ==============================


class TestScaleOutRequest:
    def test_auto_generated_uuid(self):
        req = ScaleOutRequest(request_id="", status=ScaleOutStatus.PENDING)
        assert len(req.request_id) > 0
        assert req.request_id != ""

    def test_auto_generated_timestamps(self):
        req = ScaleOutRequest(request_id="x", status=ScaleOutStatus.PENDING)
        assert req.created_at > 0
        assert req.updated_at == req.created_at

    def test_explicit_fields(self):
        req = ScaleOutRequest(
            request_id="my-id",
            status=ScaleOutStatus.PENDING,
            model_name="actor",
            num_replicas=3,
            engine_urls=["http://a:1"],
            timeout_secs=300.0,
        )
        assert req.request_id == "my-id"
        assert req.model_name == "actor"
        assert req.num_replicas == 3
        assert req.engine_urls == ["http://a:1"]
        assert req.timeout_secs == 300.0

    def test_default_collections(self):
        req = ScaleOutRequest(request_id="x", status=ScaleOutStatus.PENDING)
        assert req.engine_urls == []
        assert req.engine_ids == []
        assert req.failed_engines == []

    def test_update_status_changes_timestamp(self, monkeypatch):
        monkeypatch.setattr(rollout_scale_types, "time", SimpleNamespace(time=lambda: 10.0))
        req = ScaleOutRequest(request_id="x", status=ScaleOutStatus.PENDING)
        old_ts = req.updated_at
        monkeypatch.setattr(rollout_scale_types, "time", SimpleNamespace(time=lambda: 20.0))
        req.update_status(ScaleOutStatus.CREATING)
        assert req.status == ScaleOutStatus.CREATING
        assert req.updated_at > old_ts
        assert req.error_message is None

    def test_update_status_with_error_message(self):
        req = ScaleOutRequest(request_id="x", status=ScaleOutStatus.PENDING)
        req.update_status(ScaleOutStatus.FAILED, "boom")
        assert req.status == ScaleOutStatus.FAILED
        assert req.error_message == "boom"

    @pytest.mark.parametrize(
        "status_str,expected",
        [
            ("PENDING", False),
            ("CREATING", False),
            ("CONNECTING", False),
            ("HEALTH_CHECKING", False),
            ("WEIGHT_SYNCING", False),
            ("READY", False),
            ("REMOVING", False),
            ("ACTIVE", True),
            ("PARTIAL", True),
            ("FAILED", True),
            ("CANCELLED", True),
        ],
    )
    def test_is_terminal(self, status_str, expected):
        req = ScaleOutRequest(request_id="x", status=ScaleOutStatus(status_str))
        assert req.is_terminal() is expected

    @pytest.mark.parametrize(
        "status_str,expected",
        [
            ("PENDING", True),
            ("CREATING", True),
            ("CONNECTING", False),
            ("HEALTH_CHECKING", False),
            ("WEIGHT_SYNCING", False),
            ("ACTIVE", False),
            ("FAILED", False),
            ("CANCELLED", False),
        ],
    )
    def test_can_cancel(self, status_str, expected):
        req = ScaleOutRequest(request_id="x", status=ScaleOutStatus(status_str))
        assert req.can_cancel() is expected

    def test_to_dict_keys(self):
        req = ScaleOutRequest(request_id="abc", status=ScaleOutStatus.ACTIVE)
        d = req.to_dict()
        expected_keys = {
            "request_id",
            "status",
            "model_name",
            "num_replicas",
            "engine_urls",
            "engine_ids",
            "failed_engines",
            "created_at",
            "updated_at",
            "error_message",
            "weight_version",
            "failure_categories",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self):
        req = ScaleOutRequest(
            request_id="abc",
            status=ScaleOutStatus.ACTIVE,
            model_name="reward",
            num_replicas=2,
            engine_urls=["u1"],
            engine_ids=["e1"],
            failed_engines=["e2"],
            weight_version="v3",
        )
        d = req.to_dict()
        assert d["request_id"] == "abc"
        assert d["status"] == "ACTIVE"
        assert d["model_name"] == "reward"
        assert d["num_replicas"] == 2
        assert d["engine_urls"] == ["u1"]
        assert d["engine_ids"] == ["e1"]
        assert d["failed_engines"] == ["e2"]
        assert d["weight_version"] == "v3"


# ============================== ScaleInStatus ===============================


class TestScaleInStatus:
    def test_all_values_present(self):
        expected = {"PENDING", "DRAINING", "REMOVING", "COMPLETED", "FAILED"}
        assert {s.value for s in ScaleInStatus} == expected


# ============================== ScaleInRequest ==============================


class TestScaleInRequest:
    def test_auto_generated_uuid(self):
        req = ScaleInRequest(request_id="", status=ScaleInStatus.PENDING)
        assert len(req.request_id) > 0

    def test_auto_generated_timestamps(self):
        req = ScaleInRequest(request_id="x", status=ScaleInStatus.PENDING)
        assert req.created_at > 0
        assert req.updated_at == req.created_at

    def test_update_status(self):
        req = ScaleInRequest(request_id="x", status=ScaleInStatus.PENDING)
        req.update_status(ScaleInStatus.DRAINING)
        assert req.status == ScaleInStatus.DRAINING
        assert req.error_message is None

    def test_update_status_with_error(self):
        req = ScaleInRequest(request_id="x", status=ScaleInStatus.PENDING)
        req.update_status(ScaleInStatus.FAILED, "oops")
        assert req.error_message == "oops"

    @pytest.mark.parametrize(
        "status_str,expected",
        [
            ("PENDING", False),
            ("DRAINING", False),
            ("REMOVING", False),
            ("COMPLETED", True),
            ("FAILED", True),
        ],
    )
    def test_is_terminal(self, status_str, expected):
        req = ScaleInRequest(request_id="x", status=ScaleInStatus(status_str))
        assert req.is_terminal() is expected

    def test_to_dict_keys(self):
        req = ScaleInRequest(request_id="x", status=ScaleInStatus.COMPLETED)
        d = req.to_dict()
        expected_keys = {
            "request_id",
            "status",
            "model_name",
            "num_replicas",
            "engine_urls",
            "timeout_secs",
            "force",
            "dry_run",
            "created_at",
            "updated_at",
            "selected_engines",
            "removed_engines",
            "failed_engines",
            "error_message",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self):
        req = ScaleInRequest(
            request_id="xyz",
            status=ScaleInStatus.COMPLETED,
            model_name="default",
            num_replicas=2,
            removed_engines=["e1"],
            force=True,
            dry_run=False,
        )
        d = req.to_dict()
        assert d["request_id"] == "xyz"
        assert d["status"] == "COMPLETED"
        assert d["removed_engines"] == ["e1"]
        assert d["force"] is True
        assert d["dry_run"] is False

    def test_default_collections(self):
        req = ScaleInRequest(request_id="x", status=ScaleInStatus.PENDING)
        assert req.engine_urls == []
        assert req.selected_engines == []
        assert req.removed_engines == []
        assert req.failed_engines == []


@pytest.mark.parametrize(
    "request_cls, status",
    [(ScaleOutRequest, ScaleOutStatus.PENDING), (ScaleInRequest, ScaleInStatus.PENDING)],
)
def test_scale_request_serialization_roundtrip_preserves_fields(request_cls, status):
    request = request_cls(
        request_id="roundtrip",
        status=status,
        created_at=10.0,
        updated_at=12.0,
        timeout_secs=17.0,
        engine_urls=["http://engine.test:8000"],
        error_message="earlier failure",
    )
    restored = pickle.loads(pickle.dumps(request))
    assert type(restored) is request_cls
    assert restored == request
    assert restored.to_dict() == request.to_dict()
    assert restored.timeout_secs == 17.0


@pytest.mark.parametrize(
    "request_cls, status",
    [(ScaleOutRequest, ScaleOutStatus.PENDING), (ScaleInRequest, ScaleInStatus.PENDING)],
)
def test_scale_request_mutable_defaults_are_not_shared(request_cls, status):
    first = request_cls(request_id="first", status=status)
    second = request_cls(request_id="second", status=status)
    first.engine_urls.append("http://engine.test:8000")
    first.failed_engines.append("failed")
    assert second.engine_urls == []
    assert second.failed_engines == []


@pytest.mark.parametrize("empty_error", [None, ""])
@pytest.mark.parametrize(
    "request_cls, initial, updated",
    [
        (ScaleOutRequest, ScaleOutStatus.FAILED, ScaleOutStatus.CREATING),
        (ScaleInRequest, ScaleInStatus.FAILED, ScaleInStatus.DRAINING),
    ],
)
def test_scale_request_updates_preserve_existing_error_and_timestamp_contract(
    monkeypatch, request_cls, initial, updated, empty_error
):
    request = request_cls(request_id="request", status=initial, created_at=10.0, error_message="original")
    monkeypatch.setattr(rollout_scale_types, "time", SimpleNamespace(time=lambda: 30.0))
    request.update_status(updated, empty_error)
    assert request.status is updated
    assert request.created_at == 10.0
    assert request.updated_at == 30.0
    assert request.error_message == "original"
