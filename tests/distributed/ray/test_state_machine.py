# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for ScaleOutRequest / ScaleInRequest state machines, status enums, and
configuration dataclasses."""

import time

import pytest


try:
    from relax.distributed.ray.rollout import (
        EngineGroupConfig,
        ModelConfig,
        ScaleInRequest,
        ScaleInStatus,
        ScaleOutMode,
        ScaleOutRequest,
        ScaleOutStatus,
    )

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


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

    def test_update_status_changes_timestamp(self):
        req = ScaleOutRequest(request_id="x", status=ScaleOutStatus.PENDING)
        old_ts = req.updated_at
        time.sleep(0.01)
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


# ========================== EngineGroupConfig ===============================


class TestEngineGroupConfig:
    @pytest.mark.parametrize("wt", ["regular", "prefill", "decode", "placeholder"])
    def test_valid_worker_types(self, wt):
        cfg = EngineGroupConfig(worker_type=wt, num_gpus=4)
        assert cfg.worker_type == wt

    def test_invalid_worker_type(self):
        with pytest.raises(AssertionError, match="Invalid worker_type"):
            EngineGroupConfig(worker_type="bad", num_gpus=4)

    def test_zero_gpus(self):
        with pytest.raises(AssertionError, match="num_gpus must be > 0"):
            EngineGroupConfig(worker_type="regular", num_gpus=0)

    def test_negative_gpus(self):
        with pytest.raises(AssertionError, match="num_gpus must be > 0"):
            EngineGroupConfig(worker_type="regular", num_gpus=-1)

    def test_optional_fields_defaults(self):
        cfg = EngineGroupConfig(worker_type="regular", num_gpus=4)
        assert cfg.num_gpus_per_engine is None
        assert cfg.overrides == {}

    def test_optional_fields_set(self):
        cfg = EngineGroupConfig(
            worker_type="regular",
            num_gpus=4,
            num_gpus_per_engine=2,
            overrides={"model_path": "/m"},
        )
        assert cfg.num_gpus_per_engine == 2
        assert cfg.overrides == {"model_path": "/m"}


# ============================= ModelConfig ==================================


class TestModelConfig:
    def test_resolve_defaults(self):
        args = type(
            "Args",
            (),
            {
                "rollout_num_gpus_per_engine": 4,
                "hf_checkpoint": "/default/model",
                "sglang_hf_checkpoint": None,
            },
        )()
        cfg = ModelConfig(
            name="actor",
            engine_groups=[
                EngineGroupConfig(worker_type="regular", num_gpus=8),
            ],
        )
        cfg.resolve(args)
        assert cfg.engine_groups[0].num_gpus_per_engine == 4
        assert cfg.engine_groups[0].overrides["model_path"] == "/default/model"

    def test_resolve_per_group_override(self):
        args = type(
            "Args",
            (),
            {
                "rollout_num_gpus_per_engine": 4,
                "hf_checkpoint": "/default/model",
                "sglang_hf_checkpoint": None,
            },
        )()
        cfg = ModelConfig(
            name="actor",
            num_gpus_per_engine=2,
            engine_groups=[
                EngineGroupConfig(worker_type="regular", num_gpus=4, num_gpus_per_engine=8),
            ],
        )
        cfg.resolve(args)
        # Per-group override takes precedence
        assert cfg.engine_groups[0].num_gpus_per_engine == 8

    def test_has_pd_disaggregation(self):
        cfg_no = ModelConfig(
            name="m",
            engine_groups=[EngineGroupConfig(worker_type="regular", num_gpus=4)],
        )
        assert cfg_no.has_pd_disaggregation is False

        cfg_yes = ModelConfig(
            name="m",
            engine_groups=[
                EngineGroupConfig(worker_type="prefill", num_gpus=4),
                EngineGroupConfig(worker_type="decode", num_gpus=4),
            ],
        )
        assert cfg_yes.has_pd_disaggregation is True

    def test_total_num_gpus(self):
        cfg = ModelConfig(
            name="m",
            engine_groups=[
                EngineGroupConfig(worker_type="regular", num_gpus=4),
                EngineGroupConfig(worker_type="regular", num_gpus=8),
            ],
        )
        assert cfg.total_num_gpus == 12
