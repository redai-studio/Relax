# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from relax.agentic.weight_audit import (
    AUDIT_FIELD,
    PublicationLedger,
    attempt_weight_version,
    consumption_snapshot,
    diagnose,
    prepare_audit,
    record_publication,
)
from relax.utils.types import Sample
from relax.utils.utils import convert_samples_to_train_data, dict_to_tensordict


def test_mixed_attempt_is_not_assigned_to_its_last_version():
    assert attempt_weight_version({"weight_version": "12"}) == "12"
    assert attempt_weight_version({"weight_version": "12", "weight_versions": [{"version": "12"}]}) == "12"
    assert (
        attempt_weight_version({"weight_version": "12", "weight_versions": [{"version": "10"}, {"version": "12"}]})
        is None
    )
    assert attempt_weight_version({}) is None


def test_legacy_missing_provenance_counts_effective_tokens_as_unknown():
    sample = Sample(response_length=3, loss_mask=[1, 0, 1])
    report = diagnose(prepare_audit(sample), {"epoch": "run", "sequence": 1})
    assert report["effective_tokens"] == report["version_missing_tokens"] == 2
    assert report["version_missing_token_rate"] == 1


def _fixture():
    ledger = PublicationLedger()
    ledger.record("policy-run", 1, "10")
    first = ledger.snapshot("10")
    ledger.record("policy-run", 2, "11")
    ledger.record("policy-run", 3, "12")
    second = ledger.snapshot("12")
    ledger.record("policy-run", 4, "13")
    audit = {
        "session_id": "session",
        "generation_end": second,
        "spans": [
            {"start": 0, "end": 4, "weight_version": "10", "effective_tokens": 4, "publication": first},
            {"start": 7, "end": 13, "weight_version": "12", "effective_tokens": 6, "publication": second},
        ],
    }
    return ledger, audit


def test_two_publications_during_tool_wait_and_one_after_generation():
    ledger, audit = _fixture()
    report = diagnose(audit, ledger.snapshot())
    assert [s["consumption_age"] for s in report["spans"]] == [3, 1]
    assert [s["generation_age"] for s in report["spans"]] == [2, 0]
    assert report["generation_tail_lag"] == 0
    assert report["generation_version_span"] == 2
    assert report["post_generation_wait_lag"] == 1
    assert report["consumption_age_token_mean"] == 1.8
    assert report["version_missing_token_rate"] == 0
    assert "consumption_age" not in audit["spans"][0]


def test_unknown_versions_do_not_become_zero_or_latest():
    ledger, audit = _fixture()
    audit["spans"][0]["publication"] = {}
    report = diagnose(audit, ledger.snapshot())
    assert report["version_missing_token_rate"] == 0.4
    assert report["age_missing_token_rate"] == 0.4
    assert report["consumption_age_token_mean"] == 1
    assert report["spans"][0]["consumption_age"] is None
    audit["spans"][1]["publication"] = {}
    report = diagnose(audit, ledger.snapshot())
    assert report["generation_version_span"] is None
    assert report["consumption_age_token_mean"] is None
    assert report["version_missing_token_rate"] == 1


def test_generation_span_survives_missing_consumption_snapshot():
    _, audit = _fixture()
    report = diagnose(audit, {})
    assert report["generation_version_span"] == 2
    assert report["post_generation_wait_lag"] is None
    assert report["consumption_age_token_mean"] is None


def test_zero_effective_tokens_and_future_versions_are_not_valid_ages():
    ledger, audit = _fixture()
    future = diagnose(audit, {"epoch": "policy-run", "sequence": 0})
    assert future["consumption_age_token_mean"] is None
    assert future["post_generation_wait_lag"] is None
    for span in audit["spans"]:
        span["effective_tokens"] = 0
    report = diagnose(audit, ledger.snapshot())
    assert report["version_missing_token_rate"] is None
    assert report["consumption_age_token_mean"] is None


def test_ledger_is_instance_local_idempotent_and_snapshots_are_detached():
    first, second = PublicationLedger(), PublicationLedger()
    first.record("run", 1, "raw")
    first.record("run", 1, "raw")
    result = first.snapshot("raw")
    result["sequence"] = 99
    assert first.snapshot("raw")["sequence"] == 1
    assert second.snapshot("raw")["sequence"] is None


@pytest.mark.parametrize("epoch,seq,raw", [("other-writer", 1, "1"), ("run", 2, "1"), ("run", 3, "3")])
def test_conflicts_or_missing_publications_invalidate_old_frontier(epoch, seq, raw):
    ledger = PublicationLedger()
    ledger.record("run", 1, "1")
    try:
        ledger.record(epoch, seq, raw)
    except ValueError:
        pass
    assert ledger.snapshot("1") == {"epoch": None, "sequence": None, "source_sequence": None}


async def test_coordinator_snapshot_and_publish_operations_are_atomic():
    from relax.agentic.session.admission_coordinator import AdmissionCoordinator

    cls = AdmissionCoordinator.__ray_metadata__.modified_class
    coordinator = cls.__new__(cls)
    coordinator._publications = PublicationLedger()
    coordinator._active_request_permits = {"first", "second"}
    coordinator._request_permits = asyncio.Semaphore(0)
    await cls.record_publication(coordinator, "run", 1, "1")
    results = await asyncio.gather(
        cls.release_request_permit(coordinator, "first", audit_version="1", audit=True),
        cls.record_publication(coordinator, "run", 2, "2"),
        cls.release_request_permit(coordinator, "second", audit_version="2", audit=True),
        cls.record_publication(coordinator, "run", 2, "2"),
    )
    assert results[0]["sequence"] == 1
    assert results[2]["sequence"] == 2
    assert results[2]["source_sequence"] == 2
    assert await cls.release_request_permit(coordinator, "first", audit_version="1", audit=True) is None
    assert coordinator._request_permits._value == 2


def test_lost_publication_rpc_cannot_report_false_zero_age(monkeypatch):
    from ray import serve

    handle = MagicMock()
    handle.record_publication.remote.return_value.result.side_effect = TimeoutError("lost audit RPC")
    monkeypatch.setattr(serve, "get_app_handle", lambda _: handle)
    owner = SimpleNamespace(args=SimpleNamespace(use_agentic_rollout=True), weight_version=1)
    record_publication(owner)
    assert consumption_snapshot(owner) == {"audit_valid": False}
    ledger, audit = _fixture()
    report = diagnose(audit, consumption_snapshot(owner))
    assert report["consumption_age_token_mean"] is None
    assert report["version_missing_token_rate"] == 1
    handle.record_publication.remote.return_value.result.side_effect = None
    owner.weight_version = 2
    record_publication(owner)
    assert owner._agentic_publication[1] == 2
    assert consumption_snapshot(owner) == {"audit_valid": False}
    handle.record_publication.remote.assert_called_once()


async def test_publication_from_dcs_event_loop_waits_for_real_serve_response(monkeypatch):
    from concurrent.futures import Future

    from ray import serve
    from ray.serve._private.common import RequestMetadata
    from ray.serve.handle import DeploymentResponse

    ledger = PublicationLedger()

    def remote(epoch, sequence, version):
        def get(timeout_s):
            assert 0 < timeout_s <= 5
            ledger.record(epoch, sequence, version)

        future = Future()
        future.set_result(SimpleNamespace(get=get))
        return DeploymentResponse(future, RequestMetadata(request_id="audit", internal_request_id="audit"))

    handle = SimpleNamespace(record_publication=SimpleNamespace(remote=remote))
    monkeypatch.setattr(serve, "get_app_handle", lambda _: handle)
    owner = SimpleNamespace(args=SimpleNamespace(use_agentic_rollout=True), weight_version=1)
    record_publication(owner)
    snapshot = consumption_snapshot(owner)
    assert snapshot["audit_valid"] is True
    assert snapshot["sequence"] == ledger.snapshot()["sequence"] == 1
    assert snapshot["epoch"] == ledger.snapshot()["epoch"]


def test_non_agentic_and_adapter_only_publications_do_not_add_rpcs(monkeypatch):
    from ray import serve

    get_handle = MagicMock()
    monkeypatch.setattr(serve, "get_app_handle", get_handle)
    for owner in (
        SimpleNamespace(),
        SimpleNamespace(args=SimpleNamespace(use_agentic_rollout=False)),
        SimpleNamespace(args=SimpleNamespace(use_agentic_rollout=True), lora_adapter_mode=True),
        SimpleNamespace(args=SimpleNamespace(use_agentic_rollout=True), _lora_adapter_mode=True),
    ):
        record_publication(owner)
        assert consumption_snapshot(owner) == {"audit_valid": False}
    get_handle.assert_not_called()


def test_final_mask_counts_and_tensordict_roundtrip():
    _, audit = _fixture()
    sample = Sample(
        session_id="session",
        response_length=13,
        loss_mask=[0] * 4 + [0] * 3 + [1] * 6,
        metadata={
            "agentic_trace": {
                "weight_version_spans": audit["spans"],
                "generation_end_publication": audit["generation_end"],
            }
        },
    )
    row = prepare_audit(sample)
    assert [s["effective_tokens"] for s in row["spans"]] == [0, 6]
    td = dict_to_tensordict({AUDIT_FIELD: [None, row]}, 2)
    assert list(td[AUDIT_FIELD]) == [None, row]
    assert list(td[[1, 0]][AUDIT_FIELD]) == [row, None]


@pytest.mark.parametrize("remove_sample", [False, True])
def test_audit_does_not_change_existing_training_fields_or_group_identity(remove_sample):
    _, audit = _fixture()
    sample = Sample(
        index=7,
        group_index=2,
        tokens=[1] * 14,
        response_length=13,
        reward=1.0,
        loss_mask=[1] * 4 + [0] * 3 + [1] * 6,
        remove_sample=remove_sample,
        metadata={"agentic_trace": {"weight_version_spans": audit["spans"]}},
    )
    args = SimpleNamespace(
        use_agentic_rollout=False,
        custom_reward_post_process_path=None,
        reward_key=None,
        rewards_normalization=False,
        multimodal_keys=None,
        use_opd=False,
        debug_train_only=True,
    )
    # Two physical exports may share a logical sample index; neither is deduplicated.
    samples = [sample, copy.deepcopy(sample)]
    before = convert_samples_to_train_data(args, copy.deepcopy(samples))
    args.use_agentic_rollout = True
    after = convert_samples_to_train_data(args, samples)
    reports = after.pop(AUDIT_FIELD)
    assert before == after
    assert len(reports) == 2
    assert [sum(s["effective_tokens"] for s in row["spans"]) for row in reports] == (
        [0, 0] if remove_sample else [10, 10]
    )


def test_custom_converter_preserves_rows_and_supplies_unknown_provenance(monkeypatch):
    from relax.utils import utils

    custom = dict_to_tensordict({"tokens": [[1, 2], [3, 4]]}, 2)
    monkeypatch.setattr(utils, "load_function", lambda _: lambda *args: custom)
    args = SimpleNamespace(custom_convert_samples_to_train_data_path="custom", use_agentic_rollout=True)
    result = convert_samples_to_train_data(args, [Sample()])
    assert result is custom
    assert result.batch_size[0] == 2
    assert list(result[AUDIT_FIELD]) == [None, None]
