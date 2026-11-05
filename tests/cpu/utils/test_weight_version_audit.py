# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest

from relax.utils import weight_version_audit as audit


@pytest.fixture
def publication(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAX_WEIGHT_AUDIT_DIR", str(tmp_path))
    publisher = audit.PublicationLedger(str(tmp_path), "policy-main")
    tags = [publisher.publish() for _ in range(12)]
    return publisher, tags


def prepared(publication, *, missing=False, mask=None):
    publisher, tags = publication
    first = audit.generation_spans(
        {"weight_versions": [{"start": 0, "end": 3, "version": tags[9]}]},
        3,
        offset=4,
        request_id="request-0",
        attempt_id="request-0:0",
        piece_index=0,
    )
    second = audit.generation_spans(
        {} if missing else {"weight_versions": [{"start": 0, "end": 2, "version": tags[11]}]},
        2,
        offset=9,
        request_id="request-1",
        attempt_id="request-1:0",
        piece_index=0,
    )
    provenance = {
        "schema_version": 1,
        "sample_uid": "session/leaf/export-0",
        "session_id": "session",
        "leaf_state_hash": "leaf",
        "spans": first + second,
        "finish_snapshot": audit.read_frontier(),
    }
    return audit.prepare_row(provenance, 11, 7, mask if mask is not None else [1, 1, 1, 0, 0, 1, 1])


def test_two_update_tool_wait_hand_calculated(publication):
    row = prepared(publication)
    publication[0].publish()
    report = audit.audit_row(row, audit.read_frontier())
    assert report["eligible_tokens"] == report["known_version_tokens"] == 5
    assert report["observed_generation_span"] == 2
    assert report["oldest_token_age"] == 3
    assert report["newest_token_age"] == 1
    assert report["token_weighted_age"] == 2.2
    assert report["missing_token_rate"] == 0
    assert report["post_generation_published_advance"] == 1
    assert report["finish_frontier_gap"] == 0
    assert report["finish_boundary"] == "session_response_commit"


def test_missing_tokens_remain_in_denominator(publication):
    row = prepared(publication, missing=True)
    publication[0].publish()
    report = audit.audit_row(row, audit.read_frontier())
    assert report["eligible_tokens"] == 5
    assert report["missing_token_rate"] == 0.4
    assert report["token_weighted_age"] == 3
    assert report["age_coverage"] == 0.6
    assert report["observed_generation_span"] == 0
    assert report["generation_span_complete"] is False


@pytest.mark.parametrize("mask", [[0] * 7, [1, 1, 1, 0, 0, 1, 0]])
def test_final_mask_controls_exposure(publication, mask):
    row = prepared(publication, mask=mask)
    publication[0].publish()
    report = audit.audit_row(row, audit.read_frontier())
    assert report["eligible_tokens"] == sum(mask)
    assert report["token_weighted_age"] == (2.5 if sum(mask) else None)


def test_uncovered_tokens_are_unknown(publication):
    row = audit.prepare_row({"spans": []}, 5, 2, [1, 1])
    report = audit.audit_row(row, audit.read_frontier())
    assert report["missing_token_rate"] == 1
    assert report["token_weighted_age"] is None
    assert report["observed_generation_span"] is None
    assert report["unknown_reasons"] == {"uncovered_token_range": 2}


def test_missing_consumption_frontier_preserves_known_generation(publication):
    report = audit.audit_row(prepared(publication), None)
    assert report["version_coverage"] == 1
    assert report["observed_generation_span"] == 2
    assert report["age_coverage"] == 0
    assert report["token_weighted_age"] is None


def test_post_generation_advance_is_known_when_source_versions_are_missing(publication):
    row = audit.prepare_row({"spans": [], "finish_snapshot": audit.read_frontier()}, 5, 2, [1, 1])
    publication[0].publish()
    report = audit.audit_row(row, audit.read_frontier())
    assert report["missing_token_rate"] == 1
    assert report["observed_generation_span"] is None
    assert report["post_generation_published_advance"] == 1
    assert report["finish_frontier_gap"] is None


def test_restart_lineage_is_not_compared(publication):
    row = prepared(publication)
    old = audit.read_frontier()
    replacement = {**old, "lineage_id": "new-run"}
    report = audit.audit_row(row, replacement)
    assert report["missing_token_rate"] == 1
    assert report["unknown_reasons"] == {"publication_scope_mismatch": 5}


def test_missing_consumer_frontier_does_not_mix_generation_lineages(publication):
    row = prepared(publication)
    row["spans"][1].update(lineage_id="different-run", published_seq=1)
    report = audit.audit_row(row, None)
    assert report["observed_generation_span"] is None
    assert report["known_version_tokens"] == 0
    assert report["unknown_reasons"] == {"mixed_publication_lineages": 5}


def test_negative_age_is_visible_and_not_clamped(publication):
    row = prepared(publication)
    report = audit.audit_row(row, {**audit.read_frontier(), "published_seq": 11})
    assert report["negative_age_tokens"] == 2
    assert report["age_coverage"] == 0.6
    assert report["token_weighted_age"] == 1


def test_scalar_version_is_not_token_evidence(publication):
    spans = audit.generation_spans(
        {"weight_version": publication[1][9]},
        3,
        offset=4,
        request_id="req",
        attempt_id="req:0",
        piece_index=0,
    )
    assert spans[0]["published_seq"] is None
    assert spans[0]["unknown_reason"] == "missing_backend_version_intervals"


def test_controlled_abort_contract_accepts_confirmed_scalar(publication):
    spans = audit.generation_spans(
        {"weight_version": publication[1][9], "_weight_audit_scalar_bound": True},
        3,
        offset=4,
        request_id="req",
        attempt_id="req:0",
        piece_index=0,
    )
    assert spans[0]["published_seq"] == 10
    assert spans[0]["unknown_reason"] is None


@pytest.mark.parametrize("raw", ["0", "100", "checkpoint-abc", None])
def test_raw_tags_are_not_successful_publication_sequences(publication, raw):
    spans = audit.generation_spans(
        {"weight_versions": [{"start": 0, "end": 1, "version": raw}]},
        1,
        offset=4,
        request_id="req",
        attempt_id="req:0",
        piece_index=0,
    )
    assert spans[0]["published_seq"] is None


def test_one_backend_call_can_cross_versions(publication):
    spans = audit.generation_spans(
        {
            "weight_versions": [
                {"start": 0, "end": 2, "version": publication[1][9]},
                {"start": 2, "end": 3, "version": publication[1][11]},
            ]
        },
        3,
        offset=7,
        request_id="req",
        attempt_id="req:1",
        piece_index=1,
    )
    assert [(s["token_start"], s["token_end"], s["published_seq"]) for s in spans] == [(7, 9, 10), (9, 10, 12)]
    assert len({s["segment_id"] for s in spans}) == 2


@pytest.mark.parametrize(
    "intervals",
    [
        [{"start": 0, "end": 4}],
        [{"start": 0, "end": 2}, {"start": 1, "end": 3}],
        [{"start": -1, "end": 2}],
    ],
)
def test_invalid_backend_intervals_are_not_trusted(publication, intervals):
    spans = audit.generation_spans(
        {"weight_versions": intervals},
        3,
        offset=4,
        request_id="req",
        attempt_id="req:0",
        piece_index=0,
    )
    assert all(s["published_seq"] is None for s in spans)
    assert spans[0]["token_start"] == 4 and spans[-1]["token_end"] == 7


@pytest.mark.parametrize("mask", [[1], [1, 2], [-1, 1]])
def test_invalid_mask_is_rejected(mask):
    with pytest.raises(ValueError):
        audit.prepare_row({}, 5, 2, mask)


def test_nonuniform_batch_aggregates_counts(publication):
    row = prepared(publication)
    publication[0].publish()
    first = audit.audit_row(row, audit.read_frontier())
    second = audit.audit_row(
        audit.prepare_row(
            {
                "spans": [
                    {
                        **row["spans"][-1],
                        "token_start": 4,
                        "token_end": 5,
                    }
                ]
            },
            5,
            1,
            [1],
        ),
        audit.read_frontier(),
    )
    result = audit.aggregate([first, second])
    assert result["eligible_tokens"] == 6
    assert result["token_weighted_age"] == 2


def test_callback_idempotence_real_reuse_and_dummy(publication, tmp_path):
    row = prepared(publication)
    publication[0].publish()
    step = audit.AuditStep("consume-0")
    step.observe(0, [row])
    step.observe(0, [row])
    step.observe(1, [row], dummy=True)
    step.observe(2, [row])
    result = step.commit(0)
    assert result["eligible_tokens"] == 10
    assert step.commit(0) == result
    second = audit.AuditStep("consume-1")
    second.observe(0, [row])
    second.commit(0)
    summary = json.loads(next(tmp_path.glob("*/consumed-rank-0.summary.json")).read_text())
    assert summary["eligible_tokens"] == 15
    assert summary["token_weighted_age"] == 2.2
    assert summary["consumptions"] == 2


def test_report_limit_does_not_sample_aggregate(publication, tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_MAX_REPORT_BYTES", 1)
    step = audit.AuditStep("consume")
    step.observe(0, [prepared(publication)])
    assert step.commit(0)["detail_records_dropped"] == 1
    summary = json.loads(next(tmp_path.glob("*/consumed-rank-0.summary.json")).read_text())
    assert summary["eligible_tokens"] == 5
    assert summary["generation_span_max"] == 2
    assert summary["oldest_token_age"] == 2


def test_interrupted_schedule_has_no_committed_report(publication, tmp_path):
    step = audit.AuditStep("interrupted")
    step.observe(0, [prepared(publication)])
    assert not list(tmp_path.glob("*/consumed*.jsonl"))


def test_payload_round_trip_keeps_unicode():
    value = {"schema_version": 1, "session_id": "会话-一", "spans": []}
    assert json.loads(bytes(audit.encode_payload(value)).decode("utf-8")) == value


def test_writer_failure_does_not_reuse_a_loaded_policy_sequence(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("occupied")
    publisher = audit.PublicationLedger(str(blocker), "policy")
    with pytest.raises(OSError):
        publisher.publish()
    publisher.directory = tmp_path
    assert publisher.publish().endswith(":2")


def test_disabled_does_not_read_or_write(monkeypatch):
    monkeypatch.delenv("RELAX_WEIGHT_AUDIT_DIR", raising=False)
    assert not audit.enabled()
    assert audit.read_frontier() is None
    assert audit.generation_spans({}, 0, offset=0, request_id="r", attempt_id="r:0", piece_index=0) == []


@pytest.mark.parametrize("value", [[], None, "bad", {}, {"schema_version": 1, "published_seq": True}])
def test_invalid_frontier_is_unknown_not_a_training_error(tmp_path, monkeypatch, value):
    monkeypatch.setenv("RELAX_WEIGHT_AUDIT_DIR", str(tmp_path))
    (tmp_path / "frontier.json").write_text(json.dumps(value))
    assert audit.read_frontier() is None
    assert audit.AuditStep("consume").frontier is None
