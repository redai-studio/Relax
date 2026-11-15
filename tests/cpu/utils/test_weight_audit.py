# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression coverage for unknown provenance and publication-based ages."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from relax.utils.agentic_obs.weight_audit import PublicationLedger, attempt_source, audit_sample


def _publish(ledger, label, epoch="epoch"):
    assert ledger.begin("policy", epoch, label)
    return ledger.commit("policy", epoch, label)


def test_publication_counts_successes_and_rejects_late_old_epoch():
    ledger = PublicationLedger()
    first = _publish(ledger, "10")
    assert first["ordinal"] == 1
    assert ledger.commit("policy", "epoch", "10") == first
    assert ledger.begin("policy", "epoch", "20")
    assert ledger.snapshot()["ordinal"] is None
    # Failed update 20 never commits; label gaps are not policy ages.
    third = _publish(ledger, "90")
    assert third["ordinal"] == 2
    assert ledger.commit("policy", "epoch", "20") is None
    restarted = _publish(ledger, "10", epoch="replacement")
    assert restarted["ordinal"] == 1
    assert not ledger.begin("policy", "epoch", "91")
    assert ledger.snapshot() == restarted
    assert attempt_source("10", first, restarted) is None


def test_partial_attempt_keeps_old_source_while_publication_is_in_progress():
    ledger = PublicationLedger()
    first = _publish(ledger, "10")
    ledger.begin("policy", "epoch", "20")
    assert attempt_source("10", first, ledger.snapshot()) == first
    assert attempt_source(None, first, first) is None
    assert attempt_source(["10", "20"], first, first) is None
    assert attempt_source("10", first, None) is None


def _fixture():
    ledger = PublicationLedger()
    one = _publish(ledger, "10")
    _publish(ledger, "20")
    three = _publish(ledger, "90")
    trajectory = {
        "session_id": "s",
        "lineage": [
            {"kind": "obs", "token_count": 2},
            {
                "kind": "resp",
                "token_count": 4,
                "generation_segments": [{"start": 0, "end": 4, "version": one}],
            },
            {"kind": "obs", "token_count": 3},
            {
                "kind": "resp",
                "token_count": 6,
                "generation_end_version": three,
                "generation_segments": [{"start": 0, "end": 6, "version": three}],
            },
        ],
    }
    sample = {"response_length": 13, "effective_loss_mask": {"ranges": [[0, 4, 1], [7, 13, 1]]}}
    return trajectory, sample, _publish(ledger, "100")


def test_audit_tool_wait_and_post_generation_wait_are_distinct():
    trajectory, sample, consumer = _fixture()
    report = audit_sample(trajectory, sample, consumer)
    assert report["effective_tokens"] == 10
    assert report["generation_span"] == 2
    assert report["post_generation_lag"] == 1
    assert report["token_weighted_age"] == pytest.approx(1.8)
    assert [row["response_start"] for row in report["segments"]] == [0, 7]
    assert [row["consumption_age"] for row in report["segments"]] == [3, 1]
    assert report["version_missing_rate"] == report["age_missing_rate"] == 0


def test_audit_unknown_mask_and_sources_are_never_assigned_current_version():
    trajectory, sample, consumer = _fixture()
    trajectory["lineage"][-1]["generation_segments"][0]["version"] = None
    report = audit_sample(trajectory, sample, consumer)
    assert report["version_missing_rate"] == pytest.approx(0.6)
    assert report["token_weighted_age"] == 3
    assert report["generation_span"] is None
    sample["effective_loss_mask"] = None
    assert audit_sample(trajectory, sample, consumer)["token_weighted_age"] is None
    assert audit_sample(trajectory, sample, consumer)["effective_tokens"] is None


def test_audit_overlapping_intervals_cannot_double_count_known_tokens():
    trajectory, sample, consumer = _fixture()
    segment = trajectory["lineage"][1]["generation_segments"][0]
    trajectory["lineage"][1]["generation_segments"].append(copy.deepcopy(segment))
    report = audit_sample(trajectory, sample, consumer)
    assert report["known_version_tokens"] == 6
    assert report["token_weighted_age"] == 1
    assert report["version_missing_rate"] == pytest.approx(0.4)


def test_audit_mask_excludes_old_prefix_and_uncovered_tokens_remain_unknown():
    trajectory, sample, consumer = _fixture()
    sample["effective_loss_mask"]["ranges"].pop(0)
    report = audit_sample(trajectory, sample, consumer)
    assert report["effective_tokens"] == 6
    assert report["generation_span"] == 0
    assert report["token_weighted_age"] == 1
    report = audit_sample(None, sample, consumer)
    assert report["version_missing_rate"] == 1
    assert report["token_weighted_age"] is None


@pytest.mark.parametrize("ranges", [[[0, 4, 1], [3, 5, 1]], [[0, 14, 1]], [[0, 4, 0]], [[0, 4]]])
def test_audit_invalid_objective_ranges_remain_unknown(ranges):
    trajectory, sample, consumer = _fixture()
    sample["effective_loss_mask"] = {"ranges": ranges}
    report = audit_sample(trajectory, sample, consumer)
    assert report["effective_tokens"] is None
    assert report["token_weighted_age"] is None


def test_audit_intersects_sparse_mask_at_segment_boundaries():
    trajectory, sample, consumer = _fixture()
    sample["effective_loss_mask"] = {"ranges": [[2, 4, 1], [8, 10, 1], [12, 13, 1]]}
    report = audit_sample(trajectory, sample, consumer)
    assert [row["effective_tokens"] for row in report["segments"]] == [2, 3]
    assert report["token_weighted_age"] == pytest.approx(1.8)


def test_empty_audit_directory_cannot_certify_consumption():
    from relax.utils.agentic_obs.weight_report import build_weight_report

    report = build_weight_report([], [], [])
    assert not report["complete"]
    assert report["no_consumption_evidence"]
    assert not report["evidence_available"]


def test_weight_audit_demo_joins_real_session_publication_and_optimizer(tmp_path):
    # The separate process owns one Ray actor and one Gloo rank. Pure reader
    # tests cannot catch a disconnected Session/publication/consumption hook.
    output = tmp_path / "audit.json"
    subprocess.run(
        [sys.executable, "-m", "examples.agentic_weight_audit.demo", "--output", str(output)],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = json.loads(output.read_text())
    assert report["complete"] and report["token_weighted_age"] == pytest.approx(1.8)
    assert report["version_missing_rate"] == report["age_missing_rate"] == 0
    events = report["demo"]["events"]
    waiting = events.index({"event": "tool_wait_started"})
    finished = events.index({"event": "tool_wait_finished"})
    assert [event["version"]["ordinal"] for event in events[waiting + 1 : finished]] == [2, 3]
    root = Path(report["demo"]["records"])
    (train_path,) = (root / "train").glob("*.jsonl")
    train = json.loads(train_path.read_text())
    assert train["schema_version"] == "4.1.0"
    assert train["samples"][0]["sample_export_id"] == report["samples"][0]["sample_export_id"]
    assert train["samples"][0]["effective_loss_mask"] == {"ranges": [[0, 4, 1], [7, 13, 1]]}
    assert report["demo"]["training"]["parameter_update_norm"] > 0
