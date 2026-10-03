# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for speculative decoding rollout metrics."""

from types import SimpleNamespace

import pytest

from relax.utils.metrics.metric_utils import compute_spec_decoding_metrics
from relax.utils.types import Sample, get_spec_counter_values


def _counters(accept: int, propose: int, verify: int = 0, completion: int = 0) -> dict[str, int]:
    return {
        "spec_accept_token_num": accept,
        "spec_draft_token_num": propose,
        "spec_verify_ct": verify,
        "completion_token_num": completion,
    }


def _sample(*, nodes: dict[str, dict[str, int]] | None = None, **totals) -> Sample:
    sample = Sample(session_id="sess", index=0)
    sample.spec_info.nodes = dict(nodes or {})
    for key, value in totals.items():
        setattr(sample.spec_info, key, value)
    return sample


def test_spec_metrics_weight_counters_instead_of_averaging_rates():
    samples = [
        _sample(nodes={"req_sess-a_0": _counters(1, 2)}),
        _sample(nodes={"req_sess-a_1": _counters(9, 10)}),
    ]

    metrics = compute_spec_decoding_metrics(samples)

    assert metrics["spec_accept_rate"] == pytest.approx(10 / 12)
    # Averaging the two per-sample rates would have reported 0.7 instead.
    assert metrics["spec_accept_rate"] != pytest.approx(0.7)
    assert metrics["spec_nodes_total"] == 2.0
    assert metrics["spec_coverage"] == 1.0


def test_spec_metrics_count_node_shared_by_several_samples_once():
    shared = _counters(4, 8, verify=2, completion=6)
    samples = [
        _sample(nodes={"req_sess-a_0": shared, "req_sess-a_1": _counters(1, 2, verify=1, completion=2)}),
        _sample(nodes={"req_sess-a_0": shared, "req_sess-a_2": _counters(1, 1, verify=1, completion=1)}),
    ]

    metrics = compute_spec_decoding_metrics(samples)

    assert metrics["spec_nodes_total"] == 3.0
    assert metrics["spec_accept_rate"] == pytest.approx(6 / 11)
    assert metrics["spec_accept_length"] == pytest.approx(9 / 4)


def test_spec_metrics_keep_identical_independent_requests_separate():
    counters = _counters(1, 2, verify=1, completion=2)
    samples = [
        _sample(nodes={"req_sess-a_0": counters}),
        _sample(nodes={"req_sess-a_1": counters}),
    ]

    metrics = compute_spec_decoding_metrics(samples)

    assert metrics["spec_nodes_total"] == 2.0
    assert metrics["spec_accept_rate"] == pytest.approx(0.5)


def test_spec_metrics_do_not_share_counters_across_sessions():
    samples = [
        _sample(nodes={"req_sess-a_0": _counters(1, 2, verify=1, completion=2)}),
        _sample(nodes={"req_sess-b_0": _counters(3, 4, verify=1, completion=4)}),
    ]

    metrics = compute_spec_decoding_metrics(samples)

    assert metrics["spec_nodes_total"] == 2.0
    # 1/2 and 3/4 stay two nodes: 4/6, not 3.5/4 nor either single node alone.
    assert metrics["spec_accept_rate"] == pytest.approx(4 / 6)


def test_spec_metrics_pair_each_ratio_with_its_own_denominator():
    samples = [
        _sample(
            nodes={
                # Completion tokens without verify steps must not inflate the accept length.
                "req_sess-a_0": _counters(0, 0, verify=0, completion=5),
                "req_sess-a_1": _counters(3, 4, verify=2, completion=4),
            }
        )
    ]

    metrics = compute_spec_decoding_metrics(samples)

    assert metrics["spec_accept_rate"] == pytest.approx(3 / 4)
    assert metrics["spec_accept_length"] == pytest.approx(4 / 2)


def test_spec_metrics_report_unreported_nodes_separately():
    samples = [
        _sample(nodes={"req_sess-a_0": _counters(1, 2), "req_sess-a_1": {}}),
    ]

    metrics = compute_spec_decoding_metrics(samples)

    assert metrics["spec_nodes_total"] == 2.0
    assert metrics["spec_nodes_missing_counts"] == 1.0
    assert metrics["spec_coverage"] == 0.5
    assert metrics["spec_accept_rate"] == pytest.approx(0.5)


def test_spec_metrics_keep_explicit_zero_counters_covered():
    samples = [_sample(nodes={"req_sess-a_0": _counters(0, 0, verify=0, completion=0)})]

    metrics = compute_spec_decoding_metrics(samples)

    assert metrics["spec_coverage"] == 1.0
    assert metrics["spec_nodes_missing_counts"] == 0.0
    # A zero denominator reports no rate instead of a fabricated zero.
    assert "spec_accept_rate" not in metrics
    assert "spec_accept_length" not in metrics


def test_spec_metrics_keep_known_zero_numerators_with_positive_denominators():
    metrics = compute_spec_decoding_metrics([_sample(nodes={"req_s_0": _counters(0, 2, verify=1, completion=0)})])

    assert metrics["spec_accept_rate"] == 0.0
    assert metrics["spec_accept_length"] == 0.0
    assert metrics["spec_coverage"] == 1.0


def test_spec_metrics_empty_batch_has_no_metrics():
    assert compute_spec_decoding_metrics([]) == {}


@pytest.mark.parametrize("payload", [{}, {"spec_info": None}])
def test_spec_metrics_treat_absent_or_null_spec_info_as_unknown(payload):
    sample = Sample.from_dict({"status": "completed", **payload})
    metrics = compute_spec_decoding_metrics([sample])

    assert "spec_accept_rate" not in metrics
    assert "spec_accept_length" not in metrics
    assert metrics["spec_nodes_missing_counts"] == 1.0
    assert metrics["spec_coverage"] == 0.0


def test_spec_metrics_report_nodes_without_any_counters():
    metrics = compute_spec_decoding_metrics([_sample(), _sample()])

    assert metrics["spec_nodes_total"] == 2.0
    assert metrics["spec_nodes_missing_counts"] == 2.0
    assert metrics["spec_coverage"] == 0.0
    assert "spec_accept_rate" not in metrics


def test_spec_metrics_use_totals_when_a_sample_has_no_nodes():
    samples = [_sample(spec_accept_token_num=1, spec_draft_token_num=2, counts_reported=True)]

    metrics = compute_spec_decoding_metrics(samples)

    assert metrics["spec_accept_rate"] == pytest.approx(0.5)
    assert metrics["spec_coverage"] == 1.0
    assert "spec_legacy_samples" not in metrics


def _legacy_sample(**spec_info) -> Sample:
    """A sample deserialized from a dump written before this change."""
    return Sample.from_dict({"status": "completed", "spec_info": spec_info})


def test_spec_metrics_mark_samples_whose_totals_predate_node_identities():
    samples = [_legacy_sample(**_counters(3, 4, verify=2, completion=5))]

    metrics = compute_spec_decoding_metrics(samples)

    assert samples[0].spec_info.counts_reported is None
    assert metrics["spec_accept_rate"] == pytest.approx(0.75)
    assert metrics["spec_accept_length"] == pytest.approx(2.5)
    assert metrics["spec_legacy_samples"] == 1.0


def test_spec_metrics_do_not_fabricate_zero_rate_from_legacy_completion_only_dump():
    # Every legacy dump carries ``completion_token_num``; without a draft or verify
    # counter it proves no speculative report and must not become a 0% rate.
    samples = [_legacy_sample(**_counters(0, 0, verify=0, completion=7))]

    metrics = compute_spec_decoding_metrics(samples)

    assert "spec_accept_rate" not in metrics
    assert metrics["spec_legacy_samples"] == 1.0
    assert metrics["spec_nodes_missing_counts"] == 1.0
    assert metrics["spec_coverage"] == 0.0


def test_spec_metrics_treat_legacy_payload_with_missing_counter_as_unreported():
    # ``accept`` is missing, so the dump cannot prove a zero numerator.
    payload = _counters(0, 4, verify=2, completion=5)
    del payload["spec_accept_token_num"]
    samples = [_legacy_sample(**payload)]

    metrics = compute_spec_decoding_metrics(samples)

    assert samples[0].spec_info.counts_reported is None
    assert "spec_accept_rate" not in metrics
    assert metrics["spec_coverage"] == 0.0
    assert metrics["spec_accept_length"] == pytest.approx(2.5)
    assert metrics["spec_accept_rate_coverage"] == 0.0
    assert metrics["spec_accept_length_coverage"] == 1.0


def test_spec_metrics_mix_agentic_and_legacy_samples_in_one_batch():
    samples = [
        _sample(nodes={"req_sess-a_0": _counters(1, 2, verify=1, completion=2)}),
        _legacy_sample(**_counters(9, 10, verify=2, completion=11)),
    ]

    metrics = compute_spec_decoding_metrics(samples)

    assert metrics["spec_accept_rate"] == pytest.approx(10 / 12)
    assert metrics["spec_accept_length"] == pytest.approx(13 / 3)
    assert metrics["spec_nodes_total"] == 2.0
    assert metrics["spec_legacy_samples"] == 1.0


def test_spec_metrics_keep_reported_zero_totals_covered():
    samples = [_sample(spec_accept_token_num=0, spec_draft_token_num=0, counts_reported=True)]

    metrics = compute_spec_decoding_metrics(samples)

    assert metrics["spec_coverage"] == 1.0
    assert "spec_accept_rate" not in metrics


def test_spec_info_serialization_round_trips_nodes_and_report_flag():
    info = Sample.SpecInfo.from_dict(
        {
            "spec_accept_token_num": 5,
            "spec_draft_token_num": 6,
            "spec_verify_ct": 7,
            "completion_token_num": 8,
            "nodes": {"req_sess-a_0": _counters(1, 2), "req_sess-a_1": {}},
            "counts_reported": True,
        }
    )

    assert info.nodes == {"req_sess-a_0": _counters(1, 2), "req_sess-a_1": {}}
    assert info.counts_reported is True
    assert Sample.SpecInfo.from_dict(info.to_dict()).to_dict() == info.to_dict()


def test_spec_info_deserializes_dumps_written_before_node_identities():
    info = Sample.SpecInfo.from_dict(_counters(3, 4, verify=1, completion=4))

    assert info.nodes == {}
    assert info.counts_reported is None
    assert info.spec_accept_rate == pytest.approx(0.75)
    # The legacy marker survives another serialization round trip.
    assert Sample.SpecInfo.from_dict(info.to_dict()).counts_reported is None


def test_spec_info_deserializes_null_legacy_counters_without_crashing():
    info = Sample.SpecInfo.from_dict({**_counters(0, 4, verify=1, completion=4), "spec_accept_token_num": None})

    assert info.spec_accept_token_num == 0
    assert "spec_accept_token_num" not in info.available_counters


def test_spec_info_add_distinguishes_explicit_zero_from_missing_report():
    reported = Sample.SpecInfo()
    reported.add({"spec_accept_token_num": 0, "spec_draft_token_num": 0})
    missing = Sample.SpecInfo()
    missing.add({"completion_tokens": 1})

    assert reported.counts_reported is True
    assert missing.counts_reported is False


@pytest.mark.parametrize("missing_first", [True, False])
def test_spec_info_add_preserves_missing_fields_across_resumed_attempts(missing_first):
    info = Sample.SpecInfo()
    full = {"spec_num_correct_drafts": 2, "spec_num_proposed_drafts": 2, "spec_verify_ct": 1, "completion_tokens": 2}
    attempts = [{"completion_tokens": 5}, full] if missing_first else [full, {"completion_tokens": 5}]
    for attempt in attempts:
        info.add(attempt)

    assert info.to_dict()["completion_token_num"] == 7
    assert info.spec_verify_ct == 1
    assert info.available_counters == ["completion_token_num"]
    metrics = compute_spec_decoding_metrics([Sample(spec_info=info)])
    assert "spec_accept_rate" not in metrics
    assert "spec_accept_length" not in metrics
    assert metrics["spec_coverage"] == 0.0


@pytest.mark.parametrize("payload", [None, {"status": "pending"}, {"status": "pending", "spec_info": {}}])
def test_spec_info_fresh_serialization_does_not_hide_the_first_backend_report(payload):
    import json

    sample = Sample.from_dict(json.loads(json.dumps(Sample().to_dict())) if payload is None else payload)
    sample.spec_info.add(
        {"spec_num_correct_drafts": 1, "spec_num_proposed_drafts": 2, "spec_verify_ct": 1, "completion_tokens": 3}
    )
    metrics = compute_spec_decoding_metrics([sample])

    assert metrics["spec_accept_rate"] == pytest.approx(0.5)
    assert metrics["spec_accept_length"] == pytest.approx(3.0)
    assert metrics["spec_coverage"] == 1.0


def test_spec_info_unreported_attempt_stays_unknown_after_serialization_and_resume():
    info = Sample.SpecInfo()
    info.add({})
    info = Sample.SpecInfo.from_dict(info.to_dict())
    info.add(
        {"spec_num_correct_drafts": 1, "spec_num_proposed_drafts": 2, "spec_verify_ct": 1, "completion_tokens": 3}
    )
    metrics = compute_spec_decoding_metrics([Sample(spec_info=info)])

    assert "spec_accept_rate" not in metrics
    assert "spec_accept_length" not in metrics
    assert metrics["spec_coverage"] == 0.0


def test_spec_metrics_reach_rollout_log_only_with_speculative_decoding():
    pytest.importorskip("megatron.core")

    from relax.distributed.ray.rollout import _compute_spec_metrics

    samples = [
        _sample(nodes={"req_sess-a_0": _counters(1, 2)}),
        _sample(nodes={"req_sess-a_1": _counters(9, 10)}),
    ]

    enabled = _compute_spec_metrics(SimpleNamespace(sglang_speculative_algorithm="EAGLE"), samples)

    assert enabled["spec_accept_rate"] == pytest.approx(10 / 12)
    assert _compute_spec_metrics(SimpleNamespace(sglang_speculative_algorithm=None), samples) == {}


def test_spec_metrics_report_matches_hand_computed_totals():
    """Human-checkable report: ``A -> B`` and ``A -> C`` share submitted node
    A.

    | node | accepted | proposed | verify | completion |
    | ---- | -------- | -------- | ------ | ---------- |
    | A    | 4        | 8        | 2      | 6          |
    | B    | 1        | 2        | 1      | 2          |
    | C    | 1        | 1        | 1      | 1          |
    | sum  | 6        | 11       | 4      | 9          |

    Accept rate = 6 / 11, accept length = 9 / 4, and node A is counted once even
    though both exported samples contain it.
    """
    shared = _counters(4, 8, verify=2, completion=6)
    samples = [
        _sample(nodes={"req_sess-a_0": shared, "req_sess-a_1": _counters(1, 2, verify=1, completion=2)}),
        _sample(nodes={"req_sess-a_0": shared, "req_sess-a_2": _counters(1, 1, verify=1, completion=1)}),
    ]

    metrics = compute_spec_decoding_metrics(samples)

    assert metrics == {
        "spec_accept_rate": pytest.approx(6 / 11),
        "spec_accept_length": pytest.approx(9 / 4),
        "spec_nodes_total": 3.0,
        "spec_nodes_missing_counts": 0.0,
        "spec_coverage": 1.0,
        "spec_accept_rate_coverage": 1.0,
        "spec_accept_length_coverage": 1.0,
    }


@pytest.mark.parametrize("missing_key", ["spec_accept_token_num", "completion_token_num"])
@pytest.mark.parametrize("null_value", [False, True])
def test_spec_metrics_do_not_replace_missing_node_numerators_with_zero(missing_key, null_value):
    record = _counters(1, 2, verify=1, completion=3)
    if null_value:
        record[missing_key] = None
    else:
        del record[missing_key]
    info = Sample.SpecInfo.from_dict({"nodes": {"req_s_0": record}})
    metrics = compute_spec_decoding_metrics([Sample(spec_info=info)])

    missing_ratio = "spec_accept_rate" if missing_key == "spec_accept_token_num" else "spec_accept_length"
    assert missing_ratio not in metrics
    assert metrics[missing_ratio + "_coverage"] == 0.0
    assert metrics["spec_coverage"] == 0.0
    assert metrics["spec_nodes_missing_counts"] == 1.0


@pytest.mark.parametrize("missing_key", ["spec_accept_token_num", "completion_token_num"])
@pytest.mark.parametrize("null_value", [False, True])
def test_spec_metrics_preserve_partial_legacy_availability_through_json(missing_key, null_value):
    import json

    record = _counters(1, 2, verify=1, completion=3)
    if null_value:
        record[missing_key] = None
    else:
        del record[missing_key]
    sample = _legacy_sample(**record)
    expected = compute_spec_decoding_metrics([sample])
    for _ in range(2):
        sample = Sample.from_dict(json.loads(json.dumps(sample.to_dict())))
        assert compute_spec_decoding_metrics([sample]) == expected
    missing_ratio = "spec_accept_rate" if missing_key == "spec_accept_token_num" else "spec_accept_length"
    assert missing_ratio not in expected
    assert expected["spec_coverage"] == 0.0


@pytest.mark.parametrize(
    "meta,expected",
    [
        ({"spec_num_correct_drafts": None, "spec_num_proposed_drafts": 2}, {"spec_draft_token_num": 2}),
        (
            {"spec_accepted_drafts": 0, "spec_proposed_drafts": 2},
            {"spec_accept_token_num": 0, "spec_draft_token_num": 2},
        ),
        (
            {"spec_accept_token_num": 1, "spec_draft_token_num": 2},
            {"spec_accept_token_num": 1, "spec_draft_token_num": 2},
        ),
        ({"spec_verify_ct": 2, "completion_tokens": 6}, {"spec_verify_ct": 2, "completion_token_num": 6}),
        ({"spec_verify_ct": "bad", "completion_tokens": None}, {}),
        ({"spec_verify_ct": -1, "completion_tokens": 1.5}, {}),
    ],
)
def test_spec_metadata_normalizes_aliases_without_inventing_counters(meta, expected):
    assert get_spec_counter_values(meta) == expected


def test_spec_metrics_merge_partial_duplicate_records_without_adding_shared_work():
    samples = [
        _sample(nodes={"req_s_0": {"spec_accept_token_num": 1}}),
        _sample(nodes={"req_s_0": {"spec_draft_token_num": 2, "spec_verify_ct": 1, "completion_token_num": 3}}),
    ]
    snapshots = [sample.to_dict() for sample in samples]
    metrics = compute_spec_decoding_metrics(samples)

    assert metrics == compute_spec_decoding_metrics(list(reversed(samples)))
    assert metrics["spec_nodes_total"] == 1.0
    assert metrics["spec_accept_rate"] == pytest.approx(0.5)
    assert metrics["spec_accept_length"] == pytest.approx(3.0)
    assert metrics["spec_coverage"] == 1.0
    assert [sample.to_dict() for sample in samples] == snapshots


def test_spec_metrics_weight_each_ratio_by_its_own_complete_counter_pairs():
    samples = [
        _sample(
            nodes={
                "req_s_0": _counters(1, 2, verify=1, completion=3),
                "req_s_1": {"spec_accept_token_num": 9, "spec_draft_token_num": 10},
                "req_s_2": {"spec_verify_ct": 2, "completion_token_num": 6},
            }
        )
    ]
    metrics = compute_spec_decoding_metrics(samples)

    assert metrics["spec_accept_rate"] == pytest.approx(10 / 12)
    assert metrics["spec_accept_length"] == pytest.approx(9 / 3)
    assert metrics["spec_coverage"] == pytest.approx(1 / 3)
    assert metrics["spec_accept_rate_coverage"] == pytest.approx(2 / 3)
    assert metrics["spec_accept_length_coverage"] == pytest.approx(2 / 3)
