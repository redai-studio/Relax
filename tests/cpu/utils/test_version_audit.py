# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the weight version audit offline tool."""

from __future__ import annotations

import json

import pytest

from relax.utils.agentic_obs.version_audit import (
    compute_run_summary,
    compute_sample_stats,
    main,
)


def _span(version, start, end, attempt=0):
    return {"version": version, "token_start": start, "token_end": end, "attempt_seq": attempt}


class TestSampleStats:
    def test_known_versions_full_metrics(self):
        # Two generation turns: v1 produced 2 tokens, v3 produced 2; consumed at v7.
        spans = [_span("1", 0, 2), _span("3", 2, 4)]
        stats = compute_sample_stats(spans, consumed_weight_version="7")
        assert stats["oldest_version"] == "1"
        assert stats["newest_version"] == "3"
        assert stats["generation_span"] == 2  # 3 - 1
        assert stats["oldest_lag"] == 6  # 7 - 1
        assert stats["newest_lag"] == 4  # 7 - 3
        assert stats["token_weighted_lag"] == pytest.approx(5.0)  # (6*2+4*2)/4
        assert stats["post_generation_lag"] == 4  # == newest_lag
        assert stats["known_token_count"] == 4
        assert stats["unknown_token_count"] == 0
        assert stats["effective_token_count"] == 4
        assert stats["span_missing_rate"] == 0.0
        assert stats["token_missing_rate"] == 0.0

    def test_unknown_versions_excluded_from_lag_means(self):
        # Half the tokens are unknown: lags cover known tokens only, the gap is a missing rate.
        spans = [_span("1", 0, 2), _span(None, 2, 4)]
        stats = compute_sample_stats(spans, consumed_weight_version="3")
        assert stats["oldest_version"] == "1"
        assert stats["token_weighted_lag"] == 2.0  # (2*2)/2
        assert stats["known_token_count"] == 2
        assert stats["unknown_token_count"] == 2
        assert stats["span_missing_rate"] == 0.5
        assert stats["token_missing_rate"] == pytest.approx(0.5)

    def test_all_unknown_reports_null_not_zero(self):
        stats = compute_sample_stats([_span(None, 0, 3)], consumed_weight_version="5")
        assert stats["oldest_version"] is None
        assert stats["newest_version"] is None
        assert stats["generation_span"] is None
        assert stats["oldest_lag"] is None
        assert stats["newest_lag"] is None
        assert stats["token_weighted_lag"] is None
        assert stats["post_generation_lag"] is None
        assert stats["known_token_count"] == 0
        assert stats["unknown_token_count"] == 3
        assert stats["span_missing_rate"] == 1.0

    def test_non_numeric_version_treated_as_unknown(self):
        # A non-numeric version string is unknown (counted as missing), never guessed.
        spans = [_span("v1", 0, 2)]
        stats = compute_sample_stats(spans, consumed_weight_version="5")
        assert stats["oldest_version"] is None
        assert stats["known_token_count"] == 0
        assert stats["unknown_token_count"] == 2
        assert stats["span_missing_rate"] == 1.0

    def test_missing_consumed_version_lags_null(self):
        # No consumption stamp: the generation span is still computable, every lag is None.
        stats = compute_sample_stats([_span("1", 0, 2), _span("3", 2, 4)], consumed_weight_version=None)
        assert stats["generation_span"] == 2
        assert stats["oldest_lag"] is None
        assert stats["token_weighted_lag"] is None

    def test_empty_spans(self):
        stats = compute_sample_stats([], consumed_weight_version="5")
        assert stats["oldest_version"] is None
        assert stats["span_missing_rate"] is None  # no spans, no denominator

    def test_zero_token_known_span_no_division_crash(self):
        # A known span with token_start == token_end has a zero lag denominator: None, not a crash.
        stats = compute_sample_stats([{"version": "5", "token_start": 3, "token_end": 3}], consumed_weight_version="7")
        assert stats["oldest_lag"] == 2
        assert stats["token_weighted_lag"] is None

    def test_loss_mask_limits_token_weighting(self):
        # The mask keeps response tokens [0,2) only; the v9 span is entirely masked
        # out, so it must not pull the token-weighted lag towards its value.
        spans = [_span("1", 0, 2), _span("9", 2, 4)]
        stats = compute_sample_stats(spans, consumed_weight_version="7", effective_ranges=[(0, 2)])
        assert stats["known_token_count"] == 2
        assert stats["effective_token_count"] == 2
        assert stats["token_weighted_lag"] == 6.0  # (7-1)*2/2
        assert stats["oldest_lag"] == 6  # version lags still cover every known span
        assert stats["newest_lag"] == -2

    def test_all_zero_mask_reports_no_token_weighting(self):
        # A sample training on none of its tokens contributes no weighted lag.
        spans = [_span("1", 0, 4)]
        stats = compute_sample_stats(spans, consumed_weight_version="7", effective_ranges=[])
        assert stats["effective_token_count"] == 0
        assert stats["token_weighted_lag"] is None
        assert stats["token_missing_rate"] is None
        assert stats["oldest_lag"] == 6

    def test_missing_mask_counts_every_token(self):
        # No mask evidence (older schema) must not be read as an all-zero mask.
        stats = compute_sample_stats([_span("1", 0, 3)], consumed_weight_version="7", effective_ranges=None)
        assert stats["effective_token_count"] == 3
        assert stats["token_weighted_lag"] == 6.0


class TestRunSummary:
    def test_run_level_aggregates_exclude_unknown_from_means(self):
        rows = [
            {  # all known
                "oldest_lag": 6,
                "newest_lag": 4,
                "token_weighted_lag": 5.0,
                "generation_span": 2,
                "known_token_count": 4,
                "unknown_token_count": 0,
                "effective_token_count": 4,
                "spans_count": 2,
                "unknown_spans": 0,
            },
            {  # consumption stamp missing -> lags None, excluded from the means
                "oldest_lag": None,
                "newest_lag": None,
                "token_weighted_lag": None,
                "generation_span": 2,
                "known_token_count": 4,
                "unknown_token_count": 0,
                "effective_token_count": 4,
                "spans_count": 2,
                "unknown_spans": 0,
            },
            {  # fully unknown -> lags None
                "oldest_lag": None,
                "newest_lag": None,
                "token_weighted_lag": None,
                "generation_span": None,
                "known_token_count": 0,
                "unknown_token_count": 3,
                "effective_token_count": 3,
                "spans_count": 1,
                "unknown_spans": 1,
            },
        ]
        summary = compute_run_summary(rows)
        assert summary["sample_count"] == 3
        assert summary["lag_sample_count"] == 1  # only the first enters the means
        assert summary["oldest_lag_mean"] == 6.0
        assert summary["newest_lag_mean"] == 4.0
        assert summary["token_weighted_lag_mean"] == 5.0
        assert summary["generation_span_mean"] == 2.0  # first two have known versions
        assert summary["effective_token_count"] == 11
        assert summary["zero_effective_sample_count"] == 0
        assert summary["span_missing_rate"] == pytest.approx(1 / 5)
        assert summary["token_missing_rate"] == pytest.approx(3 / 11)
        assert summary["sample_missing_rate"] == pytest.approx(1 / 3)

    def test_zero_effective_samples_are_counted_apart(self):
        rows = [
            {
                "oldest_lag": 2,
                "newest_lag": 2,
                "token_weighted_lag": None,
                "generation_span": 0,
                "known_token_count": 0,
                "unknown_token_count": 0,
                "effective_token_count": 0,
                "spans_count": 1,
                "unknown_spans": 0,
            }
        ]
        summary = compute_run_summary(rows)
        assert summary["token_weighted_lag_mean"] is None
        assert summary["zero_effective_sample_count"] == 1
        assert summary["token_missing_rate"] is None

    def test_spanless_export_is_unmeasured_not_zero_effective(self):
        # An export predating span capture has no spans: it is unmeasured, not
        # a sample that trained on nothing.
        rows = [
            {
                "oldest_lag": None,
                "newest_lag": None,
                "token_weighted_lag": None,
                "generation_span": None,
                "known_token_count": 0,
                "unknown_token_count": 0,
                "effective_token_count": 0,
                "spans_count": 0,
                "unknown_spans": 0,
            }
        ]
        summary = compute_run_summary(rows)
        assert summary["zero_effective_sample_count"] == 0
        assert summary["span_missing_rate"] is None
        assert summary["sample_missing_rate"] == 0.0

    def test_median_and_max(self):
        rows = [
            {
                "oldest_lag": 1,
                "newest_lag": 1,
                "token_weighted_lag": 1.0,
                "generation_span": 0,
                "known_token_count": 1,
                "unknown_token_count": 0,
                "effective_token_count": 1,
                "spans_count": 1,
                "unknown_spans": 0,
            },
            {
                "oldest_lag": 3,
                "newest_lag": 2,
                "token_weighted_lag": 2.5,
                "generation_span": 1,
                "known_token_count": 2,
                "unknown_token_count": 0,
                "effective_token_count": 2,
                "spans_count": 2,
                "unknown_spans": 0,
            },
            {
                "oldest_lag": 10,
                "newest_lag": 5,
                "token_weighted_lag": 8.0,
                "generation_span": 4,
                "known_token_count": 3,
                "unknown_token_count": 0,
                "effective_token_count": 3,
                "spans_count": 3,
                "unknown_spans": 0,
            },
        ]
        summary = compute_run_summary(rows)
        assert summary["oldest_lag_median"] == 3
        assert summary["oldest_lag_max"] == 10
        assert summary["newest_lag_median"] == 2
        assert summary["newest_lag_max"] == 5


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


TRAJECTORY_ROW = {
    "schema_version": "3.2.0",
    "run_id": "run-1",
    "session_id": "sess-1",
    "group_id": "g-1",
    "sample_export_id": "1001",
    "lineage": [
        {"kind": "obs", "state_hash": "h0", "token_count": 2, "weight_versions": [], "weight_version_spans": []},
        {
            "kind": "resp",
            "state_hash": "h1",
            "token_count": 2,
            "weight_versions": ["1"],
            "weight_version_spans": [{"version": "1", "token_start": 0, "token_end": 2, "attempt_seq": 0}],
        },
        {"kind": "obs", "state_hash": "h2", "token_count": 4, "weight_versions": [], "weight_version_spans": []},
        {
            "kind": "resp",
            "state_hash": "h3",
            "token_count": 2,
            "weight_versions": ["3"],
            "weight_version_spans": [{"version": "3", "token_start": 0, "token_end": 2, "attempt_seq": 1}],
        },
    ],
}

TRAIN_ROW = {
    "schema_version": "3.2.0",
    "run_id": "run-1",
    "update_id": "actor:3:2",
    "dp_rank": 0,
    "samples": [{"sample_export_id": "1001", "consumed_weight_version": "7"}],
}


def _aligned_mask_ranges(response_length: int) -> dict:
    """A loss mask that keeps every response token."""
    return {"ranges": [[0, response_length, 1]]}


def test_cli_joins_streams_and_emits_report(tmp_path, capsys):
    root = tmp_path / "run-1"
    _write_jsonl(root / "trajectories" / "shard_00.jsonl", [TRAJECTORY_ROW])
    _write_jsonl(
        root / "train" / "step_00000003_dp_000.jsonl",
        [{**TRAIN_ROW, "samples": [{**TRAIN_ROW["samples"][0], "loss_mask": _aligned_mask_ranges(8)}]}],
    )

    main([str(root)])
    out = capsys.readouterr().out
    # Summary JSON comes first; detail lines start with ``sample=``.
    lines = out.splitlines()
    detail_start = next(index for index, line in enumerate(lines) if line.startswith("sample="))
    summary = json.loads("\n".join(lines[:detail_start]))
    # The join hits: span v1 [0,2) and v3 [6,8) in response coordinates (the
    # 4-token observation sits between the two resp nodes), consumed at 7, so
    # the oldest lag is 6.
    assert summary["lag_sample_count"] == 1
    assert summary["oldest_lag_max"] == 6
    assert summary["span_missing_rate"] == 0.0
    assert summary["unconfirmed_sample_count"] == 0
    # Detail lines carry the traceback chain and consumption age.
    assert "sess-1" in out
    assert "tokens=[0,2)" in out and "tokens=[6,8)" in out
    assert "age=6" in out and "age=4" in out


def test_cli_excludes_unconsumed_exports_from_training_population(tmp_path, capsys):
    # One exported trajectory reached training, another did not (filtered or
    # evaluation-only). The unconsumed export's unknown-version spans must not
    # enter sample_count or the missing rates: with one consumed sample the
    # span missing rate stays 0, not 1/4 as a naive left join would report.
    root = tmp_path / "run-1"
    unconsumed = json.loads(json.dumps(TRAJECTORY_ROW))
    unconsumed["sample_export_id"] = "1002"
    unconsumed["lineage"][1]["weight_version_spans"] = [
        {"version": None, "token_start": 0, "token_end": 2, "attempt_seq": 0}
    ]
    _write_jsonl(root / "trajectories" / "shard_00.jsonl", [TRAJECTORY_ROW, unconsumed])
    _write_jsonl(root / "train" / "step_00000003_dp_000.jsonl", [TRAIN_ROW])

    main([str(root)])
    out = capsys.readouterr().out
    summary = json.loads(out.split("sample=")[0])
    assert summary["sample_count"] == 1
    assert summary["unconfirmed_sample_count"] == 1
    assert summary["span_missing_rate"] == 0.0
    assert "sample=1002" in out and "unconfirmed=true" in out


def test_cli_keeps_consumed_sample_with_unknown_version(tmp_path, capsys):
    # A train record with no consumed version is still confirmed consumption:
    # the sample stays in the population and its lags read as unknown.
    root = tmp_path / "run-1"
    _write_jsonl(root / "trajectories" / "shard_00.jsonl", [TRAJECTORY_ROW])
    _write_jsonl(
        root / "train" / "step_00000003_dp_000.jsonl",
        [{"samples": [{"sample_export_id": "1001", "consumed_weight_version": None}]}],
    )

    main([str(root), "--jsonl"])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["confirmed"] is True
    assert rows[0]["consumed_weight_version"] is None
    assert rows[0]["oldest_lag"] is None


def test_cli_all_zero_loss_mask_reports_zero_effective_tokens(tmp_path, capsys):
    # The only consumed sample trained on nothing: no token-weighted lag, and
    # the zero-token sample is surfaced rather than silently averaging in.
    root = tmp_path / "run-1"
    _write_jsonl(root / "trajectories" / "shard_00.jsonl", [TRAJECTORY_ROW])
    _write_jsonl(
        root / "train" / "step_00000003_dp_000.jsonl",
        [{"samples": [{**TRAIN_ROW["samples"][0], "loss_mask": {"ranges": []}}]}],
    )

    main([str(root)])
    summary = json.loads(capsys.readouterr().out.split("sample=")[0])
    assert summary["sample_count"] == 1
    assert summary["effective_token_count"] == 0
    assert summary["zero_effective_sample_count"] == 1
    assert summary["token_weighted_lag_mean"] is None
    assert summary["token_missing_rate"] is None
    assert summary["oldest_lag_max"] == 6


def test_cli_partial_loss_mask_weights_only_effective_tokens(tmp_path, capsys):
    # The mask keeps only the first turn's tokens; the second turn is masked out.
    root = tmp_path / "run-1"
    _write_jsonl(root / "trajectories" / "shard_00.jsonl", [TRAJECTORY_ROW])
    _write_jsonl(
        root / "train" / "step_00000003_dp_000.jsonl",
        [{"samples": [{**TRAIN_ROW["samples"][0], "loss_mask": {"ranges": [[0, 2, 1]]}}]}],
    )

    main([str(root)])
    summary = json.loads(capsys.readouterr().out.split("sample=")[0])
    assert summary["effective_token_count"] == 2
    assert summary["token_weighted_lag_mean"] == 6.0  # (7-1)*2/2, the v3 turn is masked out


def test_cli_missing_only_filter_and_jsonl(tmp_path, capsys):
    root = tmp_path / "run-1"
    unknown_row = json.loads(json.dumps(TRAJECTORY_ROW))
    unknown_row["sample_export_id"] = "1002"
    unknown_row["lineage"][1]["weight_version_spans"] = [
        {"version": None, "token_start": 0, "token_end": 2, "attempt_seq": 0}
    ]
    _write_jsonl(root / "trajectories" / "shard_00.jsonl", [TRAJECTORY_ROW, unknown_row])
    _write_jsonl(
        root / "train" / "step_00000003_dp_000.jsonl",
        [
            TRAIN_ROW,
            {
                **TRAIN_ROW,
                "samples": [{"sample_export_id": "1002", "consumed_weight_version": "7"}],
            },
        ],
    )

    main([str(root), "--missing-only", "--jsonl"])
    out = capsys.readouterr().out
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["sample_export_id"] == "1002"
    assert rows[0]["spans"][0]["version"] is None


def test_latest_train_step_wins_including_overflow_shards(tmp_path, capsys):
    # Under backpressure train_hooks renames the rewritten shard to
    # overflow_step_...; lexicographic order would let the overflow's old stamp
    # clobber the newer step's, so shards sort numerically by
    # (rollout_id, dp_rank, is_overflow) and step 3's version 7 is final.
    root = tmp_path / "run-1"
    _write_jsonl(root / "trajectories" / "shard_00.jsonl", [TRAJECTORY_ROW])
    _write_jsonl(
        root / "train" / "step_00000002_dp_000.jsonl",
        [{**TRAIN_ROW, "samples": [{"sample_export_id": "1001", "consumed_weight_version": "5"}]}],
    )
    _write_jsonl(
        root / "train" / "overflow_step_00000003_dp_000.jsonl",
        [TRAIN_ROW],
    )

    main([str(root), "--jsonl"])
    out = capsys.readouterr().out
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["consumed_weight_version"] == "7"
    assert rows[0]["oldest_lag"] == 6  # 7 - 1: the join used version 7, not 5


def test_cli_group_filter(tmp_path, capsys):
    root = tmp_path / "run-1"
    other_group = json.loads(json.dumps(TRAJECTORY_ROW))
    other_group["sample_export_id"] = "1002"
    other_group["group_id"] = "g-2"
    _write_jsonl(root / "trajectories" / "shard_00.jsonl", [TRAJECTORY_ROW, other_group])
    _write_jsonl(
        root / "train" / "step_00000003_dp_000.jsonl",
        [
            TRAIN_ROW,
            {**TRAIN_ROW, "samples": [{"sample_export_id": "1002", "consumed_weight_version": "7"}]},
        ],
    )

    main([str(root), "--group", "g-1", "--jsonl"])
    out = capsys.readouterr().out
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert [row["sample_export_id"] for row in rows] == ["1001"]
    assert rows[0]["session_id"] == "sess-1"

    main([str(root), "--group", "g-9", "--jsonl"])
    out = capsys.readouterr().out
    assert out.strip() == ""


def _cjk_torn_tail() -> bytes:
    # A real sample line with CJK text, truncated inside the 3-byte character
    # "中": decoding the whole line raises UnicodeDecodeError — text-mode
    # iteration would raise from the line iterator and bypass the json guard.
    row = dict(TRAJECTORY_ROW, sample_export_id="2001", label="中文")
    encoded = json.dumps(row, ensure_ascii=False).encode("utf-8")
    char_start = encoded.index("中".encode("utf-8"))
    return encoded[: char_start + 2]  # keep only the first 2 of the 3 bytes


@pytest.mark.parametrize(
    "tail",
    [
        b'{"schema_version": "3.2',  # ASCII cut mid-object -> JSONDecodeError
        _cjk_torn_tail(),
    ],
    ids=["ascii_cut", "cjk_multibyte_cut"],
)
def test_cli_tolerates_torn_final_jsonl_line(tmp_path, capsys, tail):
    # The writer contract allows a crash to leave half a line (writers.py); a
    # post-mortem audit must skip it instead of crashing.
    root = tmp_path / "run-1"
    shard = root / "trajectories" / "shard_00.jsonl"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(json.dumps(TRAJECTORY_ROW).encode("utf-8") + b"\n" + tail)
    _write_jsonl(root / "train" / "step_00000003_dp_000.jsonl", [TRAIN_ROW])

    main([str(root)])
    summary = json.loads(capsys.readouterr().out.split("sample=")[0])
    assert summary["lag_sample_count"] == 1  # the complete first line still joins
    assert summary["oldest_lag_max"] == 6


def test_cli_rejects_missing_run_dir(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([str(tmp_path / "no-such-run")])
    assert excinfo.value.code == 2  # argparse parser.error
    assert "run directory not found" in capsys.readouterr().err
