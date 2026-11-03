# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import hashlib
import json

import pytest

from scripts.tools.benchmark_condition_cache import compare_runs, main, summarize_run


def write_run(path, *, mode="online", wall=(2.0, 2.0, 2.0), config=None, init_seconds=1.0):
    path.mkdir()
    for rank in range(2):
        events = [
            {
                "stage": "init",
                "rank": rank,
                "world_size": 2,
                "global_batch_size": 8,
                "run_id": path.name,
                "mode": mode,
                "metadata_sha256": "data-v1",
                "model_fingerprint": "model-v1",
                "config": config or {"seed": 42, "prefetch": 4},
                "time_init_seconds": init_seconds,
                "start_step": 0,
                "expected_steps": len(wall),
            }
        ]
        for step, seconds in enumerate(wall):
            events.append(
                {
                    "stage": "step",
                    "rank": rank,
                    "step": step,
                    "global_samples": 8,
                    "local_samples": 4,
                    "sample_ids": list(range(step * 8 + rank * 4, step * 8 + rank * 4 + 4)),
                    "wall_seconds": seconds - 0.1 if rank == 0 else seconds,
                    "condition_seconds": 0.2,
                    "encode_seconds": 0.1,
                    "cache_lookup_seconds": 0.01,
                    "cache_store_seconds": 0.02,
                    "encoding_calls": 4 if mode == "online" or step == 0 else 0,
                    "effective_hit": 4 if mode == "cached" and step else 0,
                    "cache_bytes": 64,
                    "cpu_rss_bytes": 100 + rank,
                    "shm_used_bytes": 80,
                    "gpu_allocated_bytes": 50 + rank,
                    "loss": 1.0,
                }
            )
        events.append(
            {
                "stage": "close",
                "rank": rank,
                "completed_steps": len(wall),
                "total_seconds": init_seconds + sum(wall),
                "close_seconds": 0.01,
                "cache_bytes": 0,
            }
        )
        (path / f"events-rank{rank}.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")
    return path


def edit_event(path, rank, index, **updates):
    source = path / f"events-rank{rank}.jsonl"
    events = [json.loads(line) for line in source.read_text().splitlines()]
    events[index].update(updates)
    source.write_text("\n".join(json.dumps(event) for event in events) + "\n")


def test_counts_global_samples_once_and_uses_slowest_rank(tmp_path):
    path = write_run(tmp_path / "online")
    run = summarize_run(path, cold_steps=1)
    assert run["complete"]
    assert run["training"] == {"steps": 3, "samples": 24, "seconds": 6.0, "samples_per_second": 4.0}
    assert run["cold"]["samples"] == 8
    assert run["steady"]["samples"] == 16
    assert run["counter_totals"]["encoding_calls"] == 24
    assert run["resources"]["cache_bytes"]["max_observed_bytes"] == 64
    assert run["resources"]["cache_bytes"]["sum_rank_peaks_bytes"] is None
    assert run["resources"]["shm_used_bytes"]["sum_rank_peaks_bytes"] is None
    assert run["resources"]["cpu_rss_bytes"]["sum_rank_peaks_bytes"] == 201


def test_speedup_and_payback_use_e2e_not_encoding_time(tmp_path):
    online = summarize_run(write_run(tmp_path / "online", wall=(2.0, 2.0, 2.0)), 1)
    cached = summarize_run(write_run(tmp_path / "cached", mode="cached", wall=(3.0, 1.0, 1.0), init_seconds=2), 1)
    result = compare_runs([online], [cached])
    assert result["comparable"]
    assert result["steady_throughput_improvement"] == 1
    assert result["cold_epoch_extra_seconds"] == 1
    assert result["startup_extra_seconds"] == 1
    # 2 seconds excess / (0.25 - 0.125) seconds saved per sample.
    assert result["estimated_additional_samples_to_amortize"] == 16
    assert result["estimated_additional_epochs_to_amortize"] == 2
    assert cached["effective_hit_rate"] == pytest.approx(2 / 3)
    assert cached["phase_counter_totals"]["cold"]["encoding_calls"] == 8
    assert cached["phase_counter_totals"]["steady"]["effective_hit"] == 16
    assert not result["repeated_measurements"]
    assert not result["source_provenance_verified"]


def write_source_provenance(path, patch=b"the implementation patch\n"):
    base = "1" * 40
    digest = hashlib.sha256(base.encode() + b"\n" + patch).hexdigest()
    (path / "request.json").write_text(json.dumps({"base_commit": base, "implementation_sha256": digest}))
    (path / "implementation.patch").write_bytes(patch)
    return digest


def test_matching_implementation_hashes_and_artifacts_are_verified(tmp_path):
    paths = [write_run(tmp_path / "online"), write_run(tmp_path / "cached", mode="cached")]
    expected = [write_source_provenance(path) for path in paths]
    runs = [summarize_run(path, 1) for path in paths]
    assert [run["implementation_sha256"] for run in runs] == expected
    assert all(run["source_artifact_verified"] for run in runs)
    result = compare_runs([runs[0]], [runs[1]])
    assert result["comparable"]
    assert result["source_provenance_verified"] and result["source_artifacts_verified"]


def test_different_implementations_cannot_produce_speedup(tmp_path):
    paths = [write_run(tmp_path / "online"), write_run(tmp_path / "cached", mode="cached")]
    write_source_provenance(paths[0], b"old implementation\n")
    write_source_provenance(paths[1], b"different implementation\n")
    online, cached = [summarize_run(path, 1) for path in paths]
    result = compare_runs([online], [cached])
    assert not result["comparable"] and not result["source_provenance_verified"]
    assert "implementation source hashes differ" in result["reasons"]
    assert "steady_throughput_improvement" not in result


def test_mixed_known_and_missing_implementation_provenance_is_rejected(tmp_path):
    paths = [write_run(tmp_path / "online"), write_run(tmp_path / "cached", mode="cached")]
    write_source_provenance(paths[0])
    online, cached = [summarize_run(path, 1) for path in paths]
    result = compare_runs([online], [cached])
    assert not result["comparable"] and not result["source_provenance_verified"]
    assert "implementation source provenance is missing for some runs" in result["reasons"]


def test_modified_patch_is_reported_as_invalid_source_provenance(tmp_path):
    path = write_run(tmp_path / "online")
    write_source_provenance(path)
    (path / "implementation.patch").write_bytes(b"modified after the run\n")
    run = summarize_run(path, 1)
    assert not run["complete"] and run["source_artifact_verified"] is False
    assert "implementation.patch and base_commit do not match implementation_sha256" in run["issues"]


@pytest.mark.parametrize("request_text", ["{broken", "[]", '{"implementation_sha256":"invalid"}'])
def test_malformed_source_provenance_never_becomes_a_verified_comparison(tmp_path, request_text):
    path = write_run(tmp_path / "online")
    (path / "request.json").write_text(request_text)
    run = summarize_run(path, 1)
    assert not run["complete"]
    assert run["implementation_sha256"] is None and run["source_artifact_verified"] is False
    assert any("source provenance" in issue for issue in run["issues"])


@pytest.mark.parametrize("change", [{"seed": 43, "prefetch": 4}, {"seed": 42, "prefetch": 0}])
def test_refuses_speedup_for_different_training_or_prefetch(tmp_path, change):
    online = summarize_run(write_run(tmp_path / "online"), 1)
    cached = summarize_run(write_run(tmp_path / "cached", mode="cached", config=change), 1)
    result = compare_runs([online], [cached])
    assert not result["comparable"]
    assert "steady_throughput_improvement" not in result


def test_ignores_only_cache_controls_and_output_paths(tmp_path):
    online = summarize_run(
        write_run(tmp_path / "online", config={"seed": 42, "cache_enabled": False, "save": "/a"}), 1
    )
    cached = summarize_run(
        write_run(tmp_path / "cached", mode="cached", config={"seed": 42, "cache_enabled": True, "save": "/b"}), 1
    )
    assert compare_runs([online], [cached])["comparable"]


@pytest.mark.parametrize("where", ["config", "event"])
def test_numerical_validation_lane_never_produces_performance_claim(tmp_path, where):
    paths = [write_run(tmp_path / "online"), write_run(tmp_path / "cached", mode="cached")]
    for path in paths:
        for rank in range(2):
            if where == "config":
                edit_event(path, rank, 0, config={"seed": 42, "conditioning_verification": True})
            else:
                edit_event(path, rank, 0, verification=True)
    online, cached = [summarize_run(path, 1) for path in paths]
    assert online["validation_lane"] and cached["validation_lane"]
    comparison = compare_runs([online], [cached])
    assert not comparison["comparable"]
    assert "numerical validation lane cannot be used as a performance comparison" in comparison["reasons"]
    assert "steady_throughput_improvement" not in comparison


def test_resume_checkpoint_is_not_an_automatic_performance_comparison(tmp_path):
    config = {"seed": 42, "resume_checkpoint": "/runs/previous/i2i_sft/iter_0000016"}
    online = summarize_run(write_run(tmp_path / "online", config=config), 1)
    cached = summarize_run(write_run(tmp_path / "cached", mode="cached", config=config), 1)
    result = compare_runs([online], [cached])
    assert not online["fresh_initialization"]
    assert not result["comparable"]
    assert "automatic speedup requires fresh initialization without a resume checkpoint" in result["reasons"]


def test_job_metadata_adds_full_job_scope_without_substituting_trainer_time(tmp_path):
    a = write_run(tmp_path / "online")
    b = write_run(tmp_path / "cached", mode="cached", wall=(2, 1, 1))
    for path, end in ((a, 110_000), (b, 105_000)):
        (path / "job.json").write_text(json.dumps({"status": "SUCCEEDED", "start_time": 100_000, "end_time": end}))
    online, cached = [summarize_run(path, 1) for path in (a, b)]
    assert online["job_wall_seconds"] == 10
    assert online["total_seconds"] == 7
    result = compare_runs([online], [cached])
    assert result["online_mean_job_wall_seconds"] == 10
    assert result["online_mean_trainer_seconds"] == 7
    assert result["full_job_throughput_improvement"] == 1


def test_missing_job_metadata_is_unknown_and_failed_job_prevents_claim(tmp_path):
    a = write_run(tmp_path / "online")
    b = write_run(tmp_path / "cached", mode="cached")
    online, cached = [summarize_run(path, 1) for path in (a, b)]
    result = compare_runs([online], [cached])
    assert online["job_wall_seconds"] is None and online["job_complete"] is None
    assert result["full_job_throughput_improvement"] is None
    assert "unknown" in result["job_timing_note"]
    (b / "job.json").write_text(json.dumps({"status": "FAILED", "start_time": 0, "end_time": 1000}))
    cached = summarize_run(b, 1)
    assert not cached["complete"]
    assert not compare_runs([online], [cached])["comparable"]


def test_actual_effective_sample_order_must_match(tmp_path):
    online = summarize_run(write_run(tmp_path / "online"), 1)
    path = write_run(tmp_path / "cached", mode="cached")
    edit_event(path, 0, 1, sample_ids=[3, 2, 1, 0])
    result = compare_runs([online], [summarize_run(path, 1)])
    assert not result["comparable"] and not result["sample_order_verified"]
    assert "actual effective sample order differs" in result["reasons"]


def test_missing_sample_order_remains_unverified(tmp_path):
    online = summarize_run(write_run(tmp_path / "online"), 1)
    path = write_run(tmp_path / "cached", mode="cached")
    edit_event(path, 0, 1, sample_ids=None)
    cached = summarize_run(path, 1)
    assert cached["sample_order"] is None
    result = compare_runs([online], [cached])
    assert not result["comparable"] and not result["sample_order_verified"]


def test_sample_ids_cannot_hide_a_different_effective_batch(tmp_path):
    path = write_run(tmp_path / "online")
    edit_event(path, 0, 1, sample_ids=[0])
    with pytest.raises(ValueError, match="sample_ids length"):
        summarize_run(path, 1)


def test_incomplete_rank_or_close_never_claims_speedup(tmp_path):
    online = summarize_run(write_run(tmp_path / "online"), 1)
    path = write_run(tmp_path / "cached", mode="cached")
    source = path / "events-rank1.jsonl"
    source.write_text("\n".join(source.read_text().splitlines()[:-1]) + "\n")
    cached = summarize_run(path, 1)
    assert not cached["complete"]
    assert not compare_runs([online], [cached])["comparable"]


@pytest.mark.parametrize("change", [{"status": "cancelled"}, {"status": "error"}])
def test_cancelled_or_error_close_is_incomplete(tmp_path, change):
    path = write_run(tmp_path / "online")
    edit_event(path, 0, 4, **change)
    assert not summarize_run(path, 1)["complete"]


def test_planned_length_and_rank_provenance_are_checked(tmp_path):
    path = write_run(tmp_path / "online")
    edit_event(path, 0, 0, expected_steps=8)
    edit_event(path, 1, 0, expected_steps=8)
    assert "observed steps differ from the planned run length" in summarize_run(path, 1)["issues"]
    edit_event(path, 1, 0, run_id="another-run")
    assert "rank configurations disagree" in summarize_run(path, 1)["issues"]


def test_missing_measurements_remain_unknown(tmp_path):
    path = write_run(tmp_path / "online")
    edit_event(path, 1, 2, encoding_calls=None)
    run = summarize_run(path, 1)
    assert run["counter_totals"]["encoding_calls"] is None
    assert run["counter_coverage"]["encoding_calls"] == {"observed": 5, "expected": 6}
    assert run["resources"]["disk_bytes"]["max_observed_bytes"] is None
    assert run["effective_hit_rate"] is None


@pytest.mark.parametrize(
    "updates",
    [
        {"wall_seconds": 0},
        {"wall_seconds": float("nan")},
        {"loss": float("inf")},
        {"local_samples": 3},
        {"global_samples": 4},
    ],
)
def test_rejects_invalid_measurements_or_effective_batch(tmp_path, updates):
    path = write_run(tmp_path / "online")
    edit_event(path, 0, 1, **updates)
    with pytest.raises(ValueError):
        summarize_run(path, 1)


def test_prefers_raw_events_to_duplicate_driver_log(tmp_path):
    path = write_run(tmp_path / "online")
    (path / "driver.log").write_text((path / "events-rank0.jsonl").read_text())
    assert summarize_run(path, 1)["training"]["samples"] == 24


def test_cli_writes_both_reuse_comparisons(tmp_path, capsys):
    online = write_run(tmp_path / "online")
    cached = write_run(tmp_path / "cached", mode="cached", wall=(2, 1, 1))
    output = tmp_path / "report.json"
    main(
        [
            "--online",
            str(online),
            "--cached",
            str(cached),
            "--low-reuse-online",
            str(online),
            "--low-reuse-cached",
            str(cached),
            "--cold-steps",
            "1",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert report["comparison"]["comparable"]
    assert report["low_reuse_comparison"]["comparable"]
    assert "steady throughput +100.00%" in capsys.readouterr().out


def test_repeated_run_variation_is_reported(tmp_path):
    online = [
        summarize_run(write_run(tmp_path / f"online-{index}", wall=(2, seconds, seconds)), 1)
        for index, seconds in enumerate((2, 4))
    ]
    cached = [
        summarize_run(write_run(tmp_path / f"cached-{index}", mode="cached", wall=(2, 1, 1)), 1) for index in range(2)
    ]
    result = compare_runs(online, cached)
    assert result["repeated_measurements"]
    assert result["online_steady_samples_per_second"]["stdev"] > 0


def test_log_input_and_malformed_raw_events(tmp_path):
    path = write_run(tmp_path / "online")
    log = tmp_path / "driver.log"
    log.write_text(
        "irrelevant log\n"
        + "\n".join(
            "event: " + line for source in sorted(path.glob("*.jsonl")) for line in source.read_text().splitlines()
        )
    )
    assert summarize_run(log, 1)["complete"]
    (path / "events-rank0.jsonl").write_text("{broken\n")
    with pytest.raises(ValueError, match="Malformed JSON"):
        summarize_run(path, 1)
