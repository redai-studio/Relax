import copy

import pytest
from experiments.performance_summary import cumulative_payback, summarize


def records():
    result = []
    for workload in ("sensenova_repeated", "qwen_repeated", "sensenova_lowreuse"):
        for pair in range(3):
            for variant in ("online", "cache"):
                seconds = 64.0 if variant == "online" else 60.0
                result.append(
                    {
                        "workload": workload,
                        "pair": pair,
                        "variant": variant,
                        "result": {
                            "native_contract": {"fixed_batch": 8},
                            "status": "completed",
                            "warmup_updates": 16,
                            "measured_updates": 64,
                            "cache_budget_per_rank": 0 if variant == "online" else 4096,
                            "measured_seconds": seconds,
                            "initialization_plus_total_seconds": seconds + 12,
                            "resources_and_cache": [{"wall_seconds": seconds / 64, "cache": {"bytes": 4000}}] * 80,
                        },
                    }
                )
    return result


def test_fixed_18_segments_all_retained_with_cold_and_steady_evidence():
    output = summarize(records())
    assert output["completed_segments"] == 18
    assert len(output["workloads"]) == 3
    for workload in output["workloads"].values():
        assert workload["median_at_least_5_percent"]
        assert len(workload["pairs"]) == 3
        assert workload["pairs"][0]["cold_training_cumulative_cache_seconds"] > 60


def test_no_missing_run_or_changed_ab_recipe_can_pass():
    with pytest.raises(ValueError):
        summarize(records()[:-1])
    rows = copy.deepcopy(records())
    rows[0]["result"]["native_contract"]["fixed_batch"] = 4
    with pytest.raises(ValueError):
        summarize(rows)


def test_payback_uses_actual_timeline_and_retains_non_sustained_crossing():
    a = {"resources_and_cache": [{"wall_seconds": 1.0}] * 80, "measured_seconds": 64.0}
    # Cache first wins at update1, loses at2, then becomes and stays ahead at3.
    b = {
        "resources_and_cache": [{"wall_seconds": value} for value in [0.5, 2.0] + [0.1] * 78],
        "measured_seconds": 6.4,
    }
    result = cumulative_payback(a, b)
    assert result["first_observed_cumulative_crossing_update"] == 1
    assert not result["crossing_is_sustained_to_end"]
    assert result["first_observed_sustained_crossing_update"] == 3
    assert result["cold_training_cumulative_online_seconds"] == 80
    assert result["cold_training_cumulative_cache_seconds"] == pytest.approx(10.3)
    assert "projected_payback_updates_from_warmup" in result


def test_payback_refuses_incomplete_or_nonfinite_update_evidence():
    row = {
        "resources_and_cache": [{"wall_seconds": 1.0}] * 80,
        "measured_seconds": 64.0,
    }
    for broken in [
        row | {"resources_and_cache": row["resources_and_cache"][:-1]},
        row | {"resources_and_cache": [{"wall_seconds": float("nan")}] * 80},
    ]:
        with pytest.raises(ValueError):
            cumulative_payback(row, broken)
