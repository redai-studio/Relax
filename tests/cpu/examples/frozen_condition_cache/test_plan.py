import json

from runner.campaign import KIT, plan, summarize


def test_fixed_stage_counts_and_world_policy(tmp_path):
    config = json.loads((KIT / "configs/campaign.json").read_text())
    prepared = {
        "models": {name: {"path": "/models/" + name, "identity": "a" * 64} for name in config["models"]},
        "data": {
            "root": "/data",
            "debug8": "/data/debug8.json",
            "high128": "/data/high128.json",
            "low1024": "/data/low1024.json",
        },
    }
    for world in [1, 2, 4]:
        stages = plan(config, prepared, {"world": world}, tmp_path, "python")
        assert len(stages) == 28
        assert sum(s["kind"] == "correctness" for s in stages) == 8
        assert sum(s["kind"] == "performance" for s in stages) == 18
        assert sum(s["kind"] == "compare" for s in stages) == 2
        for stage in stages:
            if stage["kind"] == "compare":
                continue
            assert stage["argv"][stage["argv"].index("--global-batch") + 1] == "8"
            assert stage["argv"][stage["argv"].index("--world") + 1] == str(world)
            assert int(stage["argv"][stage["argv"].index("--cache-bytes") + 1]) in [0, 68719476736 // world]
        assert len({stage["name"] for stage in stages}) == 28


def test_missing_results_never_success(tmp_path):
    config = json.loads((KIT / "configs/campaign.json").read_text())
    result = summarize(config, [], tmp_path)
    assert not result["technical_thresholds_met"]
    assert not result["campaign_matrix_completed"]


def test_fast_performance_cannot_override_pending_numerical_acceptance(tmp_path):
    config = json.loads((KIT / "configs/campaign.json").read_text())
    stages = tmp_path / "stages"
    stages.mkdir()
    for model in config["models"]:
        (stages / f"{model}-correctness-compare.json").write_text(
            json.dumps({"passed": True, "strict_acceptance": False, "needs_nondeterminism_review": True})
        )
    for workload in config["workloads"]:
        for repeat in range(1, 4):
            for letter in ("a", "b"):
                directory = stages / f"{workload['name']}-pair{repeat}-{letter}"
                directory.mkdir()
                row = {
                    "native_contract": {"batch": 8},
                    "status": "completed",
                    "warmup_updates": 16,
                    "measured_updates": 64,
                    "measured_seconds": 10.0 if letter == "a" else 8.0,
                    "initialization_plus_total_seconds": 15.0,
                    "warmup_seconds": 4.0,
                    "resources_and_cache": [{"cache": {}, "wall_seconds": 1.0 if letter == "a" else 0.8}] * 80,
                    "scope": "native SFT",
                }
                (directory / "result.json").write_text(json.dumps(row))
    result = summarize(config, [], tmp_path)
    assert result["campaign_matrix_completed"]
    assert result["hard_numerical_bounds_met"]
    assert result["numerical_review_required"]
    assert not result["strict_numerical_acceptance"]
    assert not result["technical_thresholds_met"]
    for model in config["models"]:
        (stages / f"{model}-correctness-compare.json").write_text(
            json.dumps({"passed": True, "strict_acceptance": True, "needs_nondeterminism_review": False})
        )
    assert summarize(config, [], tmp_path)["technical_thresholds_met"]
