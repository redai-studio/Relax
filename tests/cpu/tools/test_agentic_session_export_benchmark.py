# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU contract tests for the agentic Session export benchmark CLI."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).resolve().parents[3] / "scripts" / "benchmarks" / "benchmark_agentic_session_export.py"
    spec = importlib.util.spec_from_file_location("agentic_session_export_benchmark_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _valid_result() -> dict[str, Any]:
    return {
        "scenario": {
            "groups": 1,
            "sessions_per_group": 1,
            "turns": 2,
            "short_rps": 1.0,
            "health_rps": 1.0,
            "wave_interval_s": 1.0,
            "window_s": 1.0,
            "export_workers": 0,
        },
        "window_s": 1.0,
        "valid": True,
        "exports_in_window": True,
        "exports": {
            "expected": 1,
            "completed": 1,
            "latency_ms": {"p50": 1.0, "p99": 2.0},
            "wave_drain_ms": {"p50": 1.0},
            "throughput_per_s": 1.0,
            "mean_tokens": 1.0,
        },
        "digests": {"long-g0-s0": ["abc"]},
        "probe_counts": {
            "short": {
                "expected": 1,
                "scheduled": 1,
                "attempted": 1,
                "completed": 1,
                "succeeded": 1,
                "failed": 0,
            },
            "health": {
                "expected": 1,
                "scheduled": 1,
                "attempted": 1,
                "completed": 1,
                "succeeded": 1,
                "failed": 0,
            },
        },
        "short_errors": 0,
        "health_errors": 0,
        "teardown": {
            "pending_tasks": 0,
            "groups": 0,
            "session_records": 0,
            "inflight_exports": 0,
            "export_worker_threads": 0,
            "leaked_threads": [],
        },
        "cpu": {"process_s": 0.1, "shard_loop_s": 0.05, "threads_s": {}},
        "memory": {"start_mib": 10.0, "peak_mib": 12.0},
        "loop_lag_ms": {"p99": 2.0, "p999": 3.0, "max": 4.0, "count": 10},
        "short_ms": {"p50": 1.0, "p99": 2.0, "max": 3.0, "count": 1},
        "health_ms": {"p99": 1.0, "count": 1},
        "timeline_short_max_ms": [1.0, 0.0],
        "wave_offsets_s": [1.0],
    }


def _patch_run_main(monkeypatch, benchmark, result: dict[str, Any]) -> None:
    monkeypatch.setattr(benchmark, "_bootstrap_src", lambda src: src)
    monkeypatch.setattr(benchmark, "load_tokenizer", lambda _path: (object(), np.array([1], dtype=np.int64)))

    def fake_run(coro):
        coro.close()
        return result

    monkeypatch.setattr(benchmark, "asyncio", SimpleNamespace(run=fake_run))


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("probe", "short requests"),
        ("pending_task", "teardown"),
    ],
)
def test_run_main_marks_failed_probe_or_teardown_invalid_and_keeps_json(
    benchmark, monkeypatch, tmp_path, failure: str, expected_error: str
) -> None:
    result = _valid_result()
    if failure == "probe":
        result["short_errors"] = 1
        result["probe_counts"]["short"].update(succeeded=0, failed=1)
    else:
        result["teardown"]["pending_tasks"] = 1
    _patch_run_main(monkeypatch, benchmark, result)
    output = tmp_path / "run.json"
    args = SimpleNamespace(
        src="candidate-src",
        set=[],
        preset="default",
        tokenizer="tokenizer",
        out=str(output),
        profile=0,
    )

    with pytest.raises(SystemExit) as raised:
        benchmark.run_main(args)

    assert raised.value.code != 0
    diagnostic = json.loads(output.read_text(encoding="utf-8"))
    assert diagnostic["valid"] is False
    assert any(expected_error in item for item in diagnostic["validation_errors"])


def test_run_main_accepts_a_complete_valid_result(benchmark, monkeypatch, tmp_path) -> None:
    _patch_run_main(monkeypatch, benchmark, _valid_result())
    output = tmp_path / "run.json"
    args = SimpleNamespace(
        src="candidate-src",
        set=[],
        preset="default",
        tokenizer="tokenizer",
        out=str(output),
        profile=0,
    )

    benchmark.run_main(args)

    diagnostic = json.loads(output.read_text(encoding="utf-8"))
    assert diagnostic["valid"] is True
    assert diagnostic["validation_errors"] == []


def test_compare_rejects_candidate_only_load_change_before_starting_runs(benchmark, monkeypatch, tmp_path) -> None:
    called = False

    def fake_run(command, **_kwargs):
        nonlocal called
        called = True
        output = Path(command[command.index("--out") + 1])
        output.write_text(json.dumps(_valid_result()), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    args = SimpleNamespace(
        out_dir=str(tmp_path),
        baseline_src="baseline-src",
        candidate_src="candidate-src",
        tokenizer="tokenizer",
        preset="default",
        set=[],
        candidate_set=["short_rps=1"],
        repeats=1,
        timeout=1.0,
    )

    with pytest.raises(SystemExit) as raised:
        benchmark.compare_main(args)

    assert raised.value.code != 0
    assert not called


def test_compare_preserves_invalid_run_diagnostics_without_percentages(benchmark, monkeypatch, tmp_path) -> None:
    def fake_run(command, **_kwargs):
        result = _valid_result()
        result["short_errors"] = 1
        result["probe_counts"]["short"].update(succeeded=0, failed=1)
        output = Path(command[command.index("--out") + 1])
        output.write_text(json.dumps(result), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    args = SimpleNamespace(
        out_dir=str(tmp_path),
        baseline_src="baseline-src",
        candidate_src="candidate-src",
        tokenizer="tokenizer",
        preset="default",
        set=[],
        candidate_set=[],
        repeats=1,
        timeout=1.0,
    )

    with pytest.raises(SystemExit) as raised:
        benchmark.compare_main(args)

    assert raised.value.code != 0
    assert (tmp_path / "baseline-0.json").exists()
    assert (tmp_path / "candidate-0.json").exists()
    assert (tmp_path / "comparison-invalid.json").exists()
    assert not (tmp_path / "summary.md").exists()


def test_compare_rejects_different_export_digests_without_percentages(benchmark, monkeypatch, tmp_path) -> None:
    def fake_run(command, **_kwargs):
        result = _valid_result()
        output = Path(command[command.index("--out") + 1])
        variant = output.name.split("-")[0]
        result["digests"] = {"long-g0-s0": [variant]}
        output.write_text(json.dumps(result), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    args = SimpleNamespace(
        out_dir=str(tmp_path),
        baseline_src="baseline-src",
        candidate_src="candidate-src",
        tokenizer="tokenizer",
        preset="default",
        set=[],
        candidate_set=[],
        repeats=1,
        timeout=1.0,
    )

    with pytest.raises(SystemExit) as raised:
        benchmark.compare_main(args)

    assert raised.value.code != 0
    diagnostic = json.loads((tmp_path / "comparison-invalid.json").read_text(encoding="utf-8"))
    assert any("fields differ" in error for error in diagnostic["errors"])
    assert not (tmp_path / "summary.md").exists()


@pytest.mark.asyncio
async def test_lag_probe_counts_only_formal_window_deadlines(benchmark) -> None:
    # Binary-exact boundaries keep this check independent of clock uptime.
    probe = benchmark.LagProbe(tick_s=1 / 128)

    await probe.start(window_s=5 / 128)
    await asyncio.sleep(0.07)
    _raw, slots = await probe.stop()

    assert len(slots) == 4


@pytest.mark.asyncio
async def test_teardown_probe_supports_baseline_shard_without_export_executor(benchmark) -> None:
    shard = SimpleNamespace(_groups={}, _session_records={})

    snapshot = await benchmark._shard_leaks(shard, set())

    assert snapshot["inflight_exports"] == 0
    assert snapshot["export_worker_threads"] == 0


@pytest.mark.parametrize("field", ["tokens", "loss_mask", "rollout_log_probs"])
def test_content_digest_requires_core_training_fields(benchmark, field: str) -> None:
    payload = {"tokens": [1, 2], "loss_mask": [1], "rollout_log_probs": [-0.25]}
    payload[field] = None

    with pytest.raises(ValueError, match=field):
        benchmark._content_digest(payload)


def test_content_digest_hashes_shapes_without_assuming_fields_are_equally_long(benchmark) -> None:
    first = {"tokens": [1, 2], "loss_mask": [1], "rollout_log_probs": [-0.25]}
    different_token_shape = {"tokens": [1, 2, 3], "loss_mask": [1], "rollout_log_probs": [-0.25]}

    assert benchmark._content_digest(first) != benchmark._content_digest(different_token_shape)


def test_transport_keeps_missing_training_fields_as_invalid_content_diagnostic(benchmark) -> None:
    result = SimpleNamespace(
        exports=({"sample_payload": {"tokens": [1], "loss_mask": [1], "rollout_log_probs": None}},)
    )

    _pickled_at, digests, token_counts = benchmark._transport((result,))

    assert digests[0][0].startswith("invalid-content:")
    assert token_counts == [1]


def test_export_throughput_separates_batch_wall_time_from_active_drain(benchmark) -> None:
    groups = [["a", "b"], ["c", "d"]]
    exit_at = {"a": 0.0, "b": 0.0, "c": 2.0, "d": 2.0}
    taken_at = {"a": 5.0, "b": 5.0, "c": 7.0, "d": 7.0}

    metrics = benchmark._export_throughput(exit_at, taken_at, groups)

    assert metrics["batch_elapsed_s"] == 7.0
    assert metrics["batch_wallclock_per_s"] == pytest.approx(4 / 7)
    assert metrics["active_drain_elapsed_s"] == 10.0
    assert metrics["active_drain_per_s"] == pytest.approx(4 / 10)
