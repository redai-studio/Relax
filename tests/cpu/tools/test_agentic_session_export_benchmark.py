# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Contract tests for the Agentic Session export benchmark."""

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
    probes = {"expected": 1, "scheduled": 1, "attempted": 1, "completed": 1, "succeeded": 1, "failed": 0}
    return {
        "scenario": {
            "groups": 1,
            "sessions_per_group": 1,
            "turns": 2,
            "short_rps": 1.0,
            "health_rps": 1.0,
            "export_workers": 0,
        },
        "window_s": 1.0,
        "valid": True,
        "exports_in_window": True,
        "consumer_completed": True,
        "exports": {"expected": 1, "completed": 1, "latency_ms": {"p99": 2.0}},
        "digests": {"session": ["same-content"]},
        "probe_counts": {"short": dict(probes), "health": dict(probes)},
        "short_errors": 0,
        "health_errors": 0,
        "loop_lag_ms": {"p99": 2.0, "count": 10},
        "short_ms": {"p99": 2.0, "count": 1},
        "health_ms": {"p99": 1.0, "count": 1},
        "cpu": {"process_s": 0.1},
        "memory": {"peak_mib": 12.0},
        "teardown": {
            "pending_tasks": 0,
            "groups": 0,
            "session_records": 0,
            "inflight_exports": 0,
            "export_worker_threads": 0,
            "leaked_threads": [],
        },
    }


def _args(out_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        out_dir=str(out_dir),
        baseline_src="base",
        candidate_src="candidate",
        tokenizer="tokenizer",
        preset="default",
        set=[],
        candidate_set=[],
        repeats=1,
        timeout=1.0,
    )


def _patch_run_main(monkeypatch, benchmark, result: dict[str, Any]) -> None:
    monkeypatch.setattr(benchmark, "_bootstrap_src", lambda src: src)
    monkeypatch.setattr(benchmark, "load_tokenizer", lambda _path: (object(), np.array([1], dtype=np.int64)))

    def fake_run(coro):
        coro.close()
        return result

    monkeypatch.setattr(benchmark, "asyncio", SimpleNamespace(run=fake_run))


@pytest.mark.parametrize("failure", ["probe", "teardown"])
def test_run_main_rejects_incomplete_results_and_keeps_diagnostic(benchmark, monkeypatch, tmp_path, failure) -> None:
    result = _valid_result()
    if failure == "probe":
        result["short_errors"] = 1
        result["probe_counts"]["short"].update(succeeded=0, failed=1)
    else:
        result["teardown"]["pending_tasks"] = 1
    _patch_run_main(monkeypatch, benchmark, result)
    output = tmp_path / "run.json"
    args = SimpleNamespace(src="candidate", set=[], preset="default", tokenizer="tokenizer", out=str(output))

    with pytest.raises(SystemExit) as raised:
        benchmark.run_main(args)

    assert raised.value.code != 0
    diagnostic = json.loads(output.read_text(encoding="utf-8"))
    assert diagnostic["valid"] is False and diagnostic["validation_errors"]


def test_run_main_accepts_complete_result(benchmark, monkeypatch, tmp_path) -> None:
    _patch_run_main(monkeypatch, benchmark, _valid_result())
    output = tmp_path / "run.json"
    benchmark.run_main(
        SimpleNamespace(src="candidate", set=[], preset="default", tokenizer="tokenizer", out=str(output))
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["valid"] is True and result["validation_errors"] == []


def test_run_main_never_overwrites_a_result_file(benchmark, tmp_path) -> None:
    output = tmp_path / "run.json"
    output.write_text("old", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        benchmark.run_main(SimpleNamespace(out=str(output)))
    assert output.read_text(encoding="utf-8") == "old"


def test_compare_refuses_to_reuse_a_previous_success_directory(benchmark, monkeypatch, tmp_path) -> None:
    out_dir = tmp_path / "previous-success"
    out_dir.mkdir()
    summary = out_dir / "summary.md"
    summary.write_text("old successful report\n", encoding="utf-8")
    called = False

    def fake_run(_command, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("comparison must not overwrite an existing result directory")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="refusing to reuse"):
        benchmark.compare_main(_args(out_dir))
    assert not called and summary.read_text(encoding="utf-8") == "old successful report\n"
    assert sorted(path.name for path in out_dir.iterdir()) == ["summary.md"]


def test_compare_rejects_candidate_only_workload_change_before_running(benchmark, monkeypatch, tmp_path) -> None:
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    args = _args(tmp_path / "output")
    args.candidate_set = ["short_rps=1"]
    with pytest.raises(SystemExit):
        benchmark.compare_main(args)
    assert not called
    assert (tmp_path / "output/comparison-invalid.json").exists()


@pytest.mark.parametrize("failure", ["probe", "workload", "digest"])
def test_compare_rejects_invalid_or_unequal_runs_without_summary(benchmark, monkeypatch, tmp_path, failure) -> None:
    def fake_run(command, **_kwargs):
        result = _valid_result()
        variant = Path(command[command.index("--out") + 1]).name.split("-")[0]
        if failure == "probe":
            result["short_errors"] = 1
            result["probe_counts"]["short"].update(succeeded=0, failed=1)
        elif failure == "workload" and variant == "candidate":
            result["scenario"]["short_rps"] = 2.0
        elif failure == "digest" and variant == "candidate":
            result["digests"]["session"] = ["different-content"]
        Path(command[command.index("--out") + 1]).write_text(json.dumps(result), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    out_dir = tmp_path / "output"
    with pytest.raises(SystemExit):
        benchmark.compare_main(_args(out_dir))
    assert (out_dir / "baseline-0.json").exists() and (out_dir / "candidate-0.json").exists()
    diagnostic = json.loads((out_dir / "comparison-invalid.json").read_text(encoding="utf-8"))
    assert diagnostic["valid"] is False
    assert not (out_dir / "summary.md").exists()


def test_compare_writes_repeat_ranges_only_after_valid_matching_runs(benchmark, monkeypatch, tmp_path) -> None:
    def fake_run(command, **_kwargs):
        Path(command[command.index("--out") + 1]).write_text(json.dumps(_valid_result()), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    out_dir = tmp_path / "output"
    benchmark.compare_main(_args(out_dir))
    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "baseline median [min, max]" in summary
    assert "event-loop lag P99 (ms)" in summary and "short chat P99 (ms)" in summary
    assert "all runs valid: True" in summary


@pytest.mark.asyncio
async def test_lag_probe_counts_only_formal_window_deadlines(benchmark) -> None:
    probe = benchmark.LagProbe(tick_s=1 / 128)
    await probe.start(window_s=5 / 128)
    await asyncio.sleep(0.07)
    assert len(await probe.stop()) == 4


@pytest.mark.asyncio
async def test_teardown_probe_supports_baseline_shard_without_export_executor(benchmark) -> None:
    shard = SimpleNamespace(_groups={}, _session_records={})
    snapshot = await benchmark._shard_leaks(shard, set())
    assert snapshot["inflight_exports"] == snapshot["export_worker_threads"] == 0


@pytest.mark.parametrize("field", ["tokens", "loss_mask", "rollout_log_probs"])
def test_content_digest_requires_core_training_fields(benchmark, field: str) -> None:
    payload = {"tokens": [1, 2], "loss_mask": [1], "rollout_log_probs": [-0.25]}
    payload[field] = None
    with pytest.raises(ValueError, match=field):
        benchmark._content_digest(payload)


def test_content_digest_includes_array_shape(benchmark) -> None:
    first = {"tokens": [1, 2], "loss_mask": [1], "rollout_log_probs": [-0.25]}
    different_shape = {"tokens": [1, 2, 3], "loss_mask": [1], "rollout_log_probs": [-0.25]}
    assert benchmark._content_digest(first) != benchmark._content_digest(different_shape)


def test_transport_marks_missing_training_content_invalid(benchmark) -> None:
    result = SimpleNamespace(
        exports=({"sample_payload": {"tokens": [1], "loss_mask": [1], "rollout_log_probs": None}},)
    )
    _taken, digests, token_counts = benchmark._transport((result,))
    assert digests[0][0].startswith("invalid-content:") and token_counts == [1]


def test_transport_roundtrips_compact_numpy_arrays(benchmark, monkeypatch) -> None:
    payload = {
        "tokens": np.array([3, 7], dtype=np.int32),
        "loss_mask": np.array([1, 0], dtype=np.int32),
        "rollout_log_probs": np.array([-0.25, 0.0], dtype=np.float32),
        "_rollout_tokens_shared": True,
    }
    received_buffers = []
    loads = benchmark.pickle.loads

    def track_loads(wire, *, buffers):
        received_buffers.extend(buffers)
        return loads(wire, buffers=buffers)

    monkeypatch.setattr(benchmark.pickle, "loads", track_loads)
    result = SimpleNamespace(exports=({"sample_payload": payload},))
    _taken, digests, token_counts = benchmark._transport((result,))

    assert received_buffers
    assert digests == [[benchmark._content_digest(payload)]]
    assert token_counts == [2]


def test_batch_throughput_uses_first_exit_to_last_taken_time(benchmark) -> None:
    result = benchmark._batch_throughput(
        {"a": 0.0, "b": 0.0, "c": 2.0}, {"a": 5.0, "b": 5.0, "c": 7.0}, [["a", "b"], ["c", "d"]]
    )
    assert result == {"completed": 3, "elapsed_s": 7.0, "per_s": pytest.approx(3 / 7)}
