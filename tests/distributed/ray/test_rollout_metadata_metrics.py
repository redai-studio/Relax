# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU tests for rollout metadata statistics and logging."""

import copy
import importlib.util
import math
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from relax.agentic.runner.ipc import SessionOutput
from relax.agentic.session.state import SessionForest
from relax.utils.types import Sample


_EXPECTED_REASONS = (
    "budget_exhausted",
    "completed",
    "context_exhausted",
    "discarded_by_pipeline",
    "env_done",
    "env_error",
    "finish_abort",
    "finish_length",
    "format_error",
    "llm_unavailable",
    "max_turns",
    "other",
    "unknown",
)


@pytest.fixture(scope="module")
def rollout_module() -> ModuleType:
    # Load the complete module, isolating only unused serving/queue imports.
    # All Sample, statistics, aggregation and logger code remains real.
    constants = ModuleType("sglang.srt.constants")
    for name in ("CUDA_GRAPH", "KV_CACHE", "WEIGHTS"):
        setattr(constants, f"GPU_MEMORY_TYPE_{name}", name.lower())
    engine = ModuleType("relax.backends.sglang.sglang_engine")
    engine.SGLangEngine = object
    name = "relax.distributed.ray._test_metadata_rollout"
    path = Path(__file__).resolve().parents[3] / "relax/distributed/ray/rollout.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "sglang.srt.constants", constants)
        patch.setitem(sys.modules, "relax.backends.sglang.sglang_engine", engine)
        patch.setitem(sys.modules, "transfer_queue", ModuleType("transfer_queue"))
        patch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


def _args(**overrides: Any) -> SimpleNamespace:
    values = {
        "log_reward_category": None,
        "reward_key": None,
        "log_passrate": False,
        "advantage_estimator": "ppo",
        "partial_rollout": False,
        "fully_async": False,
        "custom_rollout_log_function_path": None,
        "custom_eval_rollout_log_function_path": None,
        "load_debug_rollout_data": False,
        "rollout_num_gpus": 1,
        "wandb_always_use_train_step": False,
    }
    return SimpleNamespace(**(values | overrides))


def _samples() -> list[Sample]:
    return [
        Sample(
            index=0,
            response="a",
            response_length=1,
            reward=1.0,
            status=Sample.Status.COMPLETED,
            metadata={"rollout_stop_reason": "env_done", "rollout_turns": 1},
        ),
        Sample(
            index=1,
            response="b",
            response_length=2,
            reward=None,
            metadata={"stop_reason": "env_done", "rollout_turns": 2},
        ),
        Sample(
            index=2,
            response="c",
            response_length=3,
            reward=0.0,
            status=Sample.Status.TRUNCATED,
            metadata={"rollout_stop_reason": "max_turns", "rollout_turns": 4},
        ),
        Sample(index=3, response="d", response_length=4, reward=None, status=Sample.Status.ABORTED),
    ]


def _expected() -> dict[str, float]:
    return {
        f"stop_reason/{reason}/{statistic}": 0.0 for reason in _EXPECTED_REASONS for statistic in ("count", "ratio")
    } | {
        "stop_reason/env_done/count": 2.0,
        "stop_reason/env_done/ratio": 0.5,
        "stop_reason/max_turns/count": 1.0,
        "stop_reason/max_turns/ratio": 0.25,
        "stop_reason/unknown/count": 1.0,
        "stop_reason/unknown/ratio": 0.25,
        "num_turn/min": 1.0,
        "num_turn/mean": 2.0,
        "num_turn/max": 4.0,
        "num_turn/p50": 1.5,
        "num_turn/p90": 3.4,
        "num_turn/p95": 3.7,
        "num_turn/p99": 3.94,
    }


def test_rollout_metadata_empty_input(rollout_module: ModuleType) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert rollout_module._compute_rollout_metadata_stats([]) == {}
        assert rollout_module.compute_metrics_from_samples(None, []) == {}


def test_rollout_metadata_distribution_and_percentiles(rollout_module: ModuleType) -> None:
    samples = _samples()
    metrics = rollout_module._compute_rollout_metadata_stats(samples)
    assert metrics == pytest.approx(_expected())
    assert sum(value for key, value in metrics.items() if key.endswith("/count")) == len(samples)
    assert sum(value for key, value in metrics.items() if key.endswith("/ratio")) == pytest.approx(1.0)
    assert all(type(value) in (int, float) and math.isfinite(value) for value in metrics.values())
    assert all(type(value) is float for key, value in metrics.items() if key.startswith("stop_reason/"))
    reason_keys = [key for key in metrics if key.startswith("stop_reason/")]
    assert reason_keys == sorted(reason_keys)


@pytest.mark.parametrize("metadata", [None, {}, {"num_turn": 99, "agentic_trace": {"turn_count": 99}}])
def test_rollout_metadata_missing_fields(rollout_module: ModuleType, metadata: dict | None) -> None:
    samples = [Sample(metadata=metadata), Sample()]
    metrics = rollout_module._compute_rollout_metadata_stats(samples)
    assert metrics["stop_reason/unknown/count"] == 2.0
    assert metrics["stop_reason/unknown/ratio"] == 1.0
    assert all(value == 1 for key, value in metrics.items() if key.startswith("num_turn/"))
    assert rollout_module.compute_metrics_from_samples(_args(), samples)["num_turn/p99"] == 1.0


@pytest.mark.parametrize(
    ("primary", "fallback", "expected"),
    [
        ("completed", "env_done", "completed"),
        ("  env_done \t", "max_turns", "env_done"),
        ("unknown", "env_done", "unknown"),
        (None, "env_done", "env_done"),
        ("", " env_done ", "env_done"),
        (" \t", "env_done", "env_done"),
        (42, "env_done", "env_done"),
        (False, "env_done", "env_done"),
        ({}, "env_done", "env_done"),
        (None, None, "unknown"),
        ([], " \t", "unknown"),
        (None, 42, "unknown"),
        ("Env_Done", None, "other"),
        ("env_error:detail/part", None, "env_error"),
        (None, " env_error:request/42 ", "env_error"),
        ("future_reason", "env_done", "other"),
        ("env_error:detail/part", "env_done", "env_error"),
        ("other", "env_done", "other"),
        (None, "future_reason", "other"),
        ("env_error:", None, "env_error"),
        ("env_error/42", None, "other"),
        ("env_error_extra:42", None, "other"),
        ("ENV_ERROR:detail", "env_done", "other"),
        ("end_turn", None, "other"),
        ("max_tokens", None, "other"),
        ("tool_use", None, "other"),
    ],
)
def test_rollout_metadata_stop_reason_precedence(
    rollout_module: ModuleType, primary: Any, fallback: Any, expected: str
) -> None:
    sample = Sample(metadata={"rollout_stop_reason": primary, "stop_reason": fallback, "rollout_turns": 3})
    metrics = rollout_module._compute_rollout_metadata_stats([sample])
    assert metrics[f"stop_reason/{expected}/count"] == 1.0
    assert metrics[f"stop_reason/{expected}/ratio"] == 1.0
    assert len(metrics) == 33
    for reason in _EXPECTED_REASONS:
        if reason != expected:
            assert metrics[f"stop_reason/{reason}/count"] == 0.0
            assert metrics[f"stop_reason/{reason}/ratio"] == 0.0
    assert all(value == 3 for key, value in metrics.items() if key.startswith("num_turn/"))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (5, 5),
        (np.int64(7), 7),
        (np.uint64(8), 8),
        (2.0, 2),
        (np.float32(4), 4),
        (np.float64(6), 6),
        (None, 1),
        (True, 1),
        (False, 1),
        (np.bool_(False), 1),
        ("8", 1),
        (-1, 1),
        (np.int64(-3), 1),
        (-2.0, 1),
        (2.5, 1),
        (float("nan"), 1),
        (float("inf"), 1),
        (float("-inf"), 1),
        ([], 1),
        ({}, 1),
        (np.array([2]), 1),
    ],
)
def test_rollout_metadata_turn_normalization(rollout_module: ModuleType, value: Any, expected: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        metrics = rollout_module._compute_rollout_metadata_stats([Sample(metadata={"rollout_turns": value})])
    assert all(actual == expected for key, actual in metrics.items() if key.startswith("num_turn/"))


def test_rollout_metadata_is_read_only_and_batch_local(rollout_module: ModuleType) -> None:
    samples = _samples()
    samples[0].metadata["nested"] = {"values": [1, 2]}
    snapshots = copy.deepcopy(samples)
    metadata_refs = [sample.metadata for sample in samples]
    expected = rollout_module._compute_rollout_metadata_stats(samples)
    assert rollout_module._compute_rollout_metadata_stats(samples) == expected
    assert rollout_module._compute_rollout_metadata_stats(list(reversed(samples))) == pytest.approx(expected)
    assert samples == snapshots
    assert all(sample.metadata is metadata for sample, metadata in zip(samples, metadata_refs, strict=True))
    next_batch = rollout_module._compute_rollout_metadata_stats([Sample()])
    assert next_batch.keys() == expected.keys()
    assert next_batch["stop_reason/env_done/count"] == 0.0
    assert next_batch["stop_reason/env_done/ratio"] == 0.0
    assert next_batch["stop_reason/unknown/count"] == 1.0
    assert next_batch["stop_reason/unknown/ratio"] == 1.0
    assert expected["stop_reason/env_done/count"] == 2.0


@pytest.mark.parametrize("reason", _EXPECTED_REASONS)
def test_rollout_metadata_known_reasons_preserved(rollout_module: ModuleType, reason: str) -> None:
    metrics = rollout_module._compute_rollout_metadata_stats([Sample(metadata={"stop_reason": reason})])
    assert metrics[f"stop_reason/{reason}/count"] == 1.0
    assert metrics[f"stop_reason/{reason}/ratio"] == 1.0
    assert sum(value for key, value in metrics.items() if key.endswith("/count")) == 1.0
    assert sum(value for key, value in metrics.items() if key.endswith("/ratio")) == 1.0


def test_rollout_metadata_dynamic_reasons_have_bounded_keys(rollout_module: ModuleType) -> None:
    samples = [
        Sample(metadata={"stop_reason": f"env_error:request/{index} timeout https://example.invalid/{index}"})
        for index in range(100)
    ]
    samples += [Sample(metadata={"rollout_stop_reason": f"custom/{index}: " + "x" * 10000}) for index in range(100)]
    samples += [Sample(metadata=None), Sample(metadata={"stop_reason": " "})]
    before = copy.deepcopy(samples)
    metrics = rollout_module.compute_metrics_from_samples(_args(), samples)
    reason_keys = {key for key in metrics if key.startswith("stop_reason/")}
    assert reason_keys == {
        f"stop_reason/{reason}/{statistic}" for reason in _EXPECTED_REASONS for statistic in ("count", "ratio")
    }
    assert len(reason_keys) == 26
    assert metrics["stop_reason/env_error/count"] == 100.0
    assert metrics["stop_reason/other/count"] == 100.0
    assert metrics["stop_reason/unknown/count"] == 2.0
    assert metrics["stop_reason/env_error/ratio"] == pytest.approx(100 / 202)
    assert metrics["stop_reason/other/ratio"] == pytest.approx(100 / 202)
    assert metrics["stop_reason/unknown/ratio"] == pytest.approx(2 / 202)
    assert sum(metrics[key] for key in reason_keys if key.endswith("/count")) == 202
    assert sum(metrics[key] for key in reason_keys if key.endswith("/ratio")) == pytest.approx(1.0)
    assert samples == before


def test_rollout_metadata_total_metrics_preserve_existing_statistics(rollout_module: ModuleType) -> None:
    samples = _samples()
    before = copy.deepcopy(samples)
    metrics = rollout_module.compute_metrics_from_samples(_args(), samples)
    assert {key: metrics[key] for key in _expected()} == pytest.approx(_expected())
    assert metrics["raw_reward"] == 0.5
    assert metrics["truncated_ratio"] == 0.25
    assert metrics["response_len/mean"] == 2.5
    assert metrics["prefix_cache_hit_rate"] == 0.0
    assert metrics["avg_cached_tokens_per_sample"] == 0.0
    assert metrics["image_count/max"] == 0
    assert samples == before


def test_rollout_metadata_partial_staleness_missing_metadata(rollout_module: ModuleType) -> None:
    samples = [Sample(index=0, metadata=None), Sample(index=1, metadata={"start_rollout_id": 5})]
    before = copy.deepcopy(samples)
    metrics = rollout_module.compute_metrics_from_samples(_args(partial_rollout=True), samples, rollout_id=7)
    assert metrics["staleness/avg"] == 1.0
    assert metrics["staleness/min"] == 0
    assert metrics["staleness/max"] == 2
    assert metrics["global_batch_size"] == 2
    assert metrics["stop_reason/unknown/count"] == 2.0
    assert samples == before


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("dynamic_reasons", [False, True])
def test_rollout_metadata_log_publication(
    rollout_module: ModuleType, monkeypatch: pytest.MonkeyPatch, mode: str, dynamic_reasons: bool
) -> None:
    log = Mock()
    flush = Mock()
    monkeypatch.setattr(rollout_module, "save_rollout_result_jsonl", Mock())
    monkeypatch.setattr(rollout_module, "save_eval_summary_jsonl", Mock())
    monkeypatch.setattr(rollout_module.tracking_utils, "log", log)
    monkeypatch.setattr(rollout_module.tracking_utils, "flush_metrics", flush)
    args = _args()
    samples = _samples()
    expected = _expected()
    if dynamic_reasons:
        samples[0].metadata["rollout_stop_reason"] = "env_error:timeout https://example.invalid/request/42"
        samples[1].metadata["stop_reason"] = "custom_reason:request/43"
        expected.update(
            {
                "stop_reason/env_done/count": 0.0,
                "stop_reason/env_done/ratio": 0.0,
                "stop_reason/env_error/count": 1.0,
                "stop_reason/env_error/ratio": 0.25,
                "stop_reason/other/count": 1.0,
                "stop_reason/other/ratio": 0.25,
            }
        )
    if mode == "train":
        rollout_module._log_rollout_data(7, args, samples, {"extra": 2.0}, 2.0)
        prefix, step_key = "rollout/", "rollout/step"
    else:
        rollout_module._log_eval_rollout_data(7, args, {"example": {"rewards": [1.0, 0.0], "samples": samples}})
        prefix, step_key = "eval/example/", "eval/step"
    log.assert_called_once()
    logged = log.call_args.args[1]
    assert {key: logged[prefix + key] for key in expected} == pytest.approx(expected)
    assert {key for key in logged if key.startswith(prefix + "stop_reason/")} == {
        prefix + key for key in expected if key.startswith("stop_reason/")
    }
    assert logged[step_key] == 7
    assert log.call_args.kwargs == {"step_key": step_key}
    flush.assert_called_once_with(args, 7)
    if mode == "train":
        assert logged["perf/rollout_time"] == 2.0
        assert logged["extra"] == 2.0


@pytest.mark.parametrize("turn_count", [1, 3])
@pytest.mark.parametrize(
    ("reason", "category"),
    [("env_done", "env_done"), ("env_error:timeout/request/42", "env_error"), ("request/43", "other")],
)
def test_rollout_metadata_session_forest_export(
    rollout_module: ModuleType, turn_count: int, reason: str, category: str
) -> None:
    output = SessionOutput.from_payload({"metadata": {"stop_reason": reason}})
    forest = SessionForest.create_empty(session_id="metrics", metadata=output.metadata)
    leaf = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "question"}],
        train_token_delta=[1],
        rollout_token_delta=[1],
    )
    for _ in range(turn_count):
        leaf = forest.append_resp(
            parent_state_hash=leaf.state_hash,
            rollout_id=0,
            abort_count=0,
            messages_delta=[{"role": "assistant", "content": "answer"}],
            token_delta=[2],
            logprob_delta=[-0.1],
            status="completed",
        )
    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: "text")
    sample = forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=tokenizer)
    before = copy.deepcopy(sample.metadata)
    assert sample.metadata["rollout_turns"] == sample.metadata["agentic_trace"]["turn_count"] == turn_count
    metrics = rollout_module._compute_rollout_metadata_stats([sample])
    assert metrics["num_turn/p99"] == turn_count
    assert metrics[f"stop_reason/{category}/ratio"] == 1.0
    assert sample.metadata["stop_reason"] == reason
    assert sample.metadata == before
