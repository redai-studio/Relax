# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.entrypoints.repetition import diagnose_dumps
from relax.utils.metrics.metric_utils import compression_ratio, has_repetition
from relax.utils.metrics.rollout_metrics import compute_metrics_from_samples
from relax.utils.training.train_dump_utils import save_rollout_result_jsonl
from relax.utils.types import Sample


def _args(**kwargs):
    return SimpleNamespace(
        **{
            "log_reward_category": None,
            "advantage_estimator": "grpo",
            "log_passrate": False,
            "reward_key": None,
            "partial_rollout": False,
            **kwargs,
        }
    )


@pytest.mark.parametrize("algorithm", ["zlib", "gzip", "bz2", "lzma"])
def test_repetition_legacy_compression_api_preserves_byte_semantics(algorithm):
    import importlib

    module = importlib.import_module(algorithm)
    raw = ("中文 text " * 100).encode("utf-8")
    kwargs = {"level": 9} if algorithm == "zlib" else {"preset": 9} if algorithm == "lzma" else {"compresslevel": 9}
    compressed = module.compress(raw, **kwargs)
    ratio, savings = compression_ratio(raw, algorithm=algorithm)
    assert ratio == len(raw) / len(compressed)
    assert savings == 100 * (1 - len(compressed) / len(raw))
    assert compression_ratio(raw.decode("utf-8"), algorithm=algorithm) == (ratio, savings)


def test_repetition_production_metrics_and_real_dump_agree(tmp_path):
    fixture = Path(__file__).parents[1] / "fixtures" / "repetition" / "middle_repetition.jsonl"
    records = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    middle = records[0]["response"]
    # The original suffix-only detector misses precisely this committed case.
    assert len(middle) > 10000
    assert compression_ratio(middle[-10000:])[0] <= 10
    assert has_repetition(middle)
    samples = [
        Sample(index=i, group_index=0, response=response, response_length=1, reward=float(i))
        for i, response in enumerate([middle, middle[-10000:], ""])
    ]
    args = _args(rollout_result_dir=str(tmp_path / "dumps"))
    metrics = compute_metrics_from_samples(args, samples)
    assert metrics["repetition_frac"] == pytest.approx(1 / 3)
    assert metrics["raw_reward"] == 1.0
    save_rollout_result_jsonl(args, 4, samples)
    summary = diagnose_dumps([tmp_path / "dumps"], tmp_path / "report.json")
    assert summary["repetition_frac"] == metrics["repetition_frac"]
    assert [s.reward for s in samples] == [0.0, 1.0, 2.0]
    assert samples[0].response == middle


@pytest.mark.parametrize("estimator", ["ppo", "grpo", "rloo"])
def test_repetition_metrics_count_samples_not_windows_or_rewarded_subset(estimator):
    samples = [
        Sample(response="abc" * 20000, response_length=2, reward=None, group_index=0),
        Sample(response="normal response", response_length=1, reward=1.0, group_index=0),
    ]
    metrics = compute_metrics_from_samples(_args(advantage_estimator=estimator, n_samples_per_prompt=2), samples)
    assert metrics["repetition_frac"] == 0.5
    assert metrics["raw_reward"] == 1.0


def test_repetition_distributed_logger_uses_production_aggregation_when_stack_available(tmp_path, monkeypatch):
    # This optional wiring check does not substitute the detector, aggregation,
    # or writer. The CPU production-aggregation tests above always execute.
    for dependency in ("ray", "sglang", "transfer_queue", "megatron.core"):
        pytest.importorskip(dependency, reason="Full Ray rollout logger requires the training image")
    from relax.distributed.ray import rollout

    assert rollout.compute_metrics_from_samples is compute_metrics_from_samples
    captured = []
    monkeypatch.setattr(rollout.tracking_utils, "log", lambda args, data, **kwargs: captured.append(data))
    monkeypatch.setattr(rollout.tracking_utils, "flush_metrics", lambda *args: None)
    args = _args(
        rollout_result_dir=str(tmp_path),
        custom_rollout_log_function_path=None,
        load_debug_rollout_data=None,
        rollout_num_gpus=0,
        wandb_always_use_train_step=False,
    )
    samples = [
        Sample(response="x" * 20000, response_length=1, reward=1.0),
        Sample(response="ok", response_length=1, reward=0.0),
    ]
    rollout._log_rollout_data(0, args, samples, {}, 1.0)
    assert captured[0]["rollout/repetition_frac"] == 0.5
    assert (tmp_path / "train" / "0.jsonl").exists()
