# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rollout statistics must be usable without starting or importing a
runtime."""

import copy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.utils.metrics.rollout_metrics import (
    compute_metrics_from_samples,
    compute_mopd_metrics,
    compute_perf_metrics_from_samples,
)
from relax.utils.types import Sample


def _args(**overrides):
    values = dict(
        advantage_estimator="grpo",
        reward_key=None,
        log_reward_category=None,
        log_passrate=False,
        n_samples_per_prompt=2,
        partial_rollout=False,
        fully_async=False,
        rollout_num_gpus=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_rollout_metrics_and_scale_types_import_without_deployment_dependencies():
    # A fresh process prevents earlier tests' imports/stubs from hiding a
    # dependency. Exercise MOPD too, so a lazy runtime import cannot evade this.
    script = """
import importlib.abc
import sys
from types import SimpleNamespace

class NoRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ray', 'sglang', 'sglang_router', 'transfer_queue', 'megatron'}:
            raise AssertionError(f'Unexpected runtime dependency: {fullname}')

sys.meta_path.insert(0, NoRuntimeImports())
from relax.distributed.ray.rollout_scale_types import ScaleOutRequest, ScaleOutStatus
from relax.utils.metrics.rollout_metrics import compute_metrics_from_samples
from relax.utils.types import Sample

args = SimpleNamespace(advantage_estimator='grpo', reward_key=None,
                       log_reward_category=None, log_passrate=False, opd_teacher_key='source')
sample = Sample(response='ok', response_length=1, reward=1.0,
                metadata={'source': 'fixture'}, teacher_log_probs=[-2.0], rollout_log_probs=[-1.0])
assert compute_metrics_from_samples(args, [sample])['by_source/fixture/logp_gap'] == 1.0
assert ScaleOutRequest('request', ScaleOutStatus.ACTIVE).is_terminal()
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_rollout_metrics_preserve_reward_categories_cache_and_effective_lengths():
    args = _args(reward_key="score", log_reward_category="category", sglang_speculative_algorithm="EAGLE")
    samples = [
        Sample(
            index=1,
            group_index=1,
            response="first",
            response_length=4,
            loss_mask=[1, 0, 1, 0],
            reward={"score": 1.0, "category": "ok", "quality": 0.5},
            status=Sample.Status.COMPLETED,
            multimodal_train_inputs={"image_grid_thw": [[1, 4, 4]]},
            prefix_cache_info=Sample.PrefixCacheInfo(cached_tokens=4, total_prompt_tokens=8),
            spec_info=Sample.SpecInfo(spec_accept_token_num=2, spec_draft_token_num=4),
        ),
        Sample(
            index=2,
            group_index=1,
            response="second",
            response_length=4,
            reward={"score": 0.0, "quality": 1.0},
            status=Sample.Status.TRUNCATED,
            prefix_cache_info=Sample.PrefixCacheInfo(cached_tokens=2, total_prompt_tokens=4),
            spec_info=Sample.SpecInfo(spec_accept_token_num=3, spec_draft_token_num=4),
        ),
        Sample(index=3, response="unscored", response_length=3, reward=None),
    ]
    before = copy.deepcopy(samples)

    metrics = compute_metrics_from_samples(args, samples)

    assert metrics["response_len/mean"] == 3.0
    assert metrics["response_len/min"] == 2
    assert metrics["response_len/max"] == 4
    assert metrics["raw_reward"] == 0.5
    assert metrics["quality/mean"] == 0.75
    assert metrics["error_cat/ok"] == 1.0
    assert metrics["truncated_ratio"] == pytest.approx(1 / 3)
    assert metrics["repetition_frac"] == 0.0
    assert metrics["prefix_cache_hit_rate"] == 0.5
    assert metrics["avg_cached_tokens_per_sample"] == 2.0
    # The unscored sample has no counter pair; do not fabricate a zero ratio.
    assert "spec_accept_rate" not in metrics
    assert metrics["image_count/mean"] == pytest.approx(1 / 3)
    assert metrics["multimodal_token_count/max"] == 4
    assert samples == before


def test_rollout_metrics_unrewarded_samples_keep_non_reward_statistics():
    metrics = compute_metrics_from_samples(_args(log_reward_category="category"), [Sample(response_length=2)])
    assert metrics["response_len/mean"] == 2
    assert metrics["prefix_cache_hit_rate"] == 0.0
    assert "raw_reward" not in metrics
    assert not any(key.startswith(("zero_std/", "error_cat/", "by_source/")) for key in metrics)


def test_rollout_metrics_rloo_eval_omits_training_diagnostics():
    samples = [
        Sample(group_index=1, reward=0.0, response_length=2),
        Sample(group_index=1, reward=2.0, response_length=2),
    ]
    args = _args(advantage_estimator="rloo")

    training = compute_metrics_from_samples(args, samples)
    evaluation = compute_metrics_from_samples(args, samples, include_rloo_diagnostics=False)

    assert training["rloo/baseline_mean"] == 1.0
    assert training["rloo/adv_abs_mean"] == 2.0
    assert evaluation == {key: value for key, value in training.items() if not key.startswith("rloo/")}


def test_rollout_metrics_partial_staleness_counts_unique_samples():
    samples = [
        Sample(index=7, response_length=2, reward=1.0, metadata={"start_rollout_id": 2}),
        Sample(index=7, response_length=2, reward=1.0, metadata={"start_rollout_id": 4}),
        Sample(index=8, response_length=2, reward=1.0),
    ]
    metrics = compute_metrics_from_samples(_args(partial_rollout=True), samples, rollout_id=5)
    assert metrics["staleness/avg"] == pytest.approx(4 / 3)
    assert metrics["staleness/max"] == 3
    assert metrics["staleness/min"] == 0
    assert metrics["global_batch_size"] == 2
    assert metrics["zero_std/count_1.0"] == 1

    asynchronous = compute_metrics_from_samples(_args(partial_rollout=True, fully_async=True), samples, rollout_id=5)
    assert not any(key.startswith("staleness/") for key in asynchronous)
    assert "global_batch_size" not in asynchronous


def test_rollout_metrics_performance_uses_raw_and_effective_tokens():
    samples = [
        Sample(response_length=4, loss_mask=[1, 0, 1, 0], non_generation_time=1.0),
        Sample(response_length=2, non_generation_time=2.0),
    ]
    metrics = compute_perf_metrics_from_samples(_args(), samples, rollout_time=4.0)

    assert metrics["tokens_per_gpu_per_sec"] == 0.75
    assert metrics["effective_tokens_per_gpu_per_sec"] == 0.5
    assert metrics["longest_sample_tokens_per_sec"] == 1.0
    assert metrics["longest_sample_non_generation_time"] == 1.0
    assert metrics["longest_sample_tokens_per_sec_without_non_generation"] == pytest.approx(4 / 3)
    assert metrics["longest_effective_sample_non_generation_time"] == 1.5
    assert metrics["longest_effective_sample_tokens_per_sec_without_non_generation"] == 0.8

    no_wait = compute_perf_metrics_from_samples(_args(rollout_num_gpus=0), [Sample(response_length=4)], 2.0)
    assert no_wait == {
        "rollout_time": 2.0,
        "longest_sample_tokens_per_sec": 2.0,
        "longest_effective_sample_tokens_per_sec": 2.0,
    }


def test_rollout_metrics_mopd_preserves_per_sample_logprob_pairing():
    samples = [
        Sample(
            reward=1.0, metadata={"source": "math/qa-v1"}, teacher_log_probs=[-2.0, -4.0], rollout_log_probs=[-1.0]
        ),
        Sample(reward=0.0, metadata={"source": "math/qa-v1"}, teacher_log_probs=[-10.0]),
        Sample(reward=0.5, metadata={"source": "math/qa-v1"}, rollout_log_probs=[-20.0]),
        Sample(reward=0.0),
    ]
    metrics = compute_mopd_metrics(_args(opd_teacher_key="source"), samples)
    assert metrics == {
        "by_source/math_qa_v1/accuracy": pytest.approx(1 / 3),
        "by_source/math_qa_v1/mean_reward": 0.5,
        "by_source/math_qa_v1/teacher_logp": -6.5,
        "by_source/math_qa_v1/student_logp": -10.5,
        "by_source/math_qa_v1/logp_gap": 2.0,
        "by_source/math_qa_v1/rkl_approx": 1.0,
        "by_source/unknown/accuracy": 0.0,
        "by_source/unknown/mean_reward": 0.0,
    }
    assert compute_mopd_metrics(_args(), samples) == {}
    assert compute_mopd_metrics(_args(opd_teacher_key="source"), []) == {}


def test_rollout_metrics_mopd_legacy_import_is_the_same_function():
    from relax.utils.opd.opd_utils import compute_mopd_metrics as legacy

    assert legacy is compute_mopd_metrics
