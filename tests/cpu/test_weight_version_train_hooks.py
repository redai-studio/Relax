# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for consumed weight version stamping in train hooks."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from relax.utils.agentic_obs import train_hooks
from relax.utils.agentic_obs.config import resolve_agentic_obs_config
from relax.utils.agentic_obs.writers import close_observation_writers


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class _FakeMpu:
    """Minimal megatron.core.mpu stand-in for the emit path."""

    def __init__(
        self, *, tp_rank: int = 0, last_stage: bool = True, cp_size: int = 1, cp_rank: int = 0, dp_rank: int = 0
    ) -> None:
        self._tp_rank = tp_rank
        self._last_stage = last_stage
        self._cp_size = cp_size
        self._cp_rank = cp_rank
        self._dp_rank = dp_rank

    def get_tensor_model_parallel_rank(self) -> int:
        return self._tp_rank

    def is_pipeline_last_stage(self, ignore_virtual=False) -> bool:
        return self._last_stage

    def get_context_parallel_world_size(self) -> int:
        return self._cp_size

    def get_context_parallel_rank(self) -> int:
        return self._cp_rank

    def get_data_parallel_rank(self, with_context_parallel=False) -> int:
        return self._dp_rank

    def get_data_parallel_world_size(self, with_context_parallel=False) -> int:
        return 1


@pytest.fixture
def fake_mpu(monkeypatch):
    """Install a fake ``megatron.core`` so the emit path runs off-GPU."""

    def install(**kwargs) -> _FakeMpu:
        mpu = _FakeMpu(**kwargs)
        megatron = sys.modules.get("megatron") or SimpleNamespace()
        core = SimpleNamespace(mpu=mpu)
        monkeypatch.setitem(sys.modules, "megatron", megatron)
        monkeypatch.setitem(sys.modules, "megatron.core", core)
        monkeypatch.setattr(megatron, "core", core, raising=False)
        return mpu

    from relax.utils.agentic_obs import train_hooks

    train_hooks.close_train_appenders()
    close_observation_writers()
    monkeypatch.setattr(train_hooks, "_UPDATE_SEQUENCES", {})
    yield install
    train_hooks.close_train_appenders()
    close_observation_writers()


def _rollout_data(export_ids: list[int], **overrides: Any) -> dict[str, Any]:
    count = len(export_ids)
    data = {
        "sample_export_ids": [torch.tensor(export_id) for export_id in export_ids],
        "agentic_group_ids": [f"group-{export_id}" for export_id in export_ids],
        "response_lengths": [4] * count,
        "total_lengths": [6] * count,
        "loss_masks": [torch.ones(4, dtype=torch.int) for _ in range(count)],
        "advantages": [torch.full((4,), 0.5) for _ in range(count)],
        "returns": [torch.full((4,), 2.0) for _ in range(count)],
    }
    data.update(overrides)
    return data


def _train_args(tmp_path: Path, **overrides: Any) -> Namespace:
    args = {
        "dump_agentic_obs": str(tmp_path),
        "agentic_obs_run_id": "run_test",
        "agentic_obs_train_generation_id": "generation",
        "dump_agentic_obs_media": False,
        "dynamic_context_parallel": False,
        "context_parallel_size": 1,
        "n_samples_per_prompt": 1,
        "rollout_batch_size": 1,
        "advantage_estimator": "grpo",
        "agentic_custom_advantage_path": None,
        "reward_key": None,
    }
    args.update(overrides)
    return Namespace(**args)


def _emit(args, rollout_id, data, step_id=0, consumed_weight_version=None):
    with train_hooks.observe_train_step(args, rollout_id, step_id, consumed_weight_version=consumed_weight_version):
        train_hooks.capture_training_input(data)
        train_hooks.record_optimizer_result(successful=True)


def _read_train_samples(path):
    return [sample for record in _read_jsonl(path) for sample in record["samples"]]


def test_train_record_stamps_consumed_weight_version_per_sample(tmp_path, fake_mpu):
    fake_mpu()
    args = _train_args(tmp_path)
    _emit(args, 3, _rollout_data([1, 2]), step_id=2, consumed_weight_version="7")
    train_hooks.close_train_appenders()
    close_observation_writers()
    samples = _read_train_samples(resolve_agentic_obs_config(args).train_scalar_path(3, 0))
    assert all(s["consumed_weight_version"] == "7" for s in samples)
    assert samples[0]["sample_export_id"] == "1"


def test_train_record_omits_version_when_not_provided(tmp_path, fake_mpu):
    # 未传（默认 None）时显式写 null——离线工具把这类样本计为消费点未知。
    fake_mpu()
    args = _train_args(tmp_path)
    _emit(args, 3, _rollout_data([1]))
    train_hooks.close_train_appenders()
    close_observation_writers()
    samples = _read_train_samples(resolve_agentic_obs_config(args).train_scalar_path(3, 0))
    assert samples[0]["consumed_weight_version"] is None
