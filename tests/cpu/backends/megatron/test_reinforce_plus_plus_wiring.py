# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import torch
import torch.distributed as dist


def test_compute_advantages_and_returns_wires_baseline_contract(monkeypatch, tmp_path):
    try:
        import megatron  # noqa: F401
    except ModuleNotFoundError:
        megatron = ModuleType("megatron")
        megatron_core = ModuleType("megatron.core")
        megatron_core.mpu = SimpleNamespace()
        megatron.core = megatron_core
        monkeypatch.setitem(sys.modules, "megatron", megatron)
        monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)

    from relax.backends.megatron import cp_utils

    loss_module = importlib.import_module("relax.backends.megatron.loss")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_path / 'wiring-gloo-init'}",
        rank=0,
        world_size=1,
    )
    try:
        fake_mpu = SimpleNamespace(
            is_pipeline_last_stage=lambda: True,
            get_context_parallel_world_size=lambda: 1,
            get_data_parallel_group=lambda: dist.group.WORLD,
        )
        monkeypatch.setattr(loss_module, "mpu", fake_mpu)
        monkeypatch.setattr(cp_utils, "mpu", fake_mpu)

        args = Namespace(
            use_rollout_logprobs=False,
            qkv_format="thd",
            is_vl_model=False,
            uses_unsplit_forward=False,
            kl_coef=0.0,
            kl_loss_type="k2",
            advantage_estimator="reinforce_plus_plus_baseline",
            gamma=1.0,
            use_opd=False,
            normalize_advantages=True,
            dynamic_context_parallel=False,
        )
        rollout_data = {
            "log_probs": [torch.zeros(2), torch.zeros(2)],
            "ref_log_probs": None,
            "rewards": [1.0, -1.0],
            "values": None,
            "response_lengths": [2, 2],
            "loss_masks": [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])],
            "total_lengths": [2, 2],
        }

        loss_module.compute_advantages_and_returns(args, rollout_data)

        expected_raw = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, -1.0])]
        expected_normalized = [
            torch.tensor([2.0**0.5, 0.0]),
            torch.tensor([-(0.5**0.5), -(0.5**0.5)]),
        ]
        for actual, expected in zip(rollout_data["returns"], expected_raw, strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        for actual, expected in zip(rollout_data["advantages"], expected_normalized, strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

        assert rollout_data["reinforce_pp_valid_token_count"][0].item() == 3
        assert rollout_data["reinforce_pp_zero_variance"][0].item() == 0
        assert isinstance(rollout_data["reinforce_pp_advantage_raw_mean"][0], torch.Tensor)
    finally:
        dist.destroy_process_group()
        sys.modules.pop("relax.backends.megatron.loss", None)
