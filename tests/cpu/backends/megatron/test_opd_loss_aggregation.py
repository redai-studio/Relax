# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for OPD loss aggregation semantics (token-mean only)."""

import importlib
import sys
from types import ModuleType

import pytest


def _install_fake_megatron(monkeypatch):
    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    mpu = ModuleType("megatron.core.mpu")

    mpu.get_context_parallel_world_size = lambda: 1
    core.mpu = mpu
    megatron.core = core

    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu)


@pytest.fixture()
def opd_utils_module(monkeypatch):
    torch = pytest.importorskip("torch", exc_type=ImportError)
    _install_fake_megatron(monkeypatch)
    sys.modules.pop("relax.utils.opd.opd_utils", None)
    module = importlib.import_module("relax.utils.opd.opd_utils")
    yield module, torch
    sys.modules.pop("relax.utils.opd.opd_utils", None)


def test_opd_loss_token_mean(opd_utils_module):
    opd_utils, torch = opd_utils_module
    values = torch.tensor([1.0, 1.0, 10.0, 6.0])
    batch = {
        "total_lengths": [4, 5],
        "response_lengths": [2, 2],
        "loss_masks": [
            torch.tensor([1.0, 1.0]),
            torch.tensor([1.0, 0.0]),
        ],
    }

    # token_mean: (1+1+10+0) / (1+1+1+0) = 12 / 3
    assert torch.isclose(opd_utils.reduce_opd_loss(batch, values), torch.tensor(12.0 / 3.0))
