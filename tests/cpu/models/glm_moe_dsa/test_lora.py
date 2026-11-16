# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Cover absorbed LoRA gradients, including a real two-rank CPU TP group.

Existing generic merge tests cover export values, not the absorbed training
backward. The small Gloo test checks both adapters against an unsharded oracle.
"""

import importlib.util
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


def _wrapper(base, a, b, group=None, dropout=0.0):
    return SimpleNamespace(
        to_wrap=SimpleNamespace(weight=base, layer_norm_weight=torch.ones(base.shape[1])),
        adapter=SimpleNamespace(
            linear_in=SimpleNamespace(weight=a),
            linear_out=SimpleNamespace(weight=b),
            dim=b.shape[1],
            alpha=2 * b.shape[1],
            tp_group=group,
            dropout=nn.Dropout(dropout),
        ),
        training=True,
        _adapter_enabled=True,
    )


def test_glm_absorbed_lora_weight_and_gradient(glm_modules):
    helpers = glm_modules("lora")
    torch.manual_seed(7)
    base = torch.randn(6, 4)
    a = torch.randn(2, 4, requires_grad=True)
    b = torch.randn(6, 2, requires_grad=True)
    wrapper = _wrapper(base, a, b)
    x = torch.randn(3, 4, requires_grad=True)
    actual = x @ helpers.absorbed_kv_weight(wrapper).T
    expected = x @ base.T + 2 * (x @ a.T) @ b.T
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.square().sum(), (a, b, x), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), (a, b, x))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert helpers.fused_norm_weight(wrapper) is wrapper.to_wrap.layer_norm_weight
    wrapper._adapter_enabled = False
    assert helpers.absorbed_kv_weight(wrapper) is base
    assert helpers.absorbed_kv_weight(wrapper.to_wrap) is base


def test_glm_absorbed_lora_rejects_training_dropout(glm_modules):
    helpers = glm_modules("lora")
    wrapper = _wrapper(torch.zeros(6, 4), torch.ones(2, 4), torch.ones(6, 2), dropout=0.1)
    with pytest.raises(ValueError, match="lora-dropout 0"):
        helpers.absorbed_kv_weight(wrapper)
    wrapper.training = False
    torch.testing.assert_close(helpers.absorbed_kv_weight(wrapper), torch.full((6, 4), 4.0))


def _tp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location(
        "glm_lora_under_test", Path(__file__).resolve().parents[4] / "relax/models/glm_moe_dsa/lora.py"
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    group = dist.new_group([0, 1], backend="gloo", timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(13)
        base = torch.randn(6, 4, dtype=torch.float64)
        a = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
        b = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
        a_local = a.detach().chunk(2, dim=0)[rank].clone().requires_grad_()
        b_local = b.detach().chunk(2, dim=0)[rank].clone().requires_grad_()
        wrapper = _wrapper(base.chunk(2, dim=0)[rank], a_local, b_local, group)
        effective = helpers.absorbed_kv_weight(wrapper)
        expected = base + 2 * b @ a
        torch.testing.assert_close(effective, expected.chunk(2, dim=0)[rank])
        effective.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(a_local.grad, a.grad.chunk(2, dim=0)[rank])
        torch.testing.assert_close(b_local.grad, b.grad.chunk(2, dim=0)[rank])
        assert a_local.grad.abs().sum() > 0
    finally:
        dist.destroy_process_group(group)
        dist.destroy_process_group()


def test_glm_absorbed_lora_tp_gradient_matches_unsharded(tmp_path):
    processes = mp.spawn(_tp_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=False)
    try:
        # Each rank also has a 30-second collective timeout; cap overall CI time.
        for _ in range(6):
            if processes.join(timeout=10):
                return
        pytest.fail("GLM LoRA CPU TP gradient test exceeded 60 seconds")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
