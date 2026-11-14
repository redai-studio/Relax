# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check real mixed CPU/GPU Adam updates with a delayed gradient producer."""

import copy

import pytest
import torch


def _check_hybrid_stream_updates(monkeypatch: pytest.MonkeyPatch, *, patched: bool) -> None:
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
    from transformer_engine.pytorch.optimizers import FusedAdam

    from relax.backends.megatron.compat import patch_hdo_gpu_gradient_stream

    monkeypatch.setattr(
        HybridDeviceOptimizer,
        "_register_param_copy_back_gpu_hook",
        HybridDeviceOptimizer._register_param_copy_back_gpu_hook,
    )
    if patched:
        assert patch_hdo_gpu_gradient_stream()
        assert not patch_hdo_gpu_gradient_stream()

    parameters = [
        torch.nn.Parameter(torch.linspace(-0.5, 0.5, 4096, device="cuda", dtype=torch.bfloat16)) for _ in range(2)
    ]
    expected = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    options = dict(
        offload_fraction=0.5,
        param_update_in_fp32=True,
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=FusedAdam,
        lr=0.01,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    reference = HybridDeviceOptimizer(expected, overlap_cpu_optimizer_d2h_h2d=False, **options)
    actual = HybridDeviceOptimizer(parameters, overlap_cpu_optimizer_d2h_h2d=True, **options)
    assert actual.gpu_optimizer is not None and actual.cpu_optimizers
    original_prepare = actual._set_sub_optimizer_grads

    def delayed_prepare() -> None:
        # Keep the actual BF16 -> FP32 conversion behind GPU work. Without
        # the fence, Adam on the consumer stream reads an unfinished cast.
        torch.cuda._sleep(100_000_000)
        original_prepare()

    for step in range(4):
        if step == 2:
            # Loading rebuilds the GPU child and its hooks in HDO.
            actual.load_state_dict(copy.deepcopy(actual.state_dict()))
            reference.load_state_dict(copy.deepcopy(reference.state_dict()))
        if step == 1:
            # Warm up allocations and Adam state before injecting the delay.
            monkeypatch.setattr(actual, "_set_sub_optimizer_grads", delayed_prepare)
        for index, (parameter, target) in enumerate(zip(parameters, expected, strict=True)):
            grad = torch.full_like(parameter, (-1) ** (step + index) * (step + 1) / 8)
            parameter.decoupled_grad = grad
            target.decoupled_grad = grad.clone()
        reference.step()
        actual.step()
        torch.cuda.synchronize()
        for parameter, target in zip(parameters, expected, strict=True):
            torch.testing.assert_close(parameter, target, rtol=0, atol=0)
            for key in ("master_param", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(actual.state[parameter][key], reference.state[target][key], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HDO gradient stream ordering requires CUDA")
def test_hybrid_optimizer_waits_for_gpu_gradient_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    _check_hybrid_stream_updates(monkeypatch, patched=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HDO native FP32 gradient handoff requires CUDA")
def test_hybrid_optimizer_updates_native_fp32_on_cpu_and_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
    from transformer_engine.pytorch.optimizers import FusedAdam

    from relax.backends.megatron.compat import patch_hdo_gpu_gradient_stream, patch_hdo_native_fp32_gradients

    for name in ("_register_param_copy_back_gpu_hook", "_set_sub_optimizer_grads"):
        monkeypatch.setattr(HybridDeviceOptimizer, name, getattr(HybridDeviceOptimizer, name))
    patch_hdo_gpu_gradient_stream()
    assert patch_hdo_native_fp32_gradients()
    assert not patch_hdo_native_fp32_gradients()

    # The large middle parameter crosses the 90% offload threshold, leaving
    # the final native FP32 parameter on GPU, as can happen with KDA gates.
    parameters = [
        torch.nn.Parameter(torch.linspace(-0.5, 0.5, 128, device="cuda", dtype=torch.float32)),
        torch.nn.Parameter(torch.ones(4096, device="cuda", dtype=torch.bfloat16)),
        torch.nn.Parameter(torch.linspace(-0.5, 0.5, 128, device="cuda", dtype=torch.float32)),
    ]
    expected = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    options = dict(
        offload_fraction=0.9,
        param_update_in_fp32=True,
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=FusedAdam,
        lr=0.01,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    actual = HybridDeviceOptimizer(parameters, overlap_cpu_optimizer_d2h_h2d=True, **options)
    reference = HybridDeviceOptimizer(expected, overlap_cpu_optimizer_d2h_h2d=False, **options)
    assert parameters[0] in actual.gpu_params_map_cpu_copy
    assert actual.param_to_inner_param[parameters[-1]] is parameters[-1]
    assert not actual.gpu_optimizer.use_decoupled_grad

    for step in range(3):
        for index, (parameter, target) in enumerate(zip(parameters, expected, strict=True)):
            grad = torch.full_like(parameter, (-1) ** step * (step + 1) / 8)
            # A missing GPU gradient must clear the previous step's .grad.
            if step == 2 and index == 2:
                grad = None
            parameter.decoupled_grad = grad
            target.grad = grad.clone() if grad is not None else None
        reference.step()
        actual.step()
        torch.cuda.synchronize()
        for parameter, target in zip(parameters, expected, strict=True):
            torch.testing.assert_close(parameter, target, rtol=0, atol=0)
            for key in ("master_param", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(actual.state[parameter][key], reference.state[target][key], rtol=0, atol=0)
