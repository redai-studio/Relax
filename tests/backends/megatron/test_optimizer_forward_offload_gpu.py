# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise Relax's iteration boundary with real TE Adam and asynchronous CPU
offload.

This single-GPU check verifies optimizer state and allocator residency. It does
not substitute for distributed parameter-gather or whole-device memory tests.
"""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Optimizer DMA and TE Adam require CUDA")
@pytest.mark.parametrize("chunk_size_bytes", [0, 1024 * 1024], ids=["full_window", "multiple_chunks"])
def test_forward_offload_preserves_adam_updates_and_releases_master_memory(
    monkeypatch: pytest.MonkeyPatch, record_property, chunk_size_bytes: int
) -> None:
    offload_module = pytest.importorskip(
        "megatron.core.optimizer.cpu_offloading.chunked_optimizer_state_offload", exc_type=ImportError
    )
    te_optimizers = pytest.importorskip("transformer_engine.pytorch.optimizers", exc_type=ImportError)
    from megatron.core.optimizer.optimizer import MegatronOptimizer

    from relax.backends.megatron import model as module

    parameters = [
        torch.nn.Parameter(torch.linspace(-0.75, 0.75, count, device="cuda", dtype=torch.bfloat16))
        for count in (2**20, 2**18)
    ]
    reference_parameters = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    identity = [(id(parameter), parameter.data_ptr()) for parameter in parameters]
    optimizer_options = dict(
        lr=0.003,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.1,
        master_weights=True,
        master_weight_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        store_param_remainders=False,
        use_decoupled_grad=True,
    )
    inner = te_optimizers.FusedAdam(parameters, **optimizer_options)
    reference = te_optimizers.FusedAdam(reference_parameters, **optimizer_options)
    offloader = offload_module.ChunkedOptimizerStateOffloader(
        optimizer=inner,
        master_params=[],
        chunk_size_bytes=chunk_size_bytes,
        offload_fraction=1.0,
        state_dtypes=(torch.float32, torch.float32),
        optimizer_owned_master_dtypes={parameter: torch.float32 for parameter in parameters},
    )
    args = Namespace(
        empty_unused_memory_level=0,
        custom_megatron_before_train_step_hook_path=None,
        ci_test=False,
        seq_length=8,
        micro_batch_size=1,
        decoder_seq_length=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
    )
    monkeypatch.setattr(module, "get_args", lambda: args)
    monkeypatch.setattr(module.capture_hooks, "begin_step_for", Mock())
    monkeypatch.setattr(module.capture_hooks, "end_step_for", Mock())
    monkeypatch.setattr(module, "maybe_verify_critic_value_head_movement", Mock())
    monkeypatch.setattr(module.mpu, "is_pipeline_last_stage", lambda **_: False)
    monkeypatch.setattr(module, "_is_global_zero_token_step", lambda _: False)

    released_master_bytes = []
    iteration = 0
    forward_count = 0

    def offload(*, offload_master: bool) -> None:
        assert offload_master
        if iteration != 3:
            # Preserve real DMA overlap with forward on the earlier iterations.
            MegatronOptimizer.offload_optimizer_state_for_forward(wrapped, offload_master=offload_master)
            return
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        MegatronOptimizer.offload_optimizer_state_for_forward(wrapped, offload_master=offload_master)
        # Explicit synchronization is confined to this measurement/test. The
        # production lifecycle keeps DMA asynchronous with its own stream events.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated()
        released_master_bytes.append(before - after)

    def forward_backward(**_) -> list:
        nonlocal forward_count
        forward_count += 1
        for index, (parameter, expected) in enumerate(zip(parameters, reference_parameters, strict=True)):
            assert parameter.is_cuda
            if iteration:
                state = inner.state[parameter]
                assert state["master_param"].device.type == "cpu"
                assert state["master_param"].is_pinned()
                assert state["exp_avg"].device.type == "cpu"
                assert state["exp_avg_sq"].device.type == "cpu"
            # Changing gradients, LR and weight decay make stale state or an
            # extra/missing optimizer update visible across several iterations.
            parameter.decoupled_grad = parameter.detach().float() * (iteration + 1) + 0.125 * (index + 1)
            expected.decoupled_grad = expected.detach().float() * (iteration + 1) + 0.125 * (index + 1)
        return []

    def optimizer_step() -> tuple[bool, float, int]:
        offloader.step()
        return True, 1.0, 0

    def advance_lr(**_) -> None:
        inner.param_groups[0]["lr"] *= 0.9

    wrapped = SimpleNamespace(
        config=SimpleNamespace(chunked_optimizer_state_offload=True, optimizer_state_offload_fraction=1.0),
        _optimizer_state_offloader=offloader,
        offload_optimizer_state_for_forward=offload,
        optimizer_state_offload_requires_pre_forward_param_sync=lambda: False,
        zero_grad=inner.zero_grad,
        step=optimizer_step,
    )
    scheduler = SimpleNamespace(step=advance_lr)
    model = SimpleNamespace(zero_grad_buffer=Mock())
    monkeypatch.setattr(module, "get_forward_backward_func", lambda: forward_backward)

    for iteration in range(4):
        reference.zero_grad()
        assert module.train_one_step(args, 0, iteration, [], [model], wrapped, scheduler, 1, 8) == ({}, 1.0)
        reference.step()
        reference.param_groups[0]["lr"] *= 0.9
        torch.cuda.synchronize()
        for parameter, expected in zip(parameters, reference_parameters, strict=True):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
            for key in ("master_param", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    inner.state[parameter][key].detach().cpu(),
                    reference.state[expected][key].detach().cpu(),
                    rtol=0,
                    atol=0,
                )
        assert inner.param_groups[0]["step"] == reference.param_groups[0]["step"] == iteration + 1
        assert inner.param_groups[0]["lr"] == reference.param_groups[0]["lr"]
        assert [(id(parameter), parameter.data_ptr()) for parameter in parameters] == identity

    expected_master_bytes = sum(parameter.numel() * torch.float32.itemsize for parameter in parameters)
    assert forward_count == 4
    assert len(released_master_bytes) == 1
    assert min(released_master_bytes) >= expected_master_bytes * 0.95
    record_property("expected_master_bytes", expected_master_bytes)
    record_property("minimum_released_allocated_bytes", min(released_master_bytes))
