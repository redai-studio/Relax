# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.weight_update.train_offload import _SelectiveOffloadStrategy


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA D2H/H2D and real Adam state")
def test_gpu_adam_state_restores_across_serial_and_overlapped_transitions(monkeypatch):
    """Compare three updates with an optimizer that never leaves the GPU."""
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    master = torch.nn.Parameter(torch.arange(16, device="cuda", dtype=torch.float32))
    control = torch.nn.Parameter(master.detach().clone())
    optimizer, expected_optimizer = torch.optim.Adam([master]), torch.optim.Adam([control])
    wrapped = object.__new__(DistributedOptimizer)
    wrapped.config = SimpleNamespace(use_precision_aware_optimizer=False)
    wrapped.optimizer = optimizer
    wrapped.shard_fp32_from_float16_groups = [[master]]
    flat = master.detach().to(torch.bfloat16)
    weight = flat.view(4, 4)
    buffer = SimpleNamespace(param_data=flat, grad_data=torch.zeros_like(flat), param_index_map={weight: (0, 16)})
    model = SimpleNamespace(buffers=[buffer], expert_parallel_buffers=[], modules=lambda: [])
    offloader = _SelectiveOffloadStrategy([model], wrapped, SimpleNamespace(optimizer_cpu_offload=False))
    target = torch.empty_like(weight)
    consumer_stream, consumer_done = torch.cuda.Stream(), torch.cuda.Event()
    release = offloader._release_parameter_storage

    def release_after_readers(buffers):
        assert consumer_done.query() and offloader._cpu_backup_done.query()
        release(buffers)

    monkeypatch.setattr(offloader, "_release_parameter_storage", release_after_readers)
    for step, overlap in enumerate((True, False, True)):
        for parameter in (master, control):
            parameter.grad = torch.full_like(parameter, 0.25 + step)
        optimizer.step()
        expected_optimizer.step()
        with torch.no_grad():
            flat.copy_(master)
        torch.cuda.synchronize()
        assert offloader.validate_optimizer_storage_aliases() is None
        expected = flat.cpu().clone()
        offloader.begin_weight_sync({"weight": weight}, overlap=overlap)
        assert flat.untyped_storage().nbytes() > 0
        if not overlap:
            assert offloader._cpu_backup_done.query()
        assert master.device.type == "cpu"
        assert all(optimizer.state[master][name].device.type == "cpu" for name in ("exp_avg", "exp_avg_sq"))
        # Sender ACK waits only for its own reader. finish_weight_sync must
        # independently wait for the D2H reader before releasing shared storage.
        consumer_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(consumer_stream):
            target.copy_(weight)
            consumer_done.record()
        consumer_done.synchronize()
        snapshot = offloader.finish_weight_sync()
        assert flat.untyped_storage().nbytes() == 0
        assert torch.equal(snapshot["weight"].reshape(-1), expected)
        assert torch.equal(target.cpu().reshape(-1), expected)
        offloader.reload()
        assert torch.equal(flat.cpu(), expected)
        assert torch.equal(master, control)
        assert torch.equal(optimizer.state[master]["step"], expected_optimizer.state[control]["step"])
        for name in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(optimizer.state[master][name], expected_optimizer.state[control][name])
