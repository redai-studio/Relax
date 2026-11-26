# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.weight_update.train_offload import _SelectiveOffloadStrategy, warm_cpu_gradient_allocator


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires real initial parameter H2D and retained source reads"
)
def test_initial_sync_restores_only_parameters_and_reuses_completed_cpu_snapshot(monkeypatch):
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    flat = torch.arange(16, device="cuda", dtype=torch.bfloat16)
    weight, standalone = flat.view(4, 4), torch.arange(5, device="cuda", dtype=torch.float32)
    grad = torch.ones_like(flat, dtype=torch.float32)
    buffer = SimpleNamespace(param_data=flat, grad_data=grad, param_index_map={weight: (0, 16)})
    group = SimpleNamespace(param_gather_handle=None, grad_reduce_handle=None)
    model = SimpleNamespace(
        buffers=[buffer],
        expert_parallel_buffers=[],
        modules=lambda: [],
        ddp_config=SimpleNamespace(use_distributed_optimizer=True),
        bucket_groups=[group],
        expert_parallel_bucket_groups=[],
    )
    optimizer = SimpleNamespace(optimizer=object.__new__(HybridDeviceOptimizer))
    offloader = _SelectiveOffloadStrategy([model], optimizer, SimpleNamespace(optimizer_cpu_offload=True))
    weights = dict(weight=weight, alias=weight, persistent=standalone)
    snapshot = {name: tensor.cpu() for name, tensor in weights.items()}
    assert not offloader.can_release_initial_gradients(dict(weights, aliased_buffer=grad.view(4, 4)))
    group.param_gather_handle = object()
    assert not offloader.can_release_initial_gradients(weights)
    group.param_gather_handle = None
    assert offloader.can_release_initial_gradients(weights)
    torch.cuda.synchronize()
    offloader._offload_training_state(retain_parameters=True)
    assert buffer.grad_data.untyped_storage().nbytes() == 0
    assert all(torch.equal(value.cpu(), snapshot[name]) for name, value in weights.items())
    offloader.offload()
    assert flat.untyped_storage().nbytes() == buffer.grad_data.untyped_storage().nbytes() == 0
    assert offloader.initial_weight_sync_supported(weights, snapshot)
    grad = buffer.grad_data
    buffer.grad_data = flat
    assert not offloader.initial_weight_sync_supported(weights, snapshot)
    buffer.grad_data = grad
    assert not offloader.initial_weight_sync_supported(weights, {"weight": snapshot["weight"]})
    cpu_backing, owner = flat.cpu_data.data_ptr(), flat.untyped_storage()._cdata

    def no_second_backup(*args, **kwargs):
        raise AssertionError("Initial sync must reuse the completed CPU snapshot")

    monkeypatch.setattr(offloader, "_copy_parameters_to_cpu", no_second_backup)
    offloader.begin_weight_sync(weights, overlap=True, initial_snapshot=snapshot)
    assert flat.untyped_storage()._cdata == owner and weight.data_ptr() == weights["alias"].data_ptr()
    assert buffer.grad_data.untyped_storage().nbytes() == 0 and offloader.optimizer is optimizer
    assert not hasattr(weight, "_relax_cpu_offload_data")
    assert all(torch.equal(value.cpu(), snapshot[name]) for name, value in weights.items())
    with pytest.raises(RuntimeError, match="no longer match"):
        offloader.begin_weight_sync(weights, overlap=True, initial_snapshot=snapshot)
    consumer, done = torch.cuda.Stream(), torch.cuda.Event()
    target = torch.empty_like(weight)
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(consumer):
        target.copy_(weight)
        done.record()
    done.synchronize()
    result = offloader.finish_weight_sync()
    assert done.query() and torch.equal(target.cpu(), snapshot["weight"])
    assert result["weight"].data_ptr() == result["alias"].data_ptr() == cpu_backing
    assert result["persistent"] is snapshot["persistent"]
    assert flat.untyped_storage().nbytes() == 0 and offloader.initial_weight_sync_supported(weights, result)
    offloader.reload()
    assert torch.equal(weight.cpu(), snapshot["weight"]) and buffer.grad_data.untyped_storage().nbytes() > 0
    assert torch.equal(grad.cpu(), torch.zeros(16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires real singleton NCCL and CUDA copies")
def test_singleton_export_copy_matches_native_nccl_without_source_alias(monkeypatch, tmp_path):
    import torch.distributed as dist

    from relax.backends.megatron.weight_update import common

    dist.init_process_group("nccl", init_method=(tmp_path / "store").as_uri(), rank=0, world_size=1)
    try:
        monkeypatch.setattr(
            common,
            "mpu",
            SimpleNamespace(
                get_tensor_model_parallel_world_size=lambda: 1,
                get_tensor_model_parallel_group=lambda: dist.group.WORLD,
            ),
        )
        param = torch.arange(4096, device="cuda", dtype=torch.bfloat16).view(64, 64)
        param.tensor_model_parallel, param.partition_dim, param.partition_stride = True, 0, 1
        name = "decoder.layers.0.mlp.linear_fc1.weight"
        native = common.all_gather_param(SimpleNamespace(), name, param)
        copied = common.all_gather_param(SimpleNamespace(), name, param, copy_single_rank=True)
        assert torch.equal(native.view(torch.uint8), copied.view(torch.uint8))
        assert len({param.data_ptr(), native.data_ptr(), copied.data_ptr()}) == 3
        copied.zero_()
        assert torch.equal(native, param) and torch.count_nonzero(param) > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires native HDO pinned allocation and BF16 copies")
def test_allocator_warmup_preserves_real_hdo_updates_and_unused_parameters():
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
    from transformer_engine.pytorch.optimizers import FusedAdam

    weights = [
        [torch.nn.Parameter(torch.ones(64, 128, device="cuda", dtype=torch.bfloat16)) for _ in range(2)]
        for _ in range(2)
    ]
    optimizers = [
        HybridDeviceOptimizer(
            [dict(params=params, lr=3e-4, fused=True)],
            offload_fraction=1.0,
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=FusedAdam,
            pin_cpu_grads=True,
            pin_cpu_params=False,
            overlap_cpu_optimizer_d2h_h2d=False,
        )
        for params in weights
    ]
    candidate = optimizers[1]
    for step in range(3):
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        old_bindings = dict(candidate.cpu_copy_map_grad)
        warm_cpu_gradient_allocator(SimpleNamespace(optimizer=candidate))
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state())
        assert dict(candidate.cpu_copy_map_grad) == old_bindings
        assert candidate.param_to_inner_param[weights[1][1]].grad is None
        for params, optimizer in zip(weights, optimizers, strict=True):
            params[0].grad = torch.full_like(params[0], 0.125 + step)
            optimizer.step()
        torch.cuda.synchronize()
        for expected, actual in zip(weights[0], weights[1], strict=True):
            assert torch.equal(expected, actual)
            assert torch.equal(optimizers[0].param_to_inner_param[expected], candidate.param_to_inner_param[actual])
            left, right = optimizers[0].state[expected], candidate.state[actual]
            assert left.keys() == right.keys()
            for key in left:
                assert torch.equal(left[key], right[key]) if torch.is_tensor(left[key]) else left[key] == right[key]


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
