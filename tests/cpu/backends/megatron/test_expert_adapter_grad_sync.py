# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two CPU ranks reproduce fused main_grad accumulation without CUDA
kernels."""

import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _expert_adapter_worker(rank: int, rendezvous: str) -> None:
    from megatron.bridge.peft.utils import mark_expert_parallel_replicated
    from megatron.core.optimizer.optimizer import MegatronOptimizer, copy_optimizer_param_metadata

    from relax.backends.megatron import model as training
    from relax.utils.training.lora_grad_sync import exclude_replicated_expert_adapter_grads_from_norm

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        groups = [dist.new_group([member], backend="gloo") for member in range(2)]
        local = groups[rank]
        pg_collection = SimpleNamespace(tp=local, pp=local, embd=local, pos_embd=local, dp_cp=dist.group.WORLD)
        pg_collection.ep = dist.group.WORLD

        class SetupComplete(Exception):
            pass

        def stop_after_setup(_args):
            raise SetupComplete

        for memory_level in (0, 1):
            events = []

            class Chunk(torch.nn.Linear):
                def finish_grad_sync(self, **_kwargs):
                    assert events == (["release"] if memory_level else [])
                    events.append("dp_sync")

                def scale_gradients(self, scale):
                    self.weight.main_grad.mul_(scale)

            chunk = Chunk(2, 1, bias=False)
            chunk.config = SimpleNamespace(moe_router_enable_expert_bias=False, moe_router_load_balancing_type="none")
            param = chunk.weight
            with torch.no_grad():
                param.fill_(1)
            param.main_grad = torch.zeros_like(param)

            def fused_accumulate(grad):
                param.main_grad.add_(grad)
                param.grad_added_to_main_grad = True
                return torch.zeros_like(grad)

            param.register_hook(fused_accumulate)
            mark_expert_parallel_replicated(param, ep_group=pg_collection.ep)
            param.shared = False
            param.allreduce = False
            master = torch.nn.Parameter(param.detach().clone())
            copy_optimizer_param_metadata(master, param)
            ordinary = torch.nn.Parameter(torch.zeros(1))
            ordinary_master = torch.nn.Parameter(ordinary.detach().clone())
            optimizer = torch.optim.SGD([master, ordinary_master], lr=0.1)
            # DistOpt allocates CUDA shards; reproduce only its source-to-shard map on CPU.
            child = SimpleNamespace(
                optimizer=optimizer,
                model_param_group_index_map={param: (0, 0), ordinary: (0, 1)},
                config=SimpleNamespace(
                    use_precision_aware_optimizer_no_fp8_or_ds_fp8=bool(memory_level),
                    use_precision_aware_optimizer=bool(memory_level),
                ),
                tp_group=local,
            )
            wrapper = child if memory_level == 0 else SimpleNamespace(chained_optimizers=[child, optimizer])
            exclude_replicated_expert_adapter_grads_from_norm(wrapper, ep_rank=rank)
            assert not param.shared
            assert master.shared == (rank > 0)
            assert not hasattr(ordinary_master, "shared")
            reference = torch.nn.Parameter(param.detach().clone())
            reference_optimizer = torch.optim.SGD([reference], lr=0.1)
            args = SimpleNamespace(
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                empty_unused_memory_level=memory_level,
                reset_optimizer_states=False,
            )
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(training, "get_args", lambda: args)
                patch.setattr(training, "resident_sft_param_overlap_enabled", lambda _args: False)
                patch.setattr(training, "should_disable_forward_pre_hook", stop_after_setup)
                patch.setattr(training, "available_memory", lambda: {})
                patch.setattr(training, "device_module", SimpleNamespace(empty_cache=lambda: events.append("release")))
                for step in (1, 2):
                    # Run the actual train() setup on every rollout, before any backward.
                    with pytest.raises(SetupComplete):
                        training.train(0, [chunk], SimpleNamespace(scale_loss=lambda loss: loss), None, [[]], [])
                    assert not hasattr(param, "_expert_parallel_grad_hook")
                    optimizer.zero_grad()
                    chunk.zero_grad()
                    param.main_grad.zero_()
                    events.clear()
                    for _ in range(2):
                        chunk(torch.full((1, 2), float((rank + 1) * step))).sum().backward()
                    torch.testing.assert_close(param.main_grad, torch.full_like(param, 2 * (rank + 1) * step))
                    # Core divides by four global tokens; Bridge then sums the two EP shards.
                    chunk.config.finalize_model_grads_func([chunk], torch.tensor(2.0), pg_collection=pg_collection)
                    expected_grad = torch.full_like(param, 1.5 * step)
                    torch.testing.assert_close(param.main_grad, expected_grad)
                    grad = param.main_grad.clone()
                    if memory_level:
                        master.decoupled_grad = grad
                    else:
                        master.grad = grad
                    grads_for_norm = MegatronOptimizer._filter_grads_for_norm(
                        child, optimizer.param_groups[0]["params"]
                    )
                    norm_squared = sum((grad.square().sum() for grad in grads_for_norm), torch.tensor(0.0))
                    dist.all_reduce(norm_squared, group=pg_collection.ep)
                    norm = norm_squared.sqrt()
                    reference.grad = expected_grad.clone()
                    reference_norm = torch.nn.utils.clip_grad_norm_([reference], max_norm=1.0)
                    torch.testing.assert_close(norm, reference_norm)
                    grad.mul_((1.0 / (norm + 1e-6)).clamp(max=1.0))
                    master.grad = grad
                    optimizer.step()
                    reference_optimizer.step()
                    with torch.no_grad():
                        param.copy_(master)
                    torch.testing.assert_close(param, reference)
                    assert not param.shared
    finally:
        dist.destroy_process_group()


def test_train_finalizer_sums_fused_expert_gradients_once_per_step(tmp_path, monkeypatch):
    pytest.importorskip("megatron.bridge.peft.utils", exc_type=ImportError)
    if not dist.is_gloo_available():
        pytest.skip("CPU expert-gradient regression requires Gloo")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    processes = mp.start_processes(
        _expert_adapter_worker,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=False,
        start_method="spawn",
    )
    try:
        deadline = time.monotonic() + 120
        while not processes.join(timeout=1):
            assert time.monotonic() < deadline, "Expert-gradient worker timed out"
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
