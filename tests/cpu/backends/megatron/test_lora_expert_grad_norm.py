# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared expert adapters count once across EP while retaining every DP
shard."""

from datetime import timedelta
from time import monotonic
from types import MethodType, SimpleNamespace

import pytest


pytest.importorskip("megatron.core", exc_type=ImportError)


def _expert_norm_worker(rank: int, rendezvous: str) -> None:
    import torch
    import torch.distributed as dist
    from megatron.core.optimizer.optimizer import MegatronOptimizer

    from relax.utils.training.lora_grad_sync import mark_expert_adapter_optimizer_replicas

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=90))
    try:
        # Two EP groups, each holding one expert-DP shard of the shared [3, 4] gradient.
        ep_groups = [dist.new_group(members, backend="gloo") for members in ([0, 1], [2, 3])]
        ep_group = ep_groups[rank // 2]
        for precision_aware in (False, True):
            shared_model = torch.nn.Parameter(torch.ones(1))
            shared_model.expert_parallel_replicated = True
            ordinary_model = torch.nn.Parameter(torch.ones(1))
            views = [parameter.detach().view(-1) for parameter in (shared_model, ordinary_model)]
            gradients = [torch.tensor([float(3 + rank // 2)]), torch.tensor([float(rank + 1)])]
            for view, grad in zip(views, gradients, strict=True):
                view.allreduce = False
                view.tensor_model_parallel = True
                if precision_aware:
                    view.decoupled_grad = grad
                else:
                    view.grad = grad
            optimizer = SimpleNamespace(
                config=SimpleNamespace(
                    use_precision_aware_optimizer_no_fp8_or_ds_fp8=precision_aware,
                    use_precision_aware_optimizer=precision_aware,
                ),
                optimizer=SimpleNamespace(param_groups=[{"params": views}]),
                model_param_group_index_map={shared_model: (0, 0), ordinary_model: (0, 1)},
                get_parameters=lambda: views,
            )
            optimizer._filter_grads_for_norm = MethodType(MegatronOptimizer._filter_grads_for_norm, optimizer)
            optimizer.get_grads_for_grad_norm = MethodType(MegatronOptimizer.get_grads_for_grad_norm, optimizer)
            chained = SimpleNamespace(chained_optimizers=[optimizer])
            marked = mark_expert_adapter_optimizer_replicas(chained, expert_group=ep_group)
            repeated = mark_expert_adapter_optimizer_replicas(chained, expert_group=ep_group)
            assert not getattr(shared_model, "shared", False)
            assert not getattr(ordinary_model, "shared", False)
            assert len(optimizer.get_parameters()) == 2

            # Exercise MCore's actual selection with a four-rank Gloo sum.
            # CUDA norm and clipping kernels require separate GPU validation.
            selected = optimizer.get_grads_for_grad_norm()
            squared = sum((gradient.square().sum() for gradient in selected), torch.tensor(0.0))
            dist.all_reduce(squared, group=dist.group.WORLD)
            torch.testing.assert_close(squared, torch.tensor(float(3**2 + 4**2 + 1**2 + 2**2 + 3**2 + 4**2)))
            assert marked == rank % 2
            assert repeated == 0
            # Excluding norm contributions must not alter the gradients or
            # parameter list that the optimizer later clips and updates.
            torch.testing.assert_close(gradients[0], torch.tensor([float(3 + rank // 2)]), rtol=0, atol=0)
            torch.testing.assert_close(gradients[1], torch.tensor([float(rank + 1)]), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_shared_expert_norm_counts_each_dp_shard_once(tmp_path) -> None:
    import torch.multiprocessing as mp

    context = mp.spawn(_expert_norm_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4, join=False)
    deadline = monotonic() + 150
    try:
        while not context.join(timeout=1):
            if monotonic() >= deadline:
                pytest.fail("Shared expert norm regression exceeded 150 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
