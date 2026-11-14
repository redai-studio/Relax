# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from datetime import timedelta
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import pytest
import ray
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.distributed.ray.inference_handle import ManagedRolloutHandle
from relax.engine.inference.lifecycle import managed_weight_update


def _rank(rank: int, rendezvous: str, directory: str, fail: bool) -> None:
    from relax.utils.distributed_utils import init_gloo_group

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    group = init_gloo_group(distributed_timeout_minutes=1)
    original_get = ray.get
    ray.get = lambda value: value
    entered = False

    def begin(source):
        if fail:
            raise RuntimeError("manager rejected admission")
        return "weight-token"

    class Backend:
        args = SimpleNamespace(rank=rank)
        rollout_manager = ManagedRolloutHandle(
            SimpleNamespace(
                begin_weight_update=SimpleNamespace(remote=begin),
                end_weight_update=SimpleNamespace(remote=lambda token: None),
            ),
            None,
        )

        @managed_weight_update
        def update(self):
            nonlocal entered
            entered = True
            dist.all_reduce(value, group=group)

    import torch

    value = torch.ones(1)
    try:
        Backend().update()
        result = f"entered={entered},sum={int(value[0])}"
    except RuntimeError as exc:
        result = f"entered={entered},error={exc}"
    finally:
        ray.get = original_get
        dist.destroy_process_group(group)
        dist.destroy_process_group(dist.group.WORLD)
    Path(directory, f"rank-{rank}.txt").write_text(result)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires CPU Gloo collectives")
@pytest.mark.parametrize("fail", [False, True])
def test_weight_fence_outcome_reaches_all_ranks_before_collectives(tmp_path, fail):
    rendezvous = f"file://{tmp_path / 'rendezvous'}"
    context = mp.spawn(_rank, args=(rendezvous, str(tmp_path), fail), nprocs=2, join=False)
    deadline = monotonic() + 45
    try:
        while not context.join(timeout=1):
            if monotonic() >= deadline:
                raise TimeoutError("CPU weight-fence ranks did not finish within 45 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    results = [(tmp_path / f"rank-{rank}.txt").read_text() for rank in range(2)]
    if fail:
        assert results == ["entered=False,error=Inference weight-update fence failed: RuntimeError"] * 2
    else:
        assert results == ["entered=True,sum=2"] * 2
