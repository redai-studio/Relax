# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise imported Torch collectives across process-group reloads."""

from datetime import timedelta
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
dist = torch.distributed
pytestmark = pytest.mark.skipif(
    not dist.is_available() or not hasattr(dist, "all_gather_single"), reason="Torch single-tensor collective APIs"
)


def _collective_worker(rank: int, rendezvous: str, backend: str) -> None:
    from relax.utils import reloadable_process_group as reloadable

    # Megatron imports these before Relax applies its module-level wrappers.
    gather_alias = dist.all_gather_into_tensor
    scatter_alias = dist.reduce_scatter_tensor
    exchange_alias = dist.all_to_all_single
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        original_new_group = dist.new_group
        reloadable.monkey_patch_torch_dist()
        group = reloadable.ReloadableProcessGroup(original_new_group([0, 1], backend=backend), [0, 1])
        reloadable.device_utils.get_dist_backend = lambda: backend
        for cycle in range(3):
            source = torch.full((2,), rank + cycle * 10, dtype=torch.float32, device=device)
            gathered = torch.empty(4, device=device)
            gather_alias(gathered, source, group=group)
            torch.testing.assert_close(
                gathered, torch.tensor([cycle * 10] * 2 + [cycle * 10 + 1] * 2, dtype=torch.float32, device=device)
            )
            reduced = torch.empty(2, device=device)
            scatter_alias(reduced, torch.full((4,), rank + 1.0, device=device), group=group)
            torch.testing.assert_close(reduced, torch.full((2,), 3.0, device=device))
            exchanged = torch.empty(2, device=device)
            exchange_alias(exchanged, torch.tensor([rank * 10.0, rank * 10.0 + 1], device=device), group=group)
            torch.testing.assert_close(exchanged, torch.tensor([float(rank), 10.0 + rank], device=device))
            if backend == "nccl":
                with dist._coalescing_manager(group=group, device=device):
                    dist.all_gather_single(gathered, source, group=group)
                torch.testing.assert_close(
                    gathered, torch.tensor([cycle * 10] * 2 + [cycle * 10 + 1] * 2, dtype=torch.float32, device=device)
                )
                with dist._coalescing_manager(group=group, device=device):
                    dist.reduce_scatter_single(reduced, torch.full((4,), rank + 1.0, device=device), group=group)
                torch.testing.assert_close(reduced, torch.full((2,), 3.0, device=device))
            reloadable.ReloadableProcessGroup.destroy_process_groups(post_destroy_delay=0)
            assert group.group is None
            reloadable.ReloadableProcessGroup.reload_process_groups(max_retries=1)
        extra = reloadable.ReloadableProcessGroup(original_new_group([0, 1], backend=backend), [0, 1])
        # A peer with an unregistered group must reject retention on all ranks.
        requested = [group] if rank == 0 else [extra.group]
        assert reloadable.retain_process_groups(requested, metadata_group=dist.group.WORLD) is None
        selected = reloadable.retain_process_groups([group], metadata_group=dist.group.WORLD)
        assert len(selected) == 1 and selected[0] is group
        for cycle in range(3):
            reloadable.destroy_process_groups(post_destroy_delay=0)
            assert group.group is not None and extra.group is None
            reloadable.reload_process_groups(process_groups=selected, max_retries=1)
            assert extra.group is None
            value = torch.full((2,), rank + cycle, dtype=torch.float32, device=device)
            gathered = torch.empty(4, device=device)
            gather_alias(gathered, value, group=group)
            torch.testing.assert_close(
                gathered, torch.tensor([cycle, cycle, cycle + 1, cycle + 1], dtype=torch.float32, device=device)
            )
        reloadable.destroy_process_groups(post_destroy_delay=0, force=True)
        assert group.group is None and not group._keep_alive_for_weight_sync
    finally:
        reloadable.ReloadableProcessGroup.destroy_process_groups(post_destroy_delay=0, force=True)
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_collectives_follow_reloaded_inner_group(tmp_path, backend: str) -> None:
    if backend == "gloo" and not dist.is_gloo_available():
        pytest.skip("Gloo is not available")
    if backend == "nccl" and (not dist.is_nccl_available() or torch.cuda.device_count() < 2):
        pytest.skip("Two CUDA devices and NCCL are required")
    torch.multiprocessing.spawn(
        _collective_worker, args=((tmp_path / "rendezvous").as_uri(), backend), nprocs=2, join=True
    )


def _check_collectives_after_reload(rank: int, rendezvous: str) -> None:
    # Megatron caches these aliases before Relax patches torch.distributed.
    from torch.distributed import all_gather_into_tensor, all_to_all_single, reduce_scatter_tensor

    from relax.utils.reloadable_process_group import (
        destroy_process_groups,
        monkey_patch_torch_dist,
        reload_process_groups,
    )

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    monkey_patch_torch_dist()
    group = dist.new_group(ranks=[0, 1], backend="nccl")
    try:
        for cycle in range(2):
            value = torch.tensor([rank + 1.0], device="cuda")
            gathered = torch.empty(2, device="cuda")
            for gather in (all_gather_into_tensor, dist.all_gather_single):
                gather(gathered, value, group=group)
                torch.testing.assert_close(gathered, torch.tensor([1.0, 2.0], device="cuda"))

            values = torch.ones(2, device="cuda") * (rank + 1)
            reduced = torch.empty(1, device="cuda")
            for scatter in (reduce_scatter_tensor, dist.reduce_scatter_single):
                scatter(reduced, values, group=group)
                torch.testing.assert_close(reduced, torch.tensor([3.0], device="cuda"))

            all_to_all_single(gathered, torch.arange(2, device="cuda", dtype=torch.float32) + rank * 2, group=group)
            torch.testing.assert_close(gathered, torch.tensor([rank, rank + 2], device="cuda", dtype=torch.float32))
            destroy_process_groups(post_destroy_delay=0)
            if cycle == 0:
                reload_process_groups()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not hasattr(dist, "all_gather_single"),
    reason="Requires two CUDA devices and the new PyTorch tensor collective APIs",
)
def test_reloadable_process_group_cached_collectives_after_reload(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(
        _check_collectives_after_reload,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
