# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise imported Torch collectives across process-group reloads."""

from datetime import timedelta

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
    finally:
        reloadable.ReloadableProcessGroup.destroy_process_groups(post_destroy_delay=0)
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
