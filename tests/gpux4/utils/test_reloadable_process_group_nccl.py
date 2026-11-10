# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from tests.utils.test_reloadable_process_group import _collective_worker, requires_collective_apis, torch


@requires_collective_apis
def test_collectives_follow_reloaded_inner_group(tmp_path) -> None:
    if not torch.distributed.is_nccl_available() or torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA devices and NCCL are required")
    torch.multiprocessing.spawn(
        _collective_worker, args=((tmp_path / "rendezvous").as_uri(), "nccl"), nprocs=2, join=True
    )
