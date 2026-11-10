# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.utils.data.test_routing_replay_transfer import _GPU_TESTS, _transfer_worker


@pytest.mark.skipif(not _GPU_TESTS, reason="set RELAX_ROUTING_REPLAY_GPU_TEST=1 on two available GPUs")
def test_routing_replay_nccl_staging_peak_is_independent_of_batch_size(tmp_path: Path) -> None:
    if not dist.is_nccl_available() or torch.cuda.device_count() < 2:
        pytest.skip("NCCL and two available GPUs are required")
    mp.spawn(_transfer_worker, args=(str(tmp_path / "nccl_init"), (2, 1, 1), "nccl"), nprocs=2, join=True)
