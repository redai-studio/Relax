# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.models.keye_vl2.test_packed_padding import _check_packed_padding


def _cp_worker(rank: int, init_method: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=init_method)
    try:
        _check_packed_padding(2, rank, dist.group.WORLD, False)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Packed Keye CP2 regression needs two CUDA devices")
def test_keye_cp2_padded_documents_match_unpadded_forward_and_gradients(tmp_path) -> None:
    mp.spawn(_cp_worker, args=(f"file://{tmp_path / 'packed-cp-init'}",), nprocs=2, join=True)
