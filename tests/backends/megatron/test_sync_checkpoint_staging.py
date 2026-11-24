# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CUDA coverage for synchronous distributed checkpoint staging."""

import pytest
import torch
from torch.distributed import checkpoint


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for D2H checkpoint staging")
bounded_staging = pytest.importorskip(
    "megatron.core.dist_checkpointing.strategies.bounded_staging", exc_type=ImportError
)


@pytest.mark.parametrize("budget", [0, 8192])
def test_sync_cuda_roundtrip(tmp_path, budget):
    state = {
        "noncontiguous": torch.randn(32, 64, device="cuda").t(),
        "oversized": torch.randn(1024, 1024, device="cuda"),
    }
    writer = bounded_staging.make_bounded_writer(checkpoint.FileSystemWriter, tmp_path, budget)
    checkpoint.save(state, storage_writer=writer)
    restored = {key: torch.empty_like(tensor, device="cpu") for key, tensor in state.items()}
    checkpoint.load(restored, checkpoint_id=tmp_path)
    torch.testing.assert_close(restored, {key: tensor.cpu() for key, tensor in state.items()})


def test_sync_window_waits_for_caller_cuda_stream():
    source = torch.zeros(128, device="cuda")
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stager = bounded_staging._BoundedSyncStager(1024)
    try:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100_000_000)
            source.fill_(7)
            stager.start([source])
        restored = stager.take()
        torch.testing.assert_close(restored, torch.full((128,), 7.0))
        del restored
        stager.release()
    finally:
        stager.close()
