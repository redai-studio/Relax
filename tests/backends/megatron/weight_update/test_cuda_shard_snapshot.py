# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise copy-stream completion and pinned bank reuse with real CUDA device
memory."""

import pytest
import torch

from relax.backends.megatron.weight_update.shard_snapshot import CudaShardBackup, shard_bank_layout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires physical CUDA streams and pinned host memory")
def test_copy_stream_preserves_acknowledged_raw_bits_when_reusing_spare_bank():
    words = torch.arange(64, dtype=torch.int16, device="cuda")
    live = {str(i): (words + i).view(torch.bfloat16) for i in range(4)}
    live["counter"] = torch.tensor(2**48 + 5, dtype=torch.int64, device="cuda")

    def backup(spare=None):
        layout, size = shard_bank_layout(live.items(), 256)
        result = CudaShardBackup(layout, size, spare, 4096, 256)
        result.finish()
        return result

    acknowledged = backup()
    assert all(value.is_pinned() for value in acknowledged.targets.values())
    frozen = {name: value.reshape(-1).view(torch.uint8).clone() for name, value in acknowledged.targets.items()}
    for i, value in enumerate(list(live.values())[:4]):
        value.view(torch.int16)[9] = -32768 if i % 2 else 0x7FC1
    live["counter"].add_(7)
    pending = backup()
    for name, value in live.items():
        assert torch.equal(
            pending.targets[name].reshape(-1).view(torch.uint8), value.reshape(-1).view(torch.uint8).cpu()
        )
        assert torch.equal(acknowledged.targets[name].reshape(-1).view(torch.uint8), frozen[name])
    live["counter"].add_(1)
    retried = backup(pending.storage)
    assert retried.storage is pending.storage
    assert retried.targets["counter"].item() == 2**48 + 13
    assert acknowledged.targets["counter"].item() == 2**48 + 5
