# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Physical copy-stream and bank-reuse regression on a single CUDA device."""

from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.weight_update.delta_export import ShardDeltaExporter
from relax.backends.megatron.weight_update.shard_snapshot import CudaShardInputs


class BucketReader:
    def __init__(self):
        self.staged = []
        self.direct = []

    def get_hf_weight_chunks(self, shards):
        assert isinstance(shards, CudaShardInputs)
        self.direct.append({n: x.device.type for n, x in shards.shards.items()})
        buckets = [[SimpleNamespace(name=n, src_rank=0)] for n in shards]
        keys = {n: n for n in shards}
        for group in shards.prefetch(buckets, keys, rank=0):
            name = group[0].name
            self.staged.append(name in shards.active)
            yield [(name, shards[name])]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires physical CUDA streams and pinned host memory")
def test_direct_device_export_cuda_prefetch_slot_reuse_and_failed_ack():
    cur, old = BucketReader(), BucketReader()
    config = SimpleNamespace(bucket_bytes=256, gpu_budget_bytes=1024)
    s = ShardDeltaExporter(cur, old, snapshot_budget_bytes=4096, config=config)
    device = torch.device("cuda", torch.cuda.current_device())
    words = torch.arange(64, dtype=torch.int16, device=device)
    live = {str(i): (words + i).view(torch.bfloat16) for i in range(4)}
    live["counter"] = torch.tensor(2**48 + 5, dtype=torch.int64, device=device)
    s.begin(live, live_locked=True)
    assert s.sender_pipeline_active
    seed = [item for group in s.export() for item in group]
    assert all(x.full for x in seed)
    assert all(x.is_pinned() for x in s._pending.values())
    s.commit()
    assert not s.sender_pipeline_active
    first_bank = s._ack_bank.data_ptr()
    first_words = {name: t.reshape(-1).view(torch.uint8).clone() for name, t in s._acknowledged.items()}

    for i, name in enumerate(list(live)[:4]):
        live[name].view(torch.int16)[9] = -32768 if i % 2 else 0x7FC1
    live["counter"].add_(7)
    s.begin(live, live_locked=True)
    pending_bank = s._backup.storage.data_ptr()
    delta = [item for group in s.export() for item in group]
    assert all(old.staged) and all(d == "cpu" for d in old.direct[-1].values())
    assert all(d == "cuda" for d in cur.direct[-1].values())
    for i, item in enumerate(delta[:4]):
        assert item.positions.tolist() == [9]
        assert item.values.view(torch.int16).tolist() == ([-32768] if i % 2 else [0x7FC1])
    assert delta[-1].full and delta[-1].values.item() == 2**48 + 12
    s.abort()
    assert s._ack_bank.data_ptr() == first_bank
    for name, t in s._acknowledged.items():
        assert torch.equal(t.reshape(-1).view(torch.uint8), first_words[name])

    s.begin(live, live_locked=True)
    assert s._backup.storage.data_ptr() == pending_bank
    assert all(item.full for group in s.export() for item in group)
    s.commit()
    s.begin(live, live_locked=True)
    assert s._backup.storage.data_ptr() == first_bank
    stream = s.export()
    next(stream)
    s.abort()
    assert s._ack_bank.data_ptr() == pending_bank
