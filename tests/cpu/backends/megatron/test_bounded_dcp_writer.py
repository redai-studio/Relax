# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections import defaultdict

import pytest


torch = pytest.importorskip("torch")
dcp = pytest.importorskip("torch.distributed.checkpoint")


@pytest.mark.parametrize("budget", [0, 32])
def test_bounded_dcp_writer_preserves_checkpoint_and_limits_tensor_bytes(tmp_path, monkeypatch, budget):
    from relax.utils.checkpoint_write_patch import make_bounded_dcp_writer

    monkeypatch.setenv("MEGATRON_SYNC_SAVE_STAGE_BYTES", str(budget))
    expected = {str(i): torch.arange(n, dtype=torch.float32) for i, n in enumerate([4, 4, 10, 4, 4])}
    dcp.save(expected, storage_writer=make_bounded_dcp_writer(tmp_path), no_dist=True)
    metadata = dcp.FileSystemReader(tmp_path).read_metadata()
    stored = defaultdict(list)
    for index, info in metadata.storage_data.items():
        stored[info.relative_path].append(expected[index.fqn].nbytes)

    assert len(stored) == (len(expected) if budget == 0 else 3)
    assert all(len(sizes) == 1 or sum(sizes) <= budget for sizes in stored.values())
    loaded = {name: torch.empty_like(value) for name, value in expected.items()}
    dcp.load(loaded, checkpoint_id=tmp_path, no_dist=True)
    for name, value in expected.items():
        torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)
