# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from relax.distributed.checkpoint_service.backends.shard_p2p.colocate_remote import send_remote_shards
from relax.distributed.checkpoint_service.backends.shard_p2p.remote.transport import BufferedWriter


@pytest.mark.parametrize("capacity", [10, 32])
def test_remote_chunks_offsets_tail_and_connection_reuse(capacity):
    source = torch.arange(48, dtype=torch.bfloat16)[8:32]
    tensor = SimpleNamespace(is_cuda=True, dtype=source.dtype, is_contiguous=lambda: True, view=source.view)
    target, buffer = torch.zeros(48, dtype=torch.uint8), torch.empty(capacity, dtype=torch.uint8)
    writer = object.__new__(BufferedWriter)
    writer.agent = SimpleNamespace(add_remote_agent=Mock(return_value="peer"), remove_remote_agent=Mock())
    writer.targets, writer.target_registration_keys = {}, {}
    writer.peer = writer.generation = writer._active_range_batch = None
    writer.failed, writer._retained_failure = False, False
    writer.acquire = lambda: (0, buffer)

    def submit(index, writes, manifests):
        for write, payload in writes:
            for row in range(write["rows"]):
                start = write["dst_offset"] + row * write["dst_stride"]
                target[start : start + write["row_bytes"]].copy_(
                    payload[row * write["row_bytes"] : (row + 1) * write["row_bytes"]]
                )

    writer.submit_ranges = submit
    writer.finish = Mock()
    write = dict(
        id="w",
        source_rank=3,
        source="s",
        param="w",
        src_offset=0,
        src_stride=24,
        dst_offset=4,
        dst_stride=20,
        rows=2,
        row_bytes=16,
    )
    plan = {"0": dict(writes=[write], plan_hash="plan")}
    manifest = dict(generation="epoch-1", agent_metadata="YWdlbnQ=", persistent_registration_key="a")
    for version in (1, 2, 3):
        manifest.update(generation=f"epoch-{version}", persistent_registration_key="b" if version == 3 else "a")
        receipts, _ = send_remote_shards({"s": tensor}, plan, {"0": manifest}, rank=3, version=version, writer=writer)
        assert receipts["0"] == dict(
            source_rank=3, version=version, generation=f"epoch-{version}", plan_hash="plan", writes=["w"]
        )
    assert writer.agent.add_remote_agent.call_count == 2 and writer.agent.remove_remote_agent.call_count == 1
    expected = torch.zeros(48, dtype=torch.uint8)
    expected[4:20], expected[24:40] = source.view(torch.uint8)[:16], source.view(torch.uint8)[24:40]
    assert torch.equal(target, expected)
