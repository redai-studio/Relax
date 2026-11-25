# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from relax.distributed.checkpoint_service.backends.shard_p2p.replicas import partition_writes, wire_writes


def test_tp_replicas_require_equal_logical_sources_and_entirely_remote_outputs():
    from copy import deepcopy

    from relax.distributed.checkpoint_service.backends.shard_p2p.replicas import tp_replica_configurations

    commands = [[]]
    for rank in range(2):
        writes = [
            dict(
                id=f"{rank}-{name}",
                target=f"tp-{rank}",
                source="actor.w",
                source_rank=0,
                param=name,
                src_offset=16,
                src_stride=16,
                dst_offset=0,
                dst_stride=16,
                rows=4,
                row_bytes=16,
            )
            for name in ("w", "local")
        ]
        commands[0].append(
            dict(
                geometry=dict(ordinary_targets=dict(w=[4, 8], local=[4, 8])),
                writes=writes,
                delivery=dict(
                    delivery=dict(
                        receiver=dict(hostname="host", boot_id="boot", pid=rank + 1, gpu_uuid=str(rank)),
                        routes=dict(nixl=[writes[0]["id"]], cuda_ipc=[writes[1]["id"]]),
                    )
                ),
            )
        )
    capabilities = [["w", "local"], ["w", "local"]]
    configs = tp_replica_configurations(commands, capabilities)
    assert configs.keys() == {"0-0", "0-1"}
    assert configs["0-0"] == configs["0-1"]
    assert configs["0-0"]["parameters"] == ["w"]
    changed = deepcopy(commands)
    changed[0][1]["writes"][0]["src_offset"] += 2
    assert not tp_replica_configurations(changed, capabilities)
    assert not tp_replica_configurations(commands, [["w"], []])
    changed = deepcopy(commands)
    changed[0][1]["delivery"]["delivery"]["receiver"]["hostname"] = "other"
    assert not tp_replica_configurations(changed, capabilities)


@pytest.mark.parametrize("size", [2, 3, 4])
def test_replica_chunks_reconstruct_strided_rectangles_and_preserve_fallback(size):
    sources = {"left": bytes(range(100)), "right": bytes(range(100, 200))}
    writes = [
        dict(
            id="left",
            source="left",
            source_rank=0,
            param="w",
            src_offset=2,
            src_stride=18,
            dst_offset=0,
            dst_stride=24,
            rows=4,
            row_bytes=10,
        ),
        dict(
            id="right",
            source="right",
            source_rank=1,
            param="w",
            src_offset=4,
            src_stride=22,
            dst_offset=10,
            dst_stride=24,
            rows=4,
            row_bytes=14,
        ),
        dict(
            id="native",
            source="left",
            source_rank=0,
            param="native",
            src_offset=6,
            src_stride=10,
            dst_offset=0,
            dst_stride=10,
            rows=1,
            row_bytes=10,
        ),
    ]
    layouts = dict(w=dict(nbytes=96), native=dict(nbytes=10))

    def apply(plan):
        outputs = {name: bytearray(value["nbytes"]) for name, value in layouts.items()}
        for write in plan:
            for row in range(write["rows"]):
                begin = write["src_offset"] + row * write["src_stride"]
                target = write["dst_offset"] + row * write["dst_stride"]
                outputs[write["param"]][target : target + write["row_bytes"]] = sources[write["source"]][
                    begin : begin + write["row_bytes"]
                ]
        return outputs

    expected, chunks, transferred = apply(writes), [], 0
    for rank in range(size):
        plan = partition_writes(writes, layouts, ["w"], rank, size)
        assert len({write["id"] for write in plan}) == len(plan)
        assert next(write for write in plan if write["param"] == "native") == writes[-1]
        output = apply(plan)
        width = layouts["w"]["nbytes"] // size
        chunks.append(output["w"][rank * width : (rank + 1) * width])
        transferred += sum(write["rows"] * write["row_bytes"] for write in plan if write["param"] == "w")
        assert output["native"] == expected["native"]
    assert b"".join(chunks) == expected["w"] and transferred == len(expected["w"])
    with pytest.raises(ValueError, match="equal BF16"):
        partition_writes(writes, layouts, ["native"], 0, 3)


def test_transposed_column_shards_reconstruct_across_partial_replica_rows(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import torch

    from relax.distributed.checkpoint_service.backends.shard_p2p.colocate_remote import send_remote_shards
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.transport import BufferedWriter

    expected = torch.arange(72, dtype=torch.bfloat16).view(12, 6)
    # Live sources have nonzero storage offsets, as Megatron flat buffers do.
    sources = {
        str(rank): torch.cat(
            (torch.zeros(8, dtype=torch.bfloat16), expected[:, rank * 3 : (rank + 1) * 3].reshape(-1))
        )[8:]
        for rank in range(2)
    }
    writes = [
        dict(
            id=str(rank),
            param="w",
            source=str(rank),
            source_rank=rank,
            src_offset=0,
            src_stride=6,
            dst_offset=rank * 6,
            dst_stride=12,
            rows=12,
            row_bytes=6,
        )
        for rank in range(2)
    ]
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    layouts, chunks = {"w": dict(nbytes=144)}, []
    for replica in range(4):
        target = torch.zeros(144, dtype=torch.uint8)
        writer = object.__new__(BufferedWriter)
        writer.agent = SimpleNamespace(
            add_remote_agent=Mock(return_value="peer"),
            remove_remote_agent=Mock(),
            get_xfer_descs=lambda ranges, _memory_type: ranges,
        )
        writer.targets, writer.target_registration_keys = {}, {}
        writer.peer = writer.generation = writer._active_range_batch = None
        writer.failed, writer._retained_failure = False, False
        writer.profile, writer.timeout = {}, 5
        buffer = torch.empty(48, dtype=torch.uint8)
        writer.acquire = lambda: (0, buffer)
        writer.finish = Mock()

        def write_from_descriptors(requests, _sources, _deadline):
            ((source_ranges, target_ranges, peer),) = requests
            assert peer == "peer"
            for (source_ptr, count, _), (target_ptr, length, device_id) in zip(
                source_ranges, target_ranges, strict=True
            ):
                src, dst = source_ptr - buffer.data_ptr(), target_ptr - target.data_ptr()
                assert count == length and device_id == 0
                assert 0 <= src <= buffer.numel() - count and 0 <= dst <= target.numel() - count
                target[dst : dst + count].copy_(buffer[src : src + count])

        writer._transfer_many = write_from_descriptors
        writer.submit_ranges = lambda _, batch, manifests: writer._write_ranges(batch, manifests, Mock())
        partial = partition_writes(wire_writes(writes, {"w": [12, 6]}), layouts, ["w"], replica, 4)
        for owner, source in sources.items():
            plan = dict(writes=[write for write in partial if write["source"] == owner], plan_hash="plan")
            tensor = SimpleNamespace(is_cuda=True, dtype=source.dtype, is_contiguous=lambda: True, view=source.view)
            send_remote_shards(
                {owner: tensor},
                {"0": plan},
                {
                    "0": dict(
                        generation="epoch",
                        agent_metadata="YWdlbnQ=",
                        persistent_registration_key="a",
                        gpu_id=0,
                        params={"w": dict(address=target.data_ptr(), nbytes=target.numel())},
                    )
                },
                rank=int(owner),
                version=2,
                writer=writer,
            )
        chunks.append(target[replica * 36 : (replica + 1) * 36])
    actual = torch.cat(chunks).view(torch.bfloat16).view(6, 12).t().contiguous()
    assert torch.equal(actual, expected)
