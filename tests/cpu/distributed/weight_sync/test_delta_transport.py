# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from relax.distributed.weight_sync.codec import encode, tensors, validate_directory
from relax.distributed.weight_sync.tcp_transport import TcpFrameSource, receive


@pytest.mark.parametrize("empty", [False, True])
def test_tcp_delivers_same_immutable_wire_to_all_fixed_recipients(empty):
    positions = torch.empty(0, dtype=torch.int32) if empty else torch.tensor([0, 3], dtype=torch.int32)
    bits = torch.empty(0, dtype=torch.int16) if empty else torch.tensor([0x7FC1, -32768], dtype=torch.int16)
    metadata, raw = encode([("weight", (2, 2), torch.bfloat16, positions, bits.view(torch.bfloat16))], max_bytes=1024)
    immutable = raw.numpy().tobytes()
    source = TcpFrameSource("127.0.0.1", timeout_s=3, host_budget_bytes=1024)
    try:
        descriptor = source.publish(immutable, {"rank0", "rank1"})
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(receive, descriptor, member, timeout_s=3, max_bytes=1024) for member in ("rank0", "rank1")
            ]
            payloads = [future.result(timeout=5) for future in futures]
        source.finish()
        raw.zero_()
        for payload in payloads:
            assert payload == immutable
            restored = torch.frombuffer(payload, dtype=torch.uint8) if payload else torch.empty(0, dtype=torch.uint8)
            entry, restored_positions, restored_values = next(iter(tensors(metadata, restored)))
            assert entry["dtype"] == "bfloat16"
            assert torch.equal(restored_positions, positions)
            assert torch.equal(restored_values.view(torch.int16), bits)
    finally:
        source.close()


def test_wire_retains_mixed_dtype_full_values_and_rejects_overlapping_ranges():
    bf16 = torch.tensor([0x7FC1, -32768], dtype=torch.int16).view(torch.bfloat16)
    fp32 = torch.tensor([0x7FC00002, 0], dtype=torch.int32).view(torch.float32)
    integer = torch.tensor([2**40, -(2**40)], dtype=torch.int64)
    metadata, raw = encode(
        [
            ("bf16", (2,), bf16.dtype, None, bf16),
            ("fp32", (2,), fp32.dtype, None, fp32),
            ("integer", (2,), integer.dtype, None, integer),
        ],
        max_bytes=1024,
    )
    values = [value for _, _, value in tensors(metadata, raw)]
    assert torch.equal(values[0].view(torch.int16), bf16.view(torch.int16))
    assert torch.equal(values[1].view(torch.int32), fp32.view(torch.int32))
    assert torch.equal(values[2], integer)
    metadata["tensors"][1]["offset"] = 0
    with pytest.raises(ValueError, match="overlap"):
        validate_directory(metadata, max_bytes=1024)


@pytest.mark.parametrize("fail_consume", [False, True])
def test_tcp_pipeline_keeps_one_frame_and_drains_actual_receiver_ack(monkeypatch, fail_consume):
    from threading import Event
    from types import SimpleNamespace

    import ray

    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    receiving, release = Event(), Event()
    delivered, actions = [], []

    def remote(action, **kw):
        actions.append(action)
        if action == "receive":
            receiving.set()
            assert release.wait(5)
            delivered.append(receive(kw["descriptor"], "1", timeout_s=3, max_bytes=16))
        elif action == "consume":
            if fail_consume:
                raise RuntimeError("receiver failed to install frame")
            return {"full_required": [f"name{kw['round_id']}"]}
        elif action == "finalize":
            return {"success": True}

    monkeypatch.setattr(ray, "get", lambda refs, **kw: refs)
    transfer = DeltaTransfer(DeltaSyncConfig(stream="test", staging_budget_bytes=16), "cpu")
    transfer.engines = (SimpleNamespace(delta_weight_sync=SimpleNamespace(remote=remote)),)
    transfer.ranks, transfer.tcp_members = [[1]], {"1"}
    transfer.routes, transfer.connection = [], {"world_size": 1}
    transfer.tcp = TcpFrameSource("127.0.0.1", timeout_s=3, host_budget_bytes=16)
    try:
        transfer.begin(version="2", base="1", seed=False)
        metadata, raw = encode([("w", (2,), torch.float32, None, torch.ones(2))], max_bytes=16)
        first = raw.numpy().tobytes()
        assert not transfer.submit_frame(metadata, raw)
        assert receiving.wait(3)
        # Produce the next frame on the main thread while the socket is pending.
        raw.zero_()
        transfer.prepare_frame(raw)
        raw.fill_(99)
        with pytest.raises(RuntimeError, match="already waiting"):
            transfer.prepare_frame(raw)
        with pytest.raises(RuntimeError, match="Drain"):
            transfer.submit_frame(metadata, raw)
        with pytest.raises(RuntimeError, match="Drain"):
            transfer.finalize()
        release.set()
        if fail_consume:
            with pytest.raises(RuntimeError, match="failed to install"):
                transfer.drain_frame()
            with pytest.raises(RuntimeError, match="Drain"):
                transfer.finalize()
            assert actions == ["begin", "receive", "ready", "consume"]
        else:
            assert transfer.drain_frame() == {"name0"}
            assert not transfer.submit_frame(metadata, raw)
            assert transfer.drain_frame() == {"name1"}
            assert transfer.finalize()["frames"] == 2
            assert actions == ["begin", "receive", "ready", "consume", "receive", "ready", "consume", "finalize"]
            assert delivered == [first, bytes(8)]
        transfer.close()
        assert actions[-1] == "close"
    finally:
        release.set()
        if transfer.tcp is not None:
            transfer.tcp.close()


def test_empty_staged_groups_still_consume_directory_admission():
    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    transfer = DeltaTransfer(DeltaSyncConfig(stream="test"), "cpu")
    transfer._directory_bytes = (16 << 20) - 2
    assert not transfer.can_stage({"tensors": []}, torch.empty(0, dtype=torch.uint8))


def test_empty_directory_heap_is_charged_even_when_json_is_small():
    import json

    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    metadata, raw = encode(
        [("weight", (1,), torch.float32, torch.empty(0, dtype=torch.int32), torch.empty(0))], max_bytes=8
    )
    transfer = DeltaTransfer(DeltaSyncConfig(stream="test"), "cpu")
    wire = len(json.dumps(metadata, separators=(",", ":")).encode())
    assert transfer._retained_directory_size(metadata) > wire
    transfer._directory_bytes = (16 << 20) - wire
    assert not transfer.can_stage(metadata, raw)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float64])
def test_adjacent_staged_frames_preserve_eight_byte_decode_alignment(dtype):
    from relax.backends.sglang.weight_sync_proxy import _StagingArena

    arena = _StagingArena(64, "cpu")
    first = torch.arange(5, dtype=torch.bfloat16)
    second = torch.tensor([17], dtype=dtype)
    _, raw = encode([("bf16", (5,), first.dtype, None, first)], max_bytes=64)
    arena.reserve(0, raw.numel()).copy_(raw)
    metadata, raw = encode([("buffer", (1,), dtype, None, second)], max_bytes=64)
    slab = arena.reserve(1, raw.numel())
    slab.copy_(raw)
    _, _, decoded = next(iter(tensors(metadata, slab)))
    assert torch.equal(decoded, second)
    arena.release(0)
    arena.release(1)
    assert arena.free == [(0, 64)]


def test_lane_staging_capacity_is_per_gpu_across_mixed_rounds():
    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    transfer = DeltaTransfer(DeltaSyncConfig(stream="test", staging_budget_bytes=32), "cpu")
    first = [torch.empty(size, dtype=torch.uint8) for size in (24, 8, 0, 5)]
    second = [torch.empty(size, dtype=torch.uint8) for size in (1, 24, 0, 16)]
    transfer._staged_bytes = transfer._staged_after(first)
    assert transfer.can_stage([{}] * 4, second)
    transfer._staged_bytes = transfer._staged_after(second)
    assert transfer._staged_bytes == [32, 32, 0, 24]
    assert not transfer.can_stage({}, torch.empty(1, dtype=torch.uint8))
    assert transfer.can_stage({}, torch.empty(0, dtype=torch.uint8))


def _send_four_lanes(rank, rendezvous):
    from datetime import timedelta
    from time import monotonic

    import torch.distributed as dist

    from relax.distributed.weight_sync.gpu_transport import GpuSyncGroup

    # Gloo exercises the real group-local peer APIs and zero-byte matching on
    # CPU. The CUDA allocator and final CUDA stream fence belong to GPU tests.
    dist.init_process_group("gloo", rank=rank, world_size=9, init_method=rendezvous, timeout=timedelta(seconds=20))
    group = object.__new__(GpuSyncGroup)
    group.device, group.rank, group.group = torch.device("cpu"), rank, dist.group.WORLD
    group.fence = lambda works, *, deadline: [w.wait(timedelta(seconds=deadline - monotonic())) for w in works]
    sizes = [3, 0, 13, 7]
    try:
        if rank == 0:
            slabs = [torch.full((n,), 10 + tp, dtype=torch.uint8) for tp, n in enumerate(sizes)]
            group.send_lanes(slabs, [[1, 2, 3, 4], [5, 6, 7, 8]], deadline=monotonic() + 15)
        else:
            tp = (rank - 1) % 4
            actual, works = group.receive({"bytes": sizes[tp], "target_shard": True})
            group.fence(works, deadline=monotonic() + 15)
            assert actual.tolist() == [10 + tp] * sizes[tp]
    finally:
        dist.destroy_process_group()


def test_two_tp4_replicas_receive_only_their_lane_over_group(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(_send_four_lanes, args=((tmp_path / "lane-wire").as_uri(),), nprocs=9, start_method="fork")


def test_source_requires_current_all_receiver_ack_and_paused_apply_for_lanes(monkeypatch):
    from types import SimpleNamespace

    import ray

    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    monkeypatch.setattr(ray, "get", lambda futures, **kwargs: futures)
    accepted = [True, True]
    remote_events = []
    transfers = []
    delivered = {}
    source = DeltaTransfer(DeltaSyncConfig(stream="s", target_shard_delta=True, staging_budget_bytes=256), "cpu")
    source.ranks = [list(range(1, 5)), list(range(5, 9))]
    source.connection = dict(world_size=9, target_shard_p2p=True)
    source.routes = []
    source.nccl_members = {str(r): r for r in range(1, 9)}
    source.group = SimpleNamespace(
        device=torch.device("cpu"),
        send_lanes=lambda raws, peers, **kw: transfers.append(
            [(peer, slab.numpy().tobytes()) for row in peers for peer, slab in zip(row, raws, strict=True)]
        ),
    )

    def remote(replica):
        def dispatch(action, **kwargs):
            remote_events.append((replica, action))
            if action == "begin":
                assert kwargs["control"]["target_shard_request"] is True
            elif action == "receive":
                assert [b["bytes"] for b in kwargs["batches"]] == [0, 6, 12, 18]
                assert all(b["target_shard"] is True for b in kwargs["batches"])
            elif action == "consume":
                delivered[replica] = kwargs["metadata"]
                return {"success": True, "full_required": []}
            elif action == "finalize":
                return {"success": True, "target_shard_capable": accepted[replica], "force_full_seed": False}
            return []

        return SimpleNamespace(delta_weight_sync=SimpleNamespace(remote=dispatch))

    source.engines = tuple(remote(i) for i in range(2))
    source.begin(version="1", base=None, seed=True)
    source.finalize()
    assert source.target_shard_capable and source.target_shard_version == "1"
    dirs, raws = [], []
    for tp in range(4):
        md, raw = encode(
            [
                (
                    "expert",
                    (4, 8),
                    torch.bfloat16,
                    torch.arange(tp, dtype=torch.int32),
                    torch.arange(tp, dtype=torch.bfloat16),
                )
            ],
            max_bytes=256,
        )
        md["target_shard"] = dict(format=1, tp_rank=tp, tp_size=4, entries=[0])
        dirs.append(md)
        raws.append(raw)
    source.begin(version="2", base="1", seed=False)
    with pytest.raises(ValueError, match="Mismatched"):
        source.stage(list(reversed(dirs)), raws)
    assert not any(ev == "receive" for _, ev in remote_events)
    source.stage(dirs, raws)
    assert not delivered
    assert not source.apply()
    assert delivered == {0: dirs, 1: dirs}
    assert transfers == [[(i + 1, raws[i % 4].numpy().tobytes()) for i in range(8)]]
    accepted[1] = False
    report = source.finalize()
    assert report["target_shard_lane_bytes"] == [0, 6, 12, 18] and report["logical_delivery_bytes"] == 72
    assert report["target_shard_changed_words"] == 6 and not source.target_shard_capable
    source.begin(version="3", base="2", seed=False)
    with pytest.raises(ValueError, match="acknowledged"):
        source.stage(dirs, raws)


@pytest.mark.parametrize("peer_agrees", [True, False])
def test_ordered_lane_consumes_hold_independent_slices_and_drain_before_mixed_or_capacity(monkeypatch, peer_agrees):
    from types import SimpleNamespace

    import ray

    from relax.backends.sglang.weight_sync_proxy import _StagingArena
    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    monkeypatch.setattr(ray, "get", lambda futures, **kwargs: futures)
    source = DeltaTransfer(DeltaSyncConfig(stream="s", target_shard_delta=True, staging_budget_bytes=32), "cpu")
    source.ranks = [list(range(1, 5)), list(range(5, 9))]
    source.nccl_members = {str(r): r for r in range(1, 9)}
    source.connection = {"world_size": 9, "target_shard_p2p": True}
    source.target_shard_version, source.target_shard_capable = "1", True
    arenas = [_StagingArena(32, "cpu") for _ in range(8)]
    views = [dict() for _ in arenas]
    calls = [[], []]

    def dispatch(replica, action, **kwargs):
        members = range(4 * replica, 4 * replica + 4)
        if action == "begin":
            return {"success": True, "ordered_target_consume": not replica or peer_agrees}
        if action == "receive":
            for member, batch in zip(members, kwargs["batches"], strict=True):
                key = kwargs["round_id"]
                views[member][key] = arenas[member].reserve(key, batch["bytes"])
        if action == "consume":
            first = kwargs["round_id"]
            epochs = [kwargs["metadata"], *kwargs.get("following", [])]
            calls[replica].append((first, len(epochs)))
            for key, frame in enumerate(epochs, first):
                for tp, member in enumerate(members):
                    # The sender progressively reuses its own tensors. All
                    # accumulated recipient frames must retain distinct bytes.
                    assert views[member].pop(key).tolist() == [key + 4 * tp] * frame[tp]["bytes"]
                    arenas[member].release(key)
            return {
                "success": True,
                "full_required": ["needs_full"] if first == 3 else [],
                "ordered_frames_ack": len(epochs),
            }
        return []

    source.engines = tuple(
        SimpleNamespace(delta_weight_sync=SimpleNamespace(remote=lambda op, i=i, **kw: dispatch(i, op, **kw)))
        for i in range(2)
    )

    def send_lanes(raw, peers, **kwargs):
        for peer in peers:
            for rank, slab in zip(peer, raw, strict=True):
                views[rank - 1][source.frame].copy_(slab)

    source.group = SimpleNamespace(device=torch.device("cpu"), send_lanes=send_lanes)
    source.begin(version="2", base="1", seed=False)
    assert source.ordered_target_consume is peer_agrees
    required = set()
    for key in range(6):
        frames, raws = [], []
        for tp, size in enumerate((4, 16, 8, 0)):
            # The second TP limits the group to two frames even though TP0
            # could retain all six. Frame 3 includes an ordinary loader input.
            frames.append(
                dict(
                    bytes=size,
                    tensors=[dict(count=1)] * (2 if key == 3 else 1),
                    target_shard=dict(format=1, tp_rank=tp, tp_size=4, entries=[0]),
                )
            )
            raws.append(torch.full((size,), key + 4 * tp, dtype=torch.uint8))
        required.update(source.send(frames, raws, defer_ordered=True))
        for raw in raws:
            raw.zero_()
    required.update(source.apply())
    expected = [(0, 2), (2, 1), (3, 1), (4, 2)] if peer_agrees else [(i, 1) for i in range(6)]
    assert calls == [expected, expected] and required == {"needs_full"}
    assert source.frame == 6 and all(arena.free == [(0, 32)] for arena in arenas)
    source.begin(version="3", base="2", seed=True)
    assert not source.ordered_target_consume


def test_ordered_ack_missing_after_send_does_not_replay_or_advance(monkeypatch):
    from types import SimpleNamespace

    import ray

    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    monkeypatch.setattr(ray, "get", lambda refs, **kwargs: refs)
    source = DeltaTransfer(DeltaSyncConfig(stream="s"), "cpu")
    source._applying = source.ordered_target_consume = True
    lane = dict(tensors=[dict(count=0)], target_shard=dict(entries=[0]))
    source._staged = [(0, [lane] * 4), (1, [lane] * 4)]
    events = []

    def dispatch(replica, action, **kwargs):
        events.append((replica, action))
        assert action == "consume" and kwargs["round_id"] == 0 and len(kwargs["following"]) == 1
        return dict(success=True, full_required=[], ordered_frames_ack=2 - replica)

    source.engines = tuple(
        SimpleNamespace(delta_weight_sync=SimpleNamespace(remote=lambda op, i=i, **kw: dispatch(i, op, **kw)))
        for i in range(2)
    )
    with pytest.raises(RuntimeError, match="Incomplete ordered"):
        source.apply()
    with pytest.raises(RuntimeError, match="Cannot replay"):
        source.apply()
    with pytest.raises(RuntimeError, match="Cannot finalize"):
        source.finalize()
    assert events == [(0, "consume"), (1, "consume")]
    assert len(source._staged) == 2
