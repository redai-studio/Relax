# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from relax.distributed.weight_sync.codec import encode, frame_flushes, validate_directory
from relax.distributed.weight_sync.tcp_transport import TcpFrameSource, receive


def test_config_rejects_staging_larger_than_receiver_budget(tmp_path):
    from relax.distributed.weight_sync.config import DeltaSyncConfig

    path = tmp_path / "delta.json"
    path.write_text('{"stream":"test","receiver_gpu_budget_bytes":16,"staging_budget_bytes":17}')
    with pytest.raises(ValueError, match="staging_budget_bytes exceeds receiver_gpu_budget_bytes"):
        DeltaSyncConfig.load(path)


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
            restored = (
                torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                if payload
                else torch.empty(0, dtype=torch.uint8)
            )
            spec, restored_positions, restored_values = next(frame_flushes(metadata, restored))
            assert spec["params"][0]["dtype"] == "bfloat16"
            assert torch.equal(restored_positions.view(torch.int32), positions)
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
    values = [value for _, _, value in frame_flushes(metadata, raw)]
    assert torch.equal(values[0].view(torch.int16), bf16.view(torch.int16))
    assert torch.equal(values[1].view(torch.int32), fp32.view(torch.int32))
    assert torch.equal(values[2], integer)
    metadata["flushes"][1]["offset"] = 0
    with pytest.raises(ValueError, match="flush boundaries"):
        validate_directory(metadata, max_bytes=1024)


def test_staged_tcp_prefix_is_immutable_bounded_and_never_applies_before_owner_barrier(monkeypatch):
    """Real TCP data with a CPU model-RPC boundary; no CUDA acceptance
    credit."""
    from types import SimpleNamespace

    import ray

    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    actions, buffers, consumed = [], {}, []

    def operation(action, **kwargs):
        actions.append(action)
        if action == "receive":
            key = kwargs["round_id"]
            buffers[key] = bytes(receive(kwargs["descriptor"], "1", timeout_s=3, max_bytes=16))
        elif action == "begin":
            assert not consumed
        elif action == "consume":
            assert "begin" in actions
            consumed.append((kwargs["round_id"], buffers.pop(kwargs["round_id"])))
            return dict(success=True, full_required=["weight"])
        elif action == "ready":
            return [dict(retained_payload_bytes=sum(map(len, buffers.values())))]

    monkeypatch.setattr(ray, "get", lambda refs, **kwargs: refs)
    transfer = DeltaTransfer(DeltaSyncConfig(stream="test", staging_budget_bytes=16), "cpu")
    transfer.engines = (SimpleNamespace(delta_weight_sync=SimpleNamespace(remote=operation)),)
    transfer.ranks, transfer.tcp_members = [[1]], {"1"}
    transfer.tcp = TcpFrameSource("127.0.0.1", timeout_s=3, host_budget_bytes=16)
    try:
        transfer.begin(version="1", base="0", seed=False, defer_apply=True)
        metadata, raw = encode([("weight", (2,), torch.float32, None, torch.ones(2))], max_bytes=16)
        expected = raw.numpy().tobytes()
        assert not transfer.submit_frame(metadata, raw, applying=False)
        raw.zero_()
        assert not transfer.drain_frame()
        transfer.stage(metadata, raw)
        assert not transfer.can_stage(metadata, raw)
        with pytest.raises(ValueError, match="capacity"):
            transfer.stage(metadata, raw)
        assert consumed == [] and actions == ["receive", "ready", "receive", "ready"]
        with pytest.raises(RuntimeError, match="generating model"):
            transfer.apply()
        with pytest.raises(RuntimeError, match="Cannot finalize"):
            transfer.finalize()
        transfer.activate()
        assert transfer.apply() == {"weight"}
        assert consumed == [(0, expected), (1, bytes(8))]
        assert transfer.can_stage(metadata, raw)
        assert transfer.send(metadata, raw) == {"weight"}
        assert consumed[-1] == (2, bytes(8))  # activate did not reset received round IDs.
        assert transfer.metrics["prestaged_frames"] == 2
        assert transfer.metrics["prestaged_bytes"] == 16
        with pytest.raises(RuntimeError, match="only once"):
            transfer.activate()
    finally:
        transfer.tcp.close()


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
            return {"success": True, "full_required": [f"name{kw['round_id']}"]}
        elif action == "finalize":
            return {"success": True, "force_full_seed": False}

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
        assert not transfer.submit_frame(metadata, raw, applying=True)
        assert receiving.wait(3)
        # Produce the next frame on the main thread while the socket is pending.
        raw.zero_()
        transfer.prepare_frame(raw)
        raw.fill_(99)
        with pytest.raises(RuntimeError, match="already waiting"):
            transfer.prepare_frame(raw)
        with pytest.raises(RuntimeError, match="Drain"):
            transfer.submit_frame(metadata, raw, applying=True)
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
            assert not transfer.submit_frame(metadata, raw, applying=True)
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


@pytest.mark.parametrize("failure", [None, "ready", "consume"])
def test_collective_install_ack_is_drained_before_next_publication(monkeypatch, failure):
    import weakref
    from threading import Event, get_ident
    from types import SimpleNamespace

    import ray

    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    entered, release = Event(), Event()
    producer = get_ident()
    actions = []

    def broadcast(raw, *, deadline):
        assert get_ident() == producer and transfer._active_raw is raw
        actions.append("broadcast")

    def operation(action, **kwargs):
        actions.append(action)
        if action in ("ready", "consume"):
            assert get_ident() != producer
            if action == "ready":
                entered.set()
                assert release.wait(5)
            if action == failure:
                raise RuntimeError(f"injected {failure} failure")
        if action == "consume":
            return {"success": True, "full_required": [f"ordinary{kwargs['round_id']}"]}
        if action == "finalize":
            return dict(success=True, force_full_seed=False, full_required_mask=[0, 0])
        return {}

    monkeypatch.setattr(ray, "get", lambda refs, **kwargs: refs)
    transfer = DeltaTransfer(DeltaSyncConfig(stream="test", transport="nccl_socket"), "cpu")
    transfer.group = SimpleNamespace(send=broadcast, close=lambda: None)
    transfer.engines = (SimpleNamespace(delta_weight_sync=SimpleNamespace(remote=operation)),)
    transfer.ranks, transfer.routes = [[1]], []
    transfer.connection = {"group_name": "test", "world_size": 2}
    transfer.begin(version="1", base=None, seed=True)
    directory, raw = encode([("w", (1,), torch.float32, None, torch.ones(1))], max_bytes=8)
    try:
        assert not transfer.submit_frame(directory, raw, applying=True)
        assert entered.wait(5)
        previous = weakref.ref(raw)
        del raw
        assert previous() is None
        directory, raw = encode([("next", (1,), torch.float32, None, torch.zeros(1))], max_bytes=8)
        assert actions == ["begin", "receive", "broadcast", "ready"]
        with pytest.raises(RuntimeError, match="Drain"):
            transfer.submit_frame(directory, raw, applying=True)
        with pytest.raises(RuntimeError, match="Drain"):
            transfer.finalize()
        release.set()
        if failure:
            with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
                transfer.drain_frame()
            with pytest.raises(RuntimeError, match="Drain"):
                transfer.finalize()
            with pytest.raises(RuntimeError, match="Drain"):
                transfer.connect([SimpleNamespace(_actor_id=b"replacement")], [1])
            assert actions[-1] == failure
        else:
            assert transfer.drain_frame() == {"ordinary0"}
            assert not transfer.submit_frame(directory, raw, applying=True)
            assert transfer.drain_frame() == {"ordinary1"}
            assert transfer.finalize()["frames"] == 2
            assert actions == ["begin"] + ["receive", "broadcast", "ready", "consume"] * 2 + ["finalize"]
    finally:
        release.set()
        transfer.close()


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
    _, _, decoded = next(frame_flushes(metadata, slab))
    assert torch.equal(decoded, second)
    arena.release(0)
    arena.release(1)
    assert arena.free == [(0, 64)]


@pytest.mark.parametrize(
    ("transport", "source_net", "peer_net", "reject"),
    [
        ("nccl_socket", None, None, False),
        ("nccl_socket", "IB", None, True),
        ("nccl_socket", None, "IB", True),
        ("nccl", "IB", "IB", False),
    ],
)
def test_collective_selects_network_only_for_unique_gpu_members(
    monkeypatch, tmp_path, transport, source_net, peer_net, reject
):
    import json
    import os
    from types import SimpleNamespace

    import ray

    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.gpu_transport import GpuSyncGroup
    from relax.distributed.weight_sync.transport import DeltaTransfer
    from relax.utils import distributed_utils

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"stream": "test", "transport": transport}))
    if source_net is None:
        monkeypatch.delenv("NCCL_NET", raising=False)
    else:
        monkeypatch.setenv("NCCL_NET", source_net)
    monkeypatch.setattr(ray, "get", lambda refs, **kwargs: refs)
    monkeypatch.setattr(ray._private.services, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="GPU-source"))
    monkeypatch.setattr(
        torch.distributed,
        "ProcessGroupNCCL",
        SimpleNamespace(Options=lambda: SimpleNamespace(config=SimpleNamespace())),
        raising=False,
    )
    groups = {}
    monkeypatch.setattr(
        distributed_utils, "init_process_group", lambda **kwargs: groups.update({kwargs["rank"]: kwargs})
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, group: None)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda group: None)

    def dispatch(op, **kwargs):
        if op == "prepare":
            return [dict(uuid="GPU-other", nccl_net=peer_net), dict(uuid="GPU-source", nccl_net="IB")]
        if op == "join":
            assert kwargs["ranks"] == [1, None]
            GpuSyncGroup(kwargs["connection"], rank=1, device="cpu", timeout_s=5)

    engine = SimpleNamespace(_actor_id=b"id", delta_weight_sync=SimpleNamespace(remote=dispatch))
    source = DeltaTransfer(DeltaSyncConfig.load(config_path), "cpu")
    try:
        if reject:
            with pytest.raises(ValueError, match="NCCL_NET"):
                source.connect([engine], [2])
            assert not groups and not source.can_reuse([engine], [2])
        else:
            assert source.connect([engine], [2]) and not source.connect([engine], [2])
            assert [route["actual_transport"] for route in source.routes] == [transport, "tcp"]
            assert set(groups) == {0, 1}
            for group in groups.values():
                assert group["backend"] == "nccl" and group["world_size"] == 2
                if transport == "nccl_socket":
                    assert group["pg_options"].config.net_name == "Socket"
                else:
                    assert group["pg_options"] is None
        assert os.getenv("NCCL_NET") == source_net
    finally:
        source.close()


def _broadcast_frames(rank, rendezvous):
    from datetime import timedelta
    from time import monotonic

    import torch.distributed as dist

    from relax.distributed.weight_sync.gpu_transport import GpuSyncGroup

    # Exercise the production broadcast and zero-byte behavior over actual Gloo.
    dist.init_process_group("gloo", rank=rank, world_size=3, init_method=rendezvous, timeout=timedelta(seconds=20))
    group = object.__new__(GpuSyncGroup)
    group.device, group.rank, group.group = torch.device("cpu"), rank, dist.group.WORLD
    group.fence = lambda works, *, deadline: [work.wait(timedelta(seconds=deadline - monotonic())) for work in works]
    try:
        for size in (17, 0, 3):
            expected = torch.arange(size, dtype=torch.uint8)
            if rank == 0:
                group.send(expected, deadline=monotonic() + 15)
            else:
                actual, works = group.receive({"bytes": size})
                group.fence(works, deadline=monotonic() + 15)
                assert torch.equal(actual, expected)
    finally:
        dist.destroy_process_group()


def test_collective_broadcasts_the_same_wire_to_every_member(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(_broadcast_frames, args=((tmp_path / "common-wire").as_uri(),), nprocs=3, start_method="fork")


def _expert_receivers(counts):
    width = 6
    engines = []
    member = 1
    for size in counts:
        rows = []
        for tp in range(size):
            rows.append(
                dict(
                    rank=member,
                    uuid=f"GPU-{member}",
                    tp_rank=tp,
                    pp_rank=0,
                    fallbacks={},
                    routes=[
                        dict(
                            prefix="model.mlp.experts",
                            expert_start=0,
                            expert_stop=1,
                            hidden_size=2,
                            intermediate_start=width * tp // size,
                            intermediate_stop=width * (tp + 1) // size,
                            intermediate_size=width,
                            dtype="float32",
                        )
                    ],
                )
            )
            member += 1
        engines.append(rows)
    return engines


def _directed_frames(rank, rendezvous, tp):
    from datetime import timedelta
    from time import monotonic

    import torch.distributed as dist

    from relax.distributed.weight_sync.expert_lanes import ExpertLanePlan
    from relax.distributed.weight_sync.gpu_transport import GpuSyncGroup

    dist.init_process_group(
        "gloo", rank=rank, world_size=1 + tp, init_method=rendezvous, timeout=timedelta(seconds=20)
    )
    group = object.__new__(GpuSyncGroup)
    group.device, group.rank, group.group = torch.device("cpu"), rank, dist.group.WORLD
    group.fence = lambda works, *, deadline: [work.wait(timedelta(seconds=deadline - monotonic())) for work in works]
    try:
        for coordinate in (0, 11):
            controls = [None]
            if rank == 0:
                metadata, raw = encode(
                    [
                        (
                            "model.mlp.experts.0.gate_proj.weight",
                            (6, 2),
                            torch.float32,
                            torch.tensor([coordinate], dtype=torch.int32),
                            torch.tensor([17.0]),
                        )
                    ],
                    max_bytes=128,
                )
                controls[0], slabs = ExpertLanePlan(_expert_receivers([tp])).partition(metadata, raw, 128, {})
                assert sum(slab.numel() > 0 for slab in slabs) == 1
            dist.broadcast_object_list(controls, src=0, group=group.group)
            if rank == 0:
                group.send_lanes(slabs, list(range(1, tp + 1)), deadline=monotonic() + 15)
            else:
                manifest = controls[0][rank - 1]
                slab, works = group.receive(dict(bytes=manifest["bytes"], expert_direct=True))
                group.fence(works, deadline=monotonic() + 15)
                validate_directory(manifest, max_bytes=128)
                if (rank == 1 and coordinate == 0) or (rank == tp and coordinate == 11):
                    [(spec, positions, values)] = list(frame_flushes(manifest, slab))
                    assert spec["expert_direct"] and spec["params"][0]["shape"] == [6 // tp, 2]
                    assert positions.view(torch.int32).tolist() == [coordinate % (12 // tp)]
                    assert values.tolist() == [17.0]
                else:
                    assert not manifest["tensors"] and not slab.numel()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp", [2, 3])
def test_collective_sends_sparse_expert_bytes_only_to_owning_rank(tmp_path, tp):
    import torch.multiprocessing as mp

    mp.start_processes(_directed_frames, args=((tmp_path / "lanes").as_uri(), tp), nprocs=1 + tp, start_method="fork")


def test_heterogeneous_replica_routes_use_admitted_ranks_and_final_ack(monkeypatch):
    from types import SimpleNamespace

    import ray

    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    expected = _expert_receivers([2, 3])
    transfer = DeltaTransfer(DeltaSyncConfig(stream="test", transport="nccl_socket", staging_budget_bytes=64), "cpu")
    transfer.ranks = [[row["rank"] for row in rows] for rows in expected]
    transfer.routes = [dict(row, delivery_id=row["rank"]) for rows in expected for row in rows]
    transfer.connection = dict(expert_direct=True, world_size=6, group_name="test")
    transfer.nccl_members = {str(rank): 6 - rank for rank in range(1, 6)}
    staged, consumed, sent = {}, [], []
    control = {}
    stale = False

    def rpc(engine, action, **kwargs):
        if action == "begin":
            control[engine] = kwargs["control"]
        elif action == "receive":
            staged[engine, kwargs["round_id"]] = [batch["bytes"] for batch in kwargs["batches"]]
        elif action == "consume":
            manifest = kwargs["metadata"]
            if isinstance(manifest, list):
                assert len(manifest) == len(expected[engine])
                assert [meta["bytes"] for meta in manifest] == staged[engine, kwargs["round_id"]]
                consumed.append(engine)
            return dict(success=True, full_required=[])
        elif action == "finalize":
            result = dict(success=True, force_full_seed=False, proxies=expected[engine])
            if control[engine]["seed"]:
                result.update(expert_routes=expected[engine], full_required_mask=[0])
            if stale and engine == 1:
                result["proxies"] = [dict(row, uuid="GPU-stale") for row in expected[engine]]
            return result

    monkeypatch.setattr(ray, "get", lambda refs, **kwargs: refs)
    transfer.engines = [
        SimpleNamespace(delta_weight_sync=SimpleNamespace(remote=lambda op, i=i, **kw: rpc(i, op, **kw)))
        for i in range(2)
    ]
    transfer.group = SimpleNamespace(
        send=lambda raw, **kw: None,
        send_lanes=lambda slabs, peers, **kw: sent.append(([s.numel() for s in slabs], peers)),
    )
    name = "model.mlp.experts.0.gate_proj.weight"
    transfer.begin(version="0", base=None, seed=True)
    full = encode([(name, (6, 2), torch.float32, None, torch.ones(12))], max_bytes=64)
    transfer.send(*transfer.route_frame(*full))
    first = transfer.finalize()
    assert first["nccl_broadcast_input_bytes"] == 48 and first["logical_delivery_bytes"] == 240
    assert all(v["expert_direct"] for v in control.values())

    transfer.begin(version="1", base="0", seed=False, defer_apply=True)
    for position in (0, 11):
        sparse = encode(
            [(name, (6, 2), torch.float32, torch.tensor([position], dtype=torch.int32), torch.ones(1))], max_bytes=64
        )
        metadata, raw = transfer.route_frame(*sparse)
        assert transfer.can_stage(metadata, raw)
        transfer.stage(metadata, raw)
    assert not consumed and sent == [([12, 0, 12, 0, 0], [5, 4, 3, 2, 1]), ([0, 12, 0, 0, 12], [5, 4, 3, 2, 1])]
    transfer.activate()
    transfer.apply()
    assert consumed == [0, 1, 0, 1]
    result = transfer.finalize()
    assert result["original_source_bytes"] == 24 and result["unpartitioned_delivery_bytes"] == 120
    assert result["nccl_p2p_send_bytes"] == result["logical_delivery_bytes"] == 48
    assert result["delivery_bytes_per_rank"] == {"1": 12, "2": 12, "3": 12, "4": 0, "5": 12}
    assert result["expert_direct_frames"] == 2 and result["direct_expert_input_bytes"] == 16
    assert result["prestaged_frames"] == 2 and result["nccl_broadcast_input_bytes"] == 0
    transfer.begin(version="2", base="1", seed=False)
    assert isinstance(transfer.route_frame(*sparse)[0], list)
    stale = True
    with pytest.raises(RuntimeError, match="GPU or rank identity"):
        transfer.finalize()
    assert transfer._expert_version == "1"


@pytest.mark.parametrize("mode", ["disabled", "tcp", "unknown", "stale"])
def test_broadcast_fallback_records_actual_expert_delivery_bytes(mode):
    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.expert_lanes import ExpertLanePlan
    from relax.distributed.weight_sync.transport import DeltaTransfer

    source = DeltaTransfer(DeltaSyncConfig(stream="test", direct_expert_delta=mode != "disabled"), "cpu")
    source.ranks = [[1, 2]]
    source.connection = {"expert_direct": mode != "tcp"}
    source._expert_plan = ExpertLanePlan(_expert_receivers([2]))
    source._expert_version = "stale" if mode == "stale" else "previous"
    source.begin(version="next", base="previous", seed=False, defer_apply=True)
    slot = "unknown" if mode == "unknown" else "model"
    frame = encode(
        [
            (
                f"{slot}.mlp.experts.0.gate_proj.weight",
                (6, 2),
                torch.float32,
                torch.tensor([0, 11], dtype=torch.int32),
                torch.ones(2),
            )
        ],
        max_bytes=64,
    )
    directory, raw = source.route_frame(*frame)
    assert directory is frame[0] and raw is frame[1]
    reasons = {
        "disabled": "direct_expert_delta_disabled",
        "tcp": "transport_requires_shared_frame",
        "unknown": "native_layout_unavailable",
        "stale": "missing_acknowledged_native_layout",
    }
    assert source.metrics["expert_fallback_delivery_bytes"] == {reasons[mode]: 32}


def test_missing_consume_ack_does_not_replay_or_advance(monkeypatch):
    from types import SimpleNamespace

    import ray

    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    monkeypatch.setattr(ray, "get", lambda refs, **kwargs: refs)
    source = DeltaTransfer(DeltaSyncConfig(stream="s"), "cpu")
    source._applying = True
    directory, raw = encode([("w", (1,), torch.float32, None, torch.ones(1))], max_bytes=8)
    source._staged = [(0, directory), (1, directory)]
    events = []

    def dispatch(replica, action, **kwargs):
        events.append((replica, action))
        assert action == "consume" and kwargs["round_id"] == 0
        return dict(success=replica == 0, full_required=[])

    source.engines = tuple(
        SimpleNamespace(delta_weight_sync=SimpleNamespace(remote=lambda op, i=i, **kw: dispatch(i, op, **kw)))
        for i in range(2)
    )
    with pytest.raises(RuntimeError, match="Incomplete receiver consume"):
        source.apply()
    with pytest.raises(RuntimeError, match="Cannot replay"):
        source.apply()
    with pytest.raises(RuntimeError, match="consume is unresolved"):
        source.stage(directory, raw)
    with pytest.raises(RuntimeError, match="Cannot finalize"):
        source.finalize()
    assert events == [(0, "consume"), (1, "consume")]
    assert len(source._staged) == 2


@pytest.mark.parametrize("bad", ["scalar", "absent", "long", "unwritten", "boolean", "failed", "missing_engine"])
def test_source_rejects_incomplete_final_mask_from_one_engine(monkeypatch, bad):
    from types import SimpleNamespace

    import ray

    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    transfer = DeltaTransfer(DeltaSyncConfig(stream="mask"), "cpu")
    transfer.connection = {}
    transfer.engines = [SimpleNamespace(delta_weight_sync=SimpleNamespace(remote=lambda *a, **k: None))] * 2
    monkeypatch.setattr(ray, "get", lambda *args, **kwargs: [{"success": True}] * 2)
    transfer.begin(version="2", base="1", seed=True)
    transfer._final_schema = {"a": ((2,), "float32"), "b": ((2,), "float32")}
    good = dict(success=True, force_full_seed=False, full_required_mask=[1, 0])
    damaged = dict(good)
    if bad == "failed":
        damaged["success"] = False
    elif bad == "absent":
        del damaged["full_required_mask"]
    else:
        damaged["full_required_mask"] = {
            "scalar": [0],
            "long": [0, 1, 0],
            "unwritten": [0, 255],
            "boolean": [0, True],
        }.get(bad)
    replies = [good] if bad == "missing_engine" else [good, damaged]
    monkeypatch.setattr(ray, "get", lambda *a, **k: replies)
    with pytest.raises(RuntimeError, match="Incomplete receiver final"):
        transfer.finalize()


@pytest.mark.parametrize(
    "failure", ["replaced_closed", "replaced_dead", "retained_dead", "nested_dead", "unavailable", "live_error"]
)
def test_membership_change_requires_retirement_of_every_surviving_receiver(monkeypatch, failure):
    from types import SimpleNamespace

    import ray

    from relax.distributed.weight_sync import transport
    from relax.distributed.weight_sync.config import DeltaSyncConfig

    prepared = []
    errors = {}
    closed = []
    source_shutdown = []

    def engine(actor_id):
        def operation(action, **kwargs):
            if action == "close":
                closed.append((actor_id, kwargs["abort_group"]))
                return errors.get(actor_id)
            if action == "prepare":
                prepared.append(actor_id)
                return [dict(uuid=f"GPU-{actor_id}", pp_size=1, tp_size=1)]
            return {}

        return SimpleNamespace(_actor_id=bytes([actor_id]), delta_weight_sync=SimpleNamespace(remote=operation))

    def get(ref, **kwargs):
        if isinstance(ref, Exception):
            raise ref
        return ref

    monkeypatch.setattr(ray, "get", get)
    monkeypatch.setattr(ray._private.services, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="GPU-source"))
    monkeypatch.setattr(
        transport,
        "GpuSyncGroup",
        lambda *args, **kwargs: SimpleNamespace(
            close=lambda: source_shutdown.append("close"), abort=lambda: source_shutdown.append("abort")
        ),
    )
    source = transport.DeltaTransfer(DeltaSyncConfig(stream="test", transport="nccl"), "cpu")
    old, retained, replacement = [engine(i) for i in range(3)]
    assert source.connect([old, retained], [1, 1])
    prepared.clear()
    error = {
        "replaced_closed": None,
        "replaced_dead": ray.exceptions.ActorDiedError(),
        "nested_dead": ray.exceptions.RayTaskError(
            "close", "nested failure", ray.exceptions.ActorDiedError()
        ).as_instanceof_cause(),
        "retained_dead": ray.exceptions.ActorDiedError(),
        "unavailable": ray.exceptions.ActorUnavailableError("network partition", None),
        "live_error": OSError("live proxy failed to close"),
    }[failure]
    errors[1 if failure == "retained_dead" else 0] = error
    try:
        if failure in {"replaced_closed", "replaced_dead"}:
            assert source.connect([replacement, retained], [1, 1])
            assert prepared == [2, 1]
        else:
            with pytest.raises(type(error)) as caught:
                source.connect([replacement, retained], [1, 1])
            assert caught.value is error and not prepared
        assert closed == [(0, True), (1, True)] and source_shutdown == ["abort"]
        if failure not in {"replaced_closed", "replaced_dead"}:
            # A failed teardown may already have closed the source group. An
            # identical old topology must not masquerade as a reusable group.
            with pytest.raises(type(error)):
                source.connect([old, retained], [1, 1])
            assert not prepared
    finally:
        errors.clear()
        source.close()


def test_abort_unregisters_only_the_owned_group_and_preserves_world(monkeypatch, tmp_path):
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d

    from relax.distributed.weight_sync.gpu_transport import GpuSyncGroup
    from relax.utils.distributed_utils import init_process_group

    dist.init_process_group("gloo", init_method=(tmp_path / "world").as_uri(), world_size=1, rank=0)
    try:
        name = "relax-delta-cpu"
        private = init_process_group(
            backend="gloo", init_method=(tmp_path / "private").as_uri(), rank=0, world_size=1, group_name=name
        )
        other = dist.new_group([0], backend="gloo")
        targets = []
        # CPU substitute for the backend terminal call; exercise actual c10d
        # dispatch, rank/name cleanup and the default group's continuing use.
        monkeypatch.setattr(c10d.ProcessGroup, "abort", lambda pg: targets.append(pg))
        owner = object.__new__(GpuSyncGroup)
        owner._group_name = name
        for invalid in (None, dist.group.WORLD, dist.GroupMember.NON_GROUP_MEMBER, other):
            owner.group = invalid
            with pytest.raises(RuntimeError, match="owned delta communicator"):
                owner.abort()
        assert not targets
        owner.group = private
        owner.abort()
        assert targets == [private] and private not in c10d._world.pg_map
        assert private not in c10d._world.pg_names and private not in c10d._world.pg_group_ranks
        for group in (other, dist.group.WORLD):
            value = torch.ones(1)
            dist.all_reduce(value, group=group)
            assert value.item() == 1
    finally:
        dist.destroy_process_group()
