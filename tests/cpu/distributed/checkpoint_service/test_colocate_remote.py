# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import base64
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from relax.distributed.checkpoint_service.backends.shard_p2p.colocate_remote import send_remote_shards
from relax.distributed.checkpoint_service.backends.shard_p2p.remote.plan import (
    PACKED_TARGET_ENCODING,
    packed_target_regions,
)
from relax.distributed.checkpoint_service.backends.shard_p2p.remote.transport import BufferedWriter


@pytest.mark.parametrize("fails", [False, True])
def test_async_registration_waits_once_and_retains_partial_failure(fails):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from relax.distributed.checkpoint_service.backends.shard_p2p.remote import transport

    entered, proceed, validating = Event(), Event(), Event()
    parameters = {"weight": torch.zeros(4)}

    def register(value):
        entered.set()
        assert proceed.wait(5)
        if fails:
            raise RuntimeError("native registration failed")

    agent = SimpleNamespace(
        get_reg_descs=lambda values: values, register_memory=Mock(side_effect=register), deregister_memory=Mock()
    )
    memory = transport.RegisteredMemory(agent, parameters)
    before = len(transport._FAILED_REGISTRATIONS)
    try:
        memory.register(batched=True, asynchronous=True)
        assert entered.wait(5)
        registration = memory._registration_future
        wait_for_registration = registration.result

        def wait_at_registration(*args, **kwargs):
            validating.set()
            return wait_for_registration(*args, **kwargs)

        with ThreadPoolExecutor(max_workers=1) as pool:
            with patch.object(registration, "result", side_effect=wait_at_registration):
                future = pool.submit(memory.validate, parameters)
                assert validating.wait(5) and not future.done()
                proceed.set()
            if fails:
                with pytest.raises(RuntimeError, match="native registration failed"):
                    future.result(timeout=5)
                with pytest.raises(RuntimeError, match="native registration failed"):
                    memory.release()
                assert memory is transport._FAILED_REGISTRATIONS[-1] and memory.parameters
                agent.deregister_memory.assert_not_called()
            else:
                future.result(timeout=5)
                memory.validate(parameters)
                memory.release()
                agent.deregister_memory.assert_called_once()
        agent.register_memory.assert_called_once()
    finally:
        proceed.set()
        del transport._FAILED_REGISTRATIONS[before:]


def test_partial_writer_registration_retains_all_memory_and_agent(monkeypatch):
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote import transport

    original = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: original(*a, **(kw | {"pin_memory": False})))
    registered = object()
    agent = SimpleNamespace(register_memory=Mock(side_effect=[registered, RuntimeError("second registration")]))
    before = len(transport._FAILED_WRITERS)
    try:
        with pytest.raises(RuntimeError, match="second registration"):
            BufferedWriter(16, agent=agent)
        assert len(transport._FAILED_WRITERS) == before + 1
        retained = transport._FAILED_WRITERS[-1]
        assert retained.failed and retained.agent is agent
        assert retained.registrations == [registered] and len(retained.buffers) == 2
        assert all(buffer.numel() == 16 for buffer in retained.buffers)
    finally:
        del transport._FAILED_WRITERS[before:]


@pytest.mark.parametrize("deregister_fails", [False, True])
def test_idle_staging_release_rebuilds_or_retains_uncertain_storage(monkeypatch, deregister_fails):
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote import transport

    original = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: original(*a, **(kw | {"pin_memory": False})))
    agent = SimpleNamespace(register_memory=Mock(side_effect=lambda value: object()), deregister_memory=Mock())
    writer = BufferedWriter(16, agent=agent)
    writer.targets = {"target": ("peer", "epoch")}
    registrations = list(writer.registrations)
    before = len(transport._FAILED_WRITERS)
    try:
        if deregister_fails:
            agent.deregister_memory.side_effect = [None, RuntimeError("deregister")]
            with pytest.raises(RuntimeError, match="deregister"):
                writer.release_buffers()
            assert writer.failed and writer.registrations == registrations and len(writer.buffers) == 2
            assert transport._FAILED_WRITERS[-1] is writer
            with pytest.raises(RuntimeError, match="unconfirmed"):
                writer.acquire()
        else:
            writer.release_buffers()
            assert not writer.buffers and not writer.registrations
            _, value = writer.acquire()
            assert value.numel() == 16 and len(writer.buffers) == 2
            assert all(value not in registrations for value in writer.registrations)
            assert writer.agent is agent and writer.targets == {"target": ("peer", "epoch")}
            writer.release_buffers()
    finally:
        writer.pool.shutdown(wait=True)
        del transport._FAILED_WRITERS[before:]


@pytest.mark.parametrize("owns_buffer", [True, False])
def test_ack_releases_colocate_staging_collectively(owns_buffer):
    from relax.distributed.checkpoint_service.backends.shard_p2p.synchronizer import (
        ShardP2PSynchronizer,
        _PendingUpdate,
    )

    core = object.__new__(ShardP2PSynchronizer)
    held, observed = {"source": object()}, []
    core.failed, core._has_remote, core._has_ipc = False, True, False
    core._pending = _PendingUpdate(1, held)
    core._wait_for_receivers = lambda *a, **kw: observed.append("ACK")

    def release():
        assert observed == ["ACK", "collective"] and core.retained_sources is held
        observed.append("release")

    def phase(action):
        observed.append("collective")
        return action()

    core._phase = phase
    core._writer = SimpleNamespace(release_buffers=release) if owns_buffer else None
    core.finish(version=1)
    assert observed == ["ACK", "collective"] + (["release"] if owns_buffer else [])
    assert core.completed_version == 1 and core.retained_sources is None


@pytest.mark.parametrize("peer_count", [1, 2])
def test_remote_range_dispatch_uses_completion_batch_for_every_peer_count(monkeypatch, peer_count):
    writer = object.__new__(BufferedWriter)
    writer.failed, writer.timeout, writer.profile = False, 120, {}
    writer.targets = {str(i): (f"peer-{i}", "epoch") for i in range(peer_count)}
    writer.agent = SimpleNamespace(get_xfer_descs=Mock(side_effect=lambda values, kind: (values, kind)))
    writer._transfer_many = Mock()
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    tensor = SimpleNamespace(device=torch.device("cuda:0"), data_ptr=lambda: 100)
    writes = [
        (dict(target=str(i), param="w", rows=2, row_bytes=4, dst_stride=4, dst_offset=0), tensor)
        for i in range(peer_count)
    ]
    manifests = {str(i): dict(gpu_id=i, params={"w": dict(address=200)}) for i in range(peer_count)}
    ready = SimpleNamespace(synchronize=Mock())
    writer._write_ranges(writes, manifests, ready)
    requests, retained_sources, deadline = writer._transfer_many.call_args.args
    assert [request[2] for request in requests] == [f"peer-{i}" for i in range(peer_count)]
    assert retained_sources is writes and deadline > 0
    assert ready.synchronize.call_count == 1 and not writer.failed


@pytest.mark.parametrize("capacity", [10, 32])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("target_count", [1, 2])
def test_remote_chunks_offsets_tail_and_connection_reuse(capacity, packed, target_count):
    source = torch.arange(48, dtype=torch.bfloat16)[8:32]
    tensor = SimpleNamespace(is_cuda=True, dtype=source.dtype, is_contiguous=lambda: True, view=source.view)
    targets = [torch.zeros(48, dtype=torch.uint8) for _ in range(target_count)]
    buffer = torch.empty(capacity, dtype=torch.uint8)
    writer = object.__new__(BufferedWriter)
    writer.agent = SimpleNamespace(
        add_remote_agent=Mock(side_effect=lambda value: value.decode()), remove_remote_agent=Mock()
    )
    writer.targets, writer.target_registration_keys = {}, {}
    writer._active_range_batch = None
    writer.failed, writer._retained_failure = False, False
    writer.acquire = lambda: (0, buffer)

    def submit(index, writes, manifests):
        for write, payload in writes:
            for row in range(write["rows"]):
                start = write["dst_offset"] + row * write["dst_stride"]
                targets[int(write["target"])][start : start + write["row_bytes"]].copy_(
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
    plans, manifests = {}, {}
    for index in range(target_count):
        key = str(index)
        target_write = dict(write, id=f"w{index}", src_offset=2 * index)
        plans[key] = dict(writes=[target_write], plan_hash=f"plan-{key}")
        manifests[key] = dict(agent_metadata=base64.b64encode(key.encode()).decode())
        if packed:
            manifests[key].update(
                target_memory_type="DRAM",
                target_encoding=PACKED_TARGET_ENCODING,
                packed_regions=packed_target_regions([target_write], {"w": 48}),
                params={"w": dict(wire_nbytes=32)},
            )
    for version in (1, 2, 3):
        for manifest in manifests.values():
            manifest.update(generation=f"epoch-{version}", persistent_registration_key="b" if version == 3 else "a")
        receipts, _ = send_remote_shards({"s": tensor}, plans, manifests, rank=3, version=version, writer=writer)
        assert receipts == {
            key: dict(
                source_rank=3,
                version=version,
                generation=f"epoch-{version}",
                plan_hash=f"plan-{key}",
                writes=[f"w{key}"],
            )
            for key in plans
        }
    assert writer.agent.add_remote_agent.call_count == 2 * target_count
    assert writer.agent.remove_remote_agent.call_count == target_count
    for index, target in enumerate(targets):
        first = source.view(torch.uint8)[2 * index : 2 * index + 16]
        last = source.view(torch.uint8)[24 + 2 * index : 40 + 2 * index]
        expected = torch.zeros(48, dtype=torch.uint8)
        if packed:
            expected[:16], expected[16:32] = first, last
        else:
            expected[4:20], expected[24:40] = first, last
        assert torch.equal(target, expected)


def test_packed_target_regions_are_plan_order_independent_and_keep_logical_holes():
    first = dict(id="a", param="w", rows=3, row_bytes=4, dst_offset=2, dst_stride=12)
    last = dict(id="b", param="w", rows=1, row_bytes=6, dst_offset=38, dst_stride=6)
    regions = packed_target_regions([last, first], {"w": 48})
    assert regions == packed_target_regions([first, last], {"w": 48})
    assert regions["w"]["a"] == dict(offset=0, dst_offset=2, dst_stride=12, rows=3, row_bytes=4)
    assert regions["w"]["b"]["offset"] == 12
    with pytest.raises(ValueError):
        packed_target_regions([first, first], {"w": 48})
    with pytest.raises(ValueError):
        packed_target_regions([first, last | dict(dst_offset=44)], {"w": 48})


def test_cached_epoch_layout_keeps_fresh_addresses_and_rejects_changed_wire(monkeypatch):
    import json

    from relax.distributed.checkpoint_service.backends.shard_p2p import colocate_remote, synchronizer
    from relax.distributed.checkpoint_service.backends.shard_p2p.plan import fingerprint
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.plan import ReceiptValidator

    write = dict(id="slice", param="w", source_rank=0, dst_offset=0, dst_stride=8, rows=2, row_bytes=4)
    plan = dict(writes=[write])
    regions = packed_target_regions(plan["writes"], {"w": 16})
    core = object.__new__(synchronizer.ShardP2PSynchronizer)
    core.rank, core.group, core.session, core.engines = 0, "gloo", "session", [object()]
    core.geometry = dict(tp_size=1)
    core.persistent_targets = core._replica_enabled = False
    core.staging_bytes, core._agent_future = 64, None
    core._receiver_plan_hashes = ("delivery",)
    core._remote_encodings = {"0-0": PACKED_TARGET_ENCODING}
    core._remote_plans = {"0-0": plan}
    core._remote_local_plans = {"0-0": dict(plan, packed_regions=regions)}
    core._remote_layouts = {"0-0": {"w": dict(nbytes=16)}}
    core._remote_manifest_layouts = {("0-0", PACKED_TARGET_ENCODING): ({"w"}, fingerprint(regions), {"w": 8})}
    core._remote_validators = {"0-0": ReceiptValidator(plan)}
    core._remote_expected_by_owner = ({"0-0"},)
    core._phase, core._gather = lambda action: action(), lambda value: [value]

    def broadcast(packet, **kwargs):
        assert "packed_regions" not in packet[0]["0-0"]

    monkeypatch.setattr(synchronizer.dist, "broadcast_object_list", broadcast)
    manifest = dict(
        tp_rank=0,
        gpu_id=0,
        target_device_id=0,
        target_memory_type="DRAM",
        persistent_registration_key="a" * 32,
        target_encoding=PACKED_TARGET_ENCODING,
        packed_regions_hash=fingerprint(regions),
        agent_metadata="YWdlbnQ=",
        params={"w": dict(address=100, dtype=str(torch.bfloat16), nbytes=16, wire_nbytes=8)},
    )
    core._fanout = lambda commands: [dict(success=True, message=json.dumps(dict(shard_p2p_epochs=[manifest])))]

    def send(sources, plans, manifests, *, version, **kwargs):
        assert manifests["0-0"]["params"]["w"]["address"] == manifest["params"]["w"]["address"]
        assert manifests["0-0"]["packed_regions"] == regions
        receipt = dict(
            source_rank=0,
            version=version,
            generation=manifest["generation"],
            plan_hash=core._remote_validators["0-0"].plan_hash,
            writes=["slice"],
        )
        return {"0-0": receipt}, None

    monkeypatch.setattr(colocate_remote, "send_remote_shards", send)
    for version in (1, 2):
        manifest.update(version=version, generation=f"epoch-{version}")
        manifest["params"]["w"]["address"] = version * 100
        core._send_remote({}, version)
    changed = {"w": {"slice": dict(regions["w"]["slice"], offset=2)}}
    manifest["packed_regions_hash"] = fingerprint(changed)
    with pytest.raises(RuntimeError, match="epoch differs"):
        core._send_remote({}, 2)
