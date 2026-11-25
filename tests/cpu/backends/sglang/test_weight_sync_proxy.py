# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections import deque
from time import monotonic
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang.weight_sync_proxy import WeightSyncProxy


pytest_plugins = ["tests.cpu.backends.sglang.test_router_registration"]


def test_proxy_consumes_entire_failed_reply_batch_before_reporting_error():
    class Channel:
        def __init__(self):
            self.replies = deque([dict(success=False, error="pending IPC"), dict(success=True)])

        def poll(self, timeout):
            return bool(self.replies)

        def recv(self):
            return self.replies.popleft()

    proxy = WeightSyncProxy(SimpleNamespace())
    proxy.children = [(None, Channel()), (None, Channel())]
    with pytest.raises(RuntimeError, match="batch failed"):
        proxy._receive_replies(deadline=monotonic() + 1)
    assert proxy._receive_replies(deadline=monotonic() + 1) == [dict(success=True), dict(success=True)]


@pytest.fixture
def native_serializer():
    return pytest.importorskip("sglang.srt.utils").MultiprocessingSerializer


def _metadata(entries, *, entropy=0):
    import base64
    import random

    from relax.distributed.weight_sync.codec import encode

    rng = random.Random(709)
    names = [f"model.layers.0.mlp.experts.{index}.down_proj.weight" for index in range(entries)]
    if entropy:
        names = [name + base64.b64encode(rng.randbytes(entropy)).decode("ascii") for name in names]
    indices = torch.empty(0, dtype=torch.int32)
    values = torch.empty(0, dtype=torch.bfloat16)
    directory, _ = encode(
        [(name, (2048, 768), values.dtype, indices, values) for name in names],
        max_bytes=16 << 20,
    )
    return directory


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_inline_candidate_preflights_actual_whole_engine_serializer_on_both_sides_of_limit(tp, native_serializer):
    from relax.backends.sglang import weight_sync_proxy as wire

    proxy = WeightSyncProxy(None)
    proxy.children = [None] * tp

    # Vary only the canonical base64 length. The actual native serializer
    # accounts for the frame control, two reducers and the whole-engine JSON.
    def measure(n):
        ctl = dict(
            op="consume",
            version="123",
            frame=17,
            metadata_id="a" * 64,
            metadata_zlib=dict(bytes=101000, data="A" * (4 * n)),
        )
        body = proxy._payload([wire._serialize(ctl, wire._CudaWireSize(), wire._CudaWireSize()) for _ in range(tp)])
        return ctl, wire._http_bytes(body)

    lo, hi = 0, wire.MAX_HTTP_BYTES // 4
    while lo + 1 < hi:
        n = (lo + hi) // 2
        if measure(n)[1] <= wire.MAX_HTTP_BYTES:
            lo = n
        else:
            hi = n
    good, low = measure(lo)
    bad, high = measure(hi)
    assert 0 <= wire.MAX_HTTP_BYTES - low < 12 * tp and 0 < high - wire.MAX_HTTP_BYTES <= 12 * tp
    assert proxy._fits_ipc(good) and not proxy._fits_ipc(bad)


def test_large_inline_preview_falls_back_for_whole_engine_before_export(monkeypatch, native_serializer):
    from relax.backends.sglang import weight_sync_proxy as wire

    proxy = WeightSyncProxy(None)
    proxy.children = [None] * 4
    proxy.version, proxy.batch = "3", 0
    proxy.metrics = {"update_directory_calls": 0, "consume_calls": 0}
    proxy._inline_frame_directory = True
    metadata = _metadata(64, entropy=140)
    data = wire.json_bytes(metadata)
    assert wire._compressed_directory(data, 4) is not None
    # The lower bound fits, but the fully serialized TP4 request exceeds 64 KiB.
    events = []

    def commands(cmds, *, deadline):
        operation = cmds[0]["op"]
        events.append(operation)
        if operation == "export":
            assert all("metadata_zlib" not in c["control"] for c in cmds)
            return [dict(serialized="ipc")] * 4
        if operation == "status":
            flags = [0] * 64
            flags[-1] = 1
            return [dict(full_required=flags)] + [dict(full_required=[0] * 64)] * 3
        return [{}] * 4

    monkeypatch.setattr(proxy, "_controls", lambda controls, **kw: events.append("document"))
    monkeypatch.setattr(proxy, "_commands", commands)
    monkeypatch.setattr(proxy, "_request", lambda *args, **kw: events.append("payload"))
    result = proxy.consume(0, metadata, timeout=30)
    assert events[:-4] == ["document"] * proxy.metrics["update_directory_calls"] and events[0] == "document"
    assert events[-4:] == ["export", "payload", "status", "release"]
    assert proxy.metrics["inline_directory_fallback_frames"] == 1
    assert result["full_required"] == [metadata["tensors"][-1]["name"]]
    assert wire._compressed_directory(wire.json_bytes(_metadata(160, entropy=140)), 4) is None


def test_ipc_manifest_admission_charges_every_verl_flush(monkeypatch):
    import zlib

    from relax.backends.sglang import weight_sync_proxy as wire
    from relax.distributed.weight_sync.codec import encode

    dense = torch.ones(128, dtype=torch.bfloat16)
    positions = torch.arange(8, dtype=torch.int32)
    values = torch.zeros(8)
    metadata, _ = encode(
        [
            ("dense", (128,), dense.dtype, None, dense),
            ("sparse", (256,), values.dtype, positions, values),
        ],
        max_bytes=4096,
    )
    proxy = WeightSyncProxy(None)
    proxy.children = [None, None]
    proxy.tp_size, proxy.pp_size = 1, 2
    proxy._inline_frame_directory = True
    documents = []
    # Drive the already-tested HTTP serializer boundary into IPC or fragments.
    monkeypatch.setattr(proxy, "_fits_ipc", lambda control: "metadata_zlib" not in control)
    monkeypatch.setattr(proxy, "_send_frame_directory", lambda data, *args, **kw: documents.append(data))
    data = wire.json_bytes(metadata)
    compressed = zlib.compress(data, level=1)
    status_bytes = len(metadata["tensors"]) + len(compressed) + 1
    allocation = ((status_bytes + 511) // 512) * 512
    # 1,280 dense decoder bytes, 4,096 bytes for the largest slot's scratch,
    # and 64 bytes for int64 positions, in addition to the IPC status slab.
    proxy._workspace_budget = 1280 + 4096 + 64 + allocation - 1
    controls, inline, tail = proxy._shared_controls(metadata, monotonic() + 1)
    assert not inline and not tail and all("metadata_ipc" not in control for control in controls)
    assert documents == [data]
    proxy._workspace_budget += 1
    controls, inline, tail = proxy._shared_controls(metadata, monotonic() + 1)
    assert not inline and tail == compressed
    assert all(control["metadata_ipc"] == [2, len(compressed), len(data)] for control in controls)
    assert documents == [data]


def test_inline_roundtrip_uses_existing_loader_ipc_status_and_version(monkeypatch, native_serializer):
    import json

    import torch

    from relax.backends.sglang import delta_loader
    from relax.backends.sglang import weight_sync_proxy as wire
    from relax.distributed.weight_sync.codec import encode

    class Matrix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16), requires_grad=False) for _ in range(40)]
            )

        def load_weights(self, items):
            for name, value in items:
                self.weights[int(name.split(".")[5])].copy_(value.T)

    for encoding in ("documents", "inline", "ipc"):
        inline = encoding != "documents"
        models = [Matrix() for _ in range(4)]
        proxy = WeightSyncProxy(None)
        assert proxy._inline_frame_directory is False
        proxy._inline_frame_directory = inline
        if encoding == "ipc":
            fits = proxy._fits_ipc
            monkeypatch.setattr(proxy, "_fits_ipc", lambda ctl: "metadata_zlib" not in ctl and fits(ctl))
        proxy.children = [None] * 4
        proxy.identities = [dict(uuid=f"cpu-{tp}") for tp in range(4)]
        events = []
        statuses = []

        def request(endpoint, body, *, deadline):
            if endpoint == "update_weight_version":
                assert all(m._relax_delta_receiver.version == body["new_version"] for m in models)
                return
            for m, serialization in zip(models, body["serialized_named_tensors"], strict=True):
                items = native_serializer.deserialize(serialization)
                ctl = json.loads(items[0][0])
                if ctl["op"] == "consume":
                    assert ("metadata_zlib" in ctl) == (encoding == "inline")
                    assert ("metadata_ipc" in ctl) == (encoding == "ipc")
                    assert torch.equal(items[0][1], raw)
                delta_loader.load_delta_weights(m, items)
            events.append(ctl["op"])

        def commands(cmds, *, deadline):
            first = cmds[0]
            if first["op"] == "export":
                acks = [wire._status_buffer(c["count"], c.get("directory_tail", b""), "cpu") for c in cmds]
                statuses[:] = [ack[: c["count"]].fill_(99) for ack, c in zip(acks, cmds, strict=True)]
                events.append("export")
                return [
                    dict(serialized=wire._serialize(c["control"], raw, ack)) for c, ack in zip(cmds, acks, strict=True)
                ]
            if first["op"] == "status":
                events.append("status")
                return [dict(full_required=s.tolist()) for s in statuses]
            events.append(first["op"])
            return [dict(success=True)] * 4

        monkeypatch.setattr(proxy, "_request", request)
        monkeypatch.setattr(proxy, "_commands", commands)
        words = torch.tensor([0x7FC1, 0, 7, -4], dtype=torch.int16)
        for version, seed in (("0", True), ("1", False)):
            proxy.begin(
                dict(
                    version=version,
                    base=None if seed else "0",
                    seed=seed,
                    receiver_gpu_budget_bytes=32768,
                    staging_budget_bytes=1024,
                ),
                timeout=30,
            )
            indices = None if seed else torch.tensor([1], dtype=torch.int32)
            if not seed:
                words[1] = -32768
            vals = (words if seed else words[1:2]).view(torch.bfloat16)
            metadata, raw = encode(
                [(item["name"], (2, 2), torch.bfloat16, indices, vals) for item in _metadata(40)["tensors"]],
                max_bytes=8192,
            )
            events.clear()
            assert proxy.consume(0, metadata, timeout=30) == dict(success=True, full_required=[])
            assert events[-4:] == ["export", "consume", "status", "release"]
            assert bool(events.count("document")) is (not inline)
            if encoding == "ipc":
                assert proxy.metrics["ipc_directory_frames"] == 1 and proxy.metrics["update_directory_calls"] == 0
                assert proxy.metrics["ipc_directory_bytes"] > 0
            result = proxy.finalize(version, timeout=30)
            assert not result["force_full_seed"]
            assert result["metrics"]["finalize_s"] >= result["metrics"]["finalize_ack_s"] >= 0
            for model in models:
                assert model._relax_delta_receiver.version == version
                assert all(torch.equal(p.view(torch.int16), words.reshape(2, 2).T) for p in model.parameters())


def test_inline_ambiguous_payload_error_does_not_start_legacy_delivery(monkeypatch, native_serializer):
    proxy = WeightSyncProxy(None)
    proxy.children = [None] * 4
    proxy.version, proxy.batch = "2", 1
    proxy._inline_frame_directory = True
    proxy.metrics = {"consume_calls": 0, "update_directory_calls": 0}
    events = []

    def export(commands, **kw):
        events.append(commands[0]["op"])
        assert all("metadata_zlib" in c["control"] for c in commands)
        return [dict(serialized="ipc")] * 4

    def failed(*args, **kw):
        events.append("payload")
        raise TimeoutError("ambiguous delivery")

    monkeypatch.setattr(proxy, "_controls", lambda *a, **kw: pytest.fail("unexpected legacy delivery"))
    monkeypatch.setattr(proxy, "_commands", export)
    monkeypatch.setattr(proxy, "_request", failed)
    with pytest.raises(TimeoutError, match="ambiguous"):
        proxy.consume(1, _metadata(160), timeout=30)
    assert events == ["export", "payload"] and proxy.batch == 1 and proxy.metrics["consume_calls"] == 0


def test_missing_final_ack_keeps_human_version_unpublished(monkeypatch):
    proxy = WeightSyncProxy(None)
    proxy.children = [None] * 4
    proxy.version, proxy.batch, proxy.staging_budget = "0", 1, 512
    events = []

    def commands(cmds, *, deadline):
        op = cmds[0]["op"]
        events.append(op)
        if op == "export":
            assert all(c["count"] == 1 for c in cmds)
            return [dict(serialized="handle")] * 4
        if op == "status":
            return [dict(full_required=[0])] * 3 + [dict(full_required=[255])]
        pytest.fail(op)

    monkeypatch.setattr(proxy, "receive", lambda *a, **k: events.append("receive"))
    monkeypatch.setattr(proxy, "ready", lambda *a, **k: events.append("ready"))
    monkeypatch.setattr(proxy, "_commands", commands)
    monkeypatch.setattr(proxy, "_request", lambda op, *a, **k: events.append(op))
    with pytest.raises(RuntimeError, match="Incomplete"):
        proxy.finalize("0", timeout=20)
    with pytest.raises(RuntimeError, match="ambiguous"):
        proxy.finalize("0", timeout=20)
    assert events == ["receive", "ready", "export", "update_weights_from_tensor", "status"]


def test_pipeline_ipc_opens_exactly_the_local_stage_payload(monkeypatch, native_serializer):
    from sglang.srt.distributed import parallel_state

    from relax.backends.sglang.delta_pipeline import local_payload

    tp = SimpleNamespace(world_size=3, rank_in_group=0)
    pp = SimpleNamespace(world_size=2, rank_in_group=0)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: tp)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: pp)
    inner = [native_serializer.serialize((p, t), output_str=True) for p in range(2) for t in range(3)]
    # An invalid pickle on the other stage must not even be opened.
    inner[4] = "AAAA"
    proxy = WeightSyncProxy(None)
    proxy.tp_size, proxy.pp_size = tp.world_size, pp.world_size
    routed = proxy._payload(inner)["serialized_named_tensors"]
    for tp.rank_in_group in range(3):
        assert local_payload(native_serializer.deserialize(routed[tp.rank_in_group])) == (0, tp.rank_in_group)
    pp.rank_in_group, tp.rank_in_group = 1, 2
    assert local_payload(native_serializer.deserialize(routed[2])) == (1, 2)
    with pytest.raises(ValueError, match="topology"):
        local_payload(native_serializer.deserialize(routed[0]))


def test_pipeline_completion_observes_flush_and_rejects_aliases():
    import json

    import torch

    from relax.backends.sglang.delta_pipeline import _install

    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    before = model.weight.detach().view(torch.uint8).clone()
    raw = torch.empty(0, dtype=torch.uint8)
    status = torch.zeros(1, dtype=torch.uint8)
    control = (json.dumps(dict(op="confirm_flush", version="2")), raw)
    receiver = model._relax_delta_receiver = SimpleNamespace(version="1", failed=False, active=None)
    with pytest.raises(ValueError, match="completed"):
        _install(model, [control, ("__delta_capability__", status)])
    assert status.item() == 0
    receiver.version = "2"
    with pytest.raises(ValueError):
        _install(model, [control, ("__delta_capability__", model.weight.detach().view(torch.uint8)[0, :1])])
    assert torch.equal(model.weight.detach().view(torch.uint8), before)
    _install(model, [control, ("__delta_capability__", status)])
    assert status.item() == 1


def test_missing_ordinary_ack_keeps_ipc_storage_and_old_version(monkeypatch):
    from relax.backends.sglang.weight_sync_proxy import _status_buffer

    assert _status_buffer(2, bytes([17, 0]), "cpu").tolist() == [255, 255, 17, 0]
    proxy = WeightSyncProxy(None)
    proxy.children = [(None, None)]
    proxy.version, proxy.batch, proxy.staging_budget = "2", 0, 1024
    operations = []

    def commands(cmds, *, deadline):
        op = cmds[0]["op"]
        operations.append(op)
        return [dict(serialized="ipc")] if op == "export" else [dict(full_required=[255])]

    monkeypatch.setattr(proxy, "_commands", commands)
    monkeypatch.setattr(proxy, "_request", lambda action, *args, **kwargs: operations.append(action))
    monkeypatch.setattr(proxy, "receive", lambda *args, **kwargs: None)
    monkeypatch.setattr(proxy, "ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(proxy, "_shared_controls", lambda *args: ([{}], False, b""))
    with pytest.raises(RuntimeError, match="install acknowledgement"):
        proxy.consume(0, _metadata(1), timeout=10)
    with pytest.raises(RuntimeError, match="ambiguous"):
        proxy.finalize("2", timeout=10)
    with pytest.raises(RuntimeError, match="ambiguous"):
        proxy.consume(0, _metadata(1), timeout=10)
    assert operations == ["export", "update_weights_from_tensor", "status"]


@pytest.mark.parametrize("fault", ["short", "unwritten", "missing_stage", "flush_result", "flush_timeout"])
def test_pipeline_final_mask_requires_every_stage_and_flush_before_release(monkeypatch, fault):
    proxy = WeightSyncProxy(None)
    proxy.tp_size, proxy.pp_size = 1, 2
    proxy.children = [None, None]
    proxy.version, proxy.batch, proxy.staging_budget, proxy._workspace_budget = "2", 2, 128, 128
    operations = []
    confirmations = []

    def commands(cmds, *, deadline):
        op = cmds[0]["op"]
        operations.append(op)
        if op == "export":
            control = cmds[0]["control"]
            confirm = control["op"] == "confirm_flush"
            confirmations.append(confirm)
            assert all(c["count"] == (0 if confirm else 3) for c in cmds)
            return [dict(serialized="ipc")] * 2
        if op == "status":
            if confirmations[-1]:
                if fault == "flush_timeout":
                    raise TimeoutError
                return [dict(full_required=[]), dict(full_required=[1])]
            if fault == "missing_stage":
                return [dict(full_required=[0, 0, 0])]
            second = {"short": [0, 1], "unwritten": [0, 1, 255]}.get(fault, [0, 1, 1])
            return [dict(full_required=[0, 0, 0]), dict(full_required=second)]
        pytest.fail(op)

    monkeypatch.setattr(proxy, "receive", lambda *a, **kw: None)
    monkeypatch.setattr(proxy, "ready", lambda *a, **kw: None)
    monkeypatch.setattr(proxy, "_payload", lambda *a, **kw: {})
    monkeypatch.setattr(proxy, "_request", lambda action, *a, **kw: operations.append(action))
    monkeypatch.setattr(proxy, "_commands", commands)
    with pytest.raises((RuntimeError, TimeoutError)):
        proxy.finalize("2", timeout=20, final_schema=dict(count=2, digest="0" * 64))
    assert "release" not in operations and "update_weight_version" not in operations
    assert len(confirmations) == (2 if fault.startswith("flush") else 1)


@pytest.mark.parametrize("local_shutdown", [False, True])
def test_proxy_retirement_fences_workers_and_reaps_each_child(local_shutdown):
    import select
    import signal
    import subprocess
    import sys
    from multiprocessing import Pipe

    from relax.backends.sglang.weight_sync_proxy import _process_identity

    # Drive the production child protocol. Only CUDA discovery and the PG are
    # replaced. Hold graceful teardown after its ACK and group abort before its ACK.
    code = """
from multiprocessing.connection import Connection
from types import SimpleNamespace
import sys
import torch
from relax.backends.sglang import weight_sync_proxy

torch.cuda.set_device = lambda _: None
torch.cuda.get_device_properties = lambda _: SimpleNamespace(uuid="test")
torch.cuda.mem_get_info = lambda _: (1024, 1024)
torch.cuda.memory_allocated = torch.cuda.memory_reserved = lambda _: 0
torch.cuda.max_memory_allocated = torch.cuda.max_memory_reserved = lambda _: 0
arena = weight_sync_proxy._StagingArena
weight_sync_proxy._StagingArena = lambda capacity, device: arena(capacity, "cpu")
weight_sync_proxy._status_buffer = lambda count, tail, device: torch.zeros(count, dtype=torch.uint8)
weight_sync_proxy._serialize = lambda *a: "local test IPC handle"
calls = []
def graceful():
    calls.append("close")
    sys.stdin.buffer.read(1)
def abort():
    calls.append("abort")
    print("entered abort", flush=True)
    sys.stdin.buffer.read(1)
weight_sync_proxy.GpuSyncGroup = lambda *a, **kw: SimpleNamespace(
    close=graceful, abort=abort, receive=lambda batch, slab: (slab, []),
    fence=lambda works, **kw: None,
)
weight_sync_proxy._child(Connection(int(sys.argv[1])))
assert calls == [sys.argv[2]], calls
"""
    proxy = WeightSyncProxy(None)
    children = []
    model = subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE)
    try:
        proxy._workers = [_process_identity(model.pid)]
        for rank in range(2):
            parent, child = Pipe()
            process = subprocess.Popen(
                [sys.executable, "-c", code, str(child.fileno()), "abort" if local_shutdown else "close"],
                pass_fds=(child.fileno(),),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            child.close()
            children.append((process, parent))
            parent.send(
                dict(
                    workers=proxy._workers,
                    device=rank,
                    rank=rank,
                    tp_size=2,
                    pp_size=1,
                    tp_rank=rank,
                    pp_rank=0,
                    timeout_s=5,
                )
            )
        for rank, (_, channel) in enumerate(children):
            assert channel.poll(30) and channel.recv()["success"]
            channel.send(dict(op="join", connection={"transport": "nccl"}, rank=rank))
            assert channel.poll(5) and channel.recv()["success"]
        proxy.children = list(children)
        if local_shutdown:
            with pytest.raises(RuntimeError, match="original model workers"):
                proxy.close(timeout=1, after_worker_exit=True)
            assert all(not channel.closed and process.poll() is None for process, channel in children)
            # One live model has an exported, unfenced IPC read. Its sibling
            # is blocked inside group abort; this child must retain its storage.
            channel = children[0][1]
            for command in (
                dict(op="receive", round=0, batch={"bytes": 1}, budget=16),
                dict(op="ready", round=0, deadline=monotonic() + 10),
                dict(op="export", round=0, control={}, count=1),
            ):
                channel.send(command)
                assert channel.poll(5) and channel.recv()["success"]
            with pytest.raises(RuntimeError, match="Retaining unresolved IPC"):
                proxy.close(timeout=1, abort_group=True)
            assert not channel.closed and not children[1][1].closed
            blocked = children[1][0].stdout
            assert select.select([blocked], [], [], 3)[0] and blocked.readline() == b"entered abort\n"
            model.kill()
            model.wait(timeout=5)
            proxy.close(timeout=5, after_worker_exit=True)
            assert all(process.returncode == -signal.SIGKILL for process, _ in children)
            proxy.close(timeout=0, after_worker_exit=True)
        else:
            with pytest.raises(subprocess.TimeoutExpired):
                proxy.close(timeout=1)
            assert all(channel.closed and process.poll() is None for process, channel in children)
            with pytest.raises(subprocess.TimeoutExpired):
                proxy.close(timeout=0.05)
            for process, _ in children:
                process.stdin.close()
            proxy.close(timeout=5)
            assert all(process.returncode == 0 for process, _ in children)
        assert all(channel.closed for _, channel in children)
        assert not proxy.children
    finally:
        model.kill()
        model.wait(timeout=5)
        model.stdin.close()
        for process, channel in children:
            process.kill()
            process.wait(timeout=5)
            process.stdin.close()
            process.stdout.close()
            channel.close()


def test_stopped_engine_keeps_children_until_every_process_is_reaped(monkeypatch, sglang_engine_module):
    import os
    import signal
    import subprocess
    import sys
    from multiprocessing import Pipe

    from relax.backends.sglang.weight_sync_proxy import _process_identity

    # A real model process provides a birth identity that can be proven dead.
    with subprocess.Popen([sys.executable, "-c", "import sys;sys.stdin.read()"], stdin=subprocess.PIPE) as model:
        owner = _process_identity(model.pid)
        model.stdin.close()
    engine = object.__new__(sglang_engine_module.SGLangEngine)
    proxy = engine._weight_sync_proxy = WeightSyncProxy(engine)
    proxy._workers = [owner]
    children = []
    try:
        for _ in range(2):
            parent, child = Pipe()
            code = (
                "from multiprocessing.connection import Connection; import sys; "
                "Connection(int(sys.argv[1])).send(dict(op='close', success=True)); sys.stdin.read()"
            )
            process = subprocess.Popen(
                [sys.executable, "-c", code, str(child.fileno())], pass_fds=(child.fileno(),), stdin=subprocess.PIPE
            )
            child.close()
            children.append((process, parent))
        first, last = children
        assert all(channel.poll(3) for _, channel in children)
        # One normal close already received its ACK, the other has a late reply.
        assert first[1].recv()["success"]
        first[1].close()
        proxy._closing.update({first[1]: (False, True), last[1]: (False, False)})
        proxy.children = children.copy()
        # An OS error must still let the owner terminate and reap its other child.
        with monkeypatch.context() as patch:

            def denied():
                raise PermissionError("test signal failure")

            patch.setattr(first[0], "kill", denied)
            with pytest.raises(PermissionError, match="test signal failure"):
                proxy.close(timeout=0.1, after_worker_exit=True)
        assert last[0].returncode == -signal.SIGKILL and first[0].poll() is None
        assert proxy.children == children and not last[1].closed
        # Delay termination to exercise a real, bounded Popen.wait timeout.
        with monkeypatch.context() as patch:
            patch.setattr(first[0], "kill", lambda: first[0].send_signal(signal.SIGSTOP))
            with pytest.raises(subprocess.TimeoutExpired):
                proxy.close(timeout=0.05, after_worker_exit=True)
        with pytest.raises(RuntimeError, match="Local GPU proxy retirement"):
            engine.delta_weight_sync("close", timeout=0.05)
        with monkeypatch.context() as patch:
            patch.setattr(proxy, "_workers", [_process_identity(os.getpid())])
            with pytest.raises(RuntimeError, match="original model workers exit"):
                engine.delta_weight_sync("close", timeout=0.05, abort_group=True)
        assert engine._weight_sync_proxy is proxy and first[0].poll() is None
        engine.delta_weight_sync("close", timeout=3, abort_group=True)
        engine.delta_weight_sync("close", timeout=0, abort_group=True)
        assert engine._weight_sync_proxy is None
        assert not proxy.children and not proxy._closing and last[1].closed
        assert first[0].returncode == -signal.SIGKILL
    finally:
        for process, channel in children:
            process.kill()
            process.wait(timeout=3)
            process.stdin.close()
            channel.close()
