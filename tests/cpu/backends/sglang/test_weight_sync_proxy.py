# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections import deque
from time import monotonic
from types import SimpleNamespace

import pytest

from relax.backends.sglang.weight_sync_proxy import WeightSyncProxy


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

    from relax.distributed.weight_sync.codec import validate_directory

    rng = random.Random(709)
    names = [f"model.layers.0.mlp.experts.{index}.down_proj.weight" for index in range(entries)]
    if entropy:
        names = [name + base64.b64encode(rng.randbytes(entropy)).decode("ascii") for name in names]
    directory = dict(
        format=1,
        bytes=0,
        tensors=[
            dict(name=n, shape=[2048, 768], dtype="bfloat16", count=0, full=False, offset=0, value_offset=0)
            for n in names
        ],
    )
    validate_directory(directory, max_bytes=16 << 20)
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

    monkeypatch.setattr(delta_loader, "sparse_loader_identity", lambda m: (Matrix.load_weights,))
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
            indices = None if seed else torch.tensor([0, 1], dtype=torch.int32)
            if not seed:
                words[:2] = torch.tensor([0x7FC2, -32768], dtype=torch.int16)
            vals = (words if seed else words[:2]).view(torch.bfloat16)
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
            assert not proxy.finalize(version, timeout=30)["force_full_seed"]
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


@pytest.mark.parametrize("fault", ["large", "malformed", "lost_http", "missing_status"])
def test_target_lane_preexport_fallback_and_ambiguous_terminal(monkeypatch, native_serializer, fault):
    proxy = WeightSyncProxy(None)
    proxy.children = [None] * 4
    proxy.version, proxy.batch, proxy.staging_budget = "2", 0, 16 << 20
    proxy._target_requested = proxy._target_capable = True
    proxy.metrics = dict(consume_calls=0, update_directory_calls=0)
    count = 250
    lanes = [
        dict(
            _metadata(count, entropy=160 if fault == "large" else 0),
            target_shard=dict(format=1, tp_rank=r, tp_size=4, entries=list(range(count))),
        )
        for r in range(4)
    ]
    events = []
    if fault == "malformed":
        lanes[3]["target_shard"]["tp_rank"] = 0

    def docs(controls, *, deadline):
        assert all(c["op"] == "document" for c in controls)
        assert len({c["id"] for c in controls}) == 4
        events.append("fragment")

    def commands(cmds, *, deadline):
        op = cmds[0]["op"]
        events.append(op)
        if op == "export":
            assert all(("metadata_zlib" not in c["control"]) is (fault == "large") for c in cmds)
            return [dict(serialized="handle")] * 4
        if op == "status":
            return [dict(full_required=[255 if fault == "missing_status" else 0] * count)] * 4
        return [dict(success=True)] * 4

    def request(*a, **kw):
        events.append("payload")
        if fault == "lost_http":
            raise TimeoutError("ambiguous")

    monkeypatch.setattr(proxy, "_commands", commands)
    monkeypatch.setattr(proxy, "_controls", docs)
    monkeypatch.setattr(proxy, "_request", request)
    if fault == "large":
        assert proxy.consume(0, lanes, timeout=30)["success"]
        assert events[-4:] == ["export", "payload", "status", "release"] and set(events[:-4]) == {"fragment"}
    else:
        with pytest.raises((ValueError, RuntimeError, TimeoutError)):
            proxy.consume(0, lanes, timeout=30)
        assert (
            events
            == {
                "malformed": [],
                "lost_http": ["export", "payload"],
                "missing_status": ["export", "payload", "status"],
            }[fault]
        )
        with pytest.raises(RuntimeError, match="ambiguous"):
            proxy.consume(0, lanes, timeout=30)
        assert "release" not in events


def test_missing_final_lane_capability_keeps_human_version_unpublished(monkeypatch):
    proxy = WeightSyncProxy(None)
    proxy.children = [None] * 4
    proxy.version, proxy.batch, proxy.staging_budget = "0", 1, 512
    proxy._target_requested = True
    events = []

    def commands(cmds, *, deadline):
        op = cmds[0]["op"]
        events.append(op)
        if op == "export":
            assert all(c["count"] == 2 for c in cmds)
            return [dict(serialized="handle")] * 4
        if op == "status":
            return [dict(full_required=[0, 1])] * 3 + [dict(full_required=[0, 255])]
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
        proxy.consume(0, {"tensors": [{"name": "w"}]}, timeout=10)
    with pytest.raises(RuntimeError, match="final acknowledgement"):
        proxy.finalize("2", timeout=10)
    assert operations == ["export", "update_weights_from_tensor", "status"] * 2
