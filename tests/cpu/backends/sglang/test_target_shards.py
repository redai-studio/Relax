# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang import delta_loader, delta_shard_layout
from relax.distributed.weight_sync.codec import encode


def model_at(rank, tp=4):
    pytest.importorskip("sglang.srt.models.qwen3_moe")
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM

    model = Qwen3MoeForCausalLM.__new__(Qwen3MoeForCausalLM)
    torch.nn.Module.__init__(model)
    model.quant_config = None
    model.config = SimpleNamespace(num_experts=3, num_hidden_layers=1, hidden_size=4, moe_intermediate_size=8)
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    experts = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(experts)
    model.model.layers[0].mlp.experts = experts
    experts.quant_config = None
    experts.quant_method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
    torch.nn.Module.__init__(experts.quant_method)
    experts.quant_method.use_flashinfer_trtllm_moe = experts.use_flashinfer_trtllm_moe = False
    experts.use_triton_kernels = experts.use_presharded_weights = experts.use_padded_loading = False
    experts._has_fused_shared = False
    experts.num_fused_shared_experts = 0
    experts.layer_id = experts.moe_ep_rank = experts._expert_storage_rank = 0
    experts.moe_ep_size, experts.moe_tp_size, experts.moe_tp_rank = 1, tp, rank
    experts._num_local_routed = experts._num_global_routed = experts.num_local_experts = 3
    local = experts.intermediate_size_per_partition = 8 // tp
    experts.w2_weight = torch.nn.Parameter(
        torch.arange(12 * local, dtype=torch.int16).view(torch.bfloat16).reshape(3, 4, local), requires_grad=False
    )
    experts.w2_weight.weight_loader = experts.weight_loader
    model.model.extra = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16), requires_grad=False)
    experts.w13_weight = torch.nn.Parameter(torch.zeros(3, 2 * local, 4, dtype=torch.bfloat16), requires_grad=False)
    experts.w13_weight.weight_loader = experts.weight_loader
    experts.moe_runner_config = SimpleNamespace(is_gated=True)
    experts.quant_method.use_flashinfer_cutlass = False
    return model, experts


def fixture_admission(monkeypatch):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    # Loader inputs, installed tensors, schema and actual ordinary target
    # loads stay real. Substitute the native-version/CUDA admission for CPU.
    monkeypatch.setattr(delta_loader, "sparse_loader_identity", lambda m: (type(m).load_weights,))
    monkeypatch.setattr(
        delta_shard_layout,
        "_runtime",
        lambda m, broadcast=False: (m.model.layers[0].mlp.experts.moe_tp_rank, FusedMoE, UnquantizedFusedMoEMethod),
    )


def source_words():
    result = {}
    for expert in range(3):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            shape = (4, 8) if proj == "down_proj" else (8, 4)
            result[f"model.layers.0.mlp.experts.{expert}.{proj}.weight"] = (
                torch.arange(32, dtype=torch.int16).reshape(shape) + expert * 97
            )
        if expert == 0:
            result["model.extra"] = torch.arange(4, dtype=torch.int16)
    return result


def lane(words, old, rank, *, full):
    entries = []
    marked = []
    for idx, (name, now) in enumerate(words.items()):
        pos = None
        value = now.flatten()
        if not full and name != "model.extra":
            axis = 1 if "down_proj" in name else 0
            local = now.chunk(4, dim=axis)[rank].contiguous().view(-1)
            before = old[name].chunk(4, dim=axis)[rank].contiguous().view(-1)
            pos = (local != before).nonzero().flatten().to(torch.int32)
            value = local.index_select(0, pos.long())
            marked.append(idx)
        entries.append((name, tuple(now.shape), torch.bfloat16, pos, value.view(torch.bfloat16)))
    meta, raw = encode(entries, max_bytes=1 << 20)
    if not full:
        meta["target_shard"] = dict(format=1, tp_rank=rank, tp_size=4, entries=marked)
    return meta, raw


def begin(model, rank, v, full, *, requested=True):
    ctl = dict(
        op="begin",
        version=str(v),
        base=None if full else str(v - 1),
        seed=full,
        receiver_gpu_budget_bytes=1 << 20,
        staging_budget_bytes=1 << 17,
    )
    if requested:
        ctl.update(target_shard_request=True, target_shard_rank=rank)
    delta_loader.load_delta_weights(model, [(json.dumps(ctl), torch.empty(0))])


def apply_frame(model, v, meta, raw):
    status = torch.full((len(meta["tensors"]),), 255, dtype=torch.uint8)
    delta_loader.load_delta_weights(
        model,
        [
            (json.dumps(dict(op="consume", frame=0, version=str(v), metadata=meta)), raw),
            ("__delta_capability__", status),
        ],
    )
    return status


def finish(model, v, *, requested=True):
    status = torch.full((2 if requested else 1,), 255, dtype=torch.uint8)
    delta_loader.load_delta_weights(
        model,
        [
            (json.dumps(dict(op="finalize", version=str(v), frames=1)), torch.empty(0)),
            ("__delta_capability__", status),
        ],
    )
    return status


def seed(model, rank, words):
    begin(model, rank, 1, True)
    meta, raw = lane(words, None, rank, full=True)
    assert not apply_frame(model, 1, meta, raw).any()
    return finish(model, 1)


@pytest.mark.parametrize("rank", range(4))
def test_per_tp_words_match_original_expert_gate_up_down_and_mixed_loader(rank, monkeypatch):
    model, _ = model_at(rank)
    ref, _ = model_at(rank)
    fixture_admission(monkeypatch)
    words = source_words()
    for v in range(1, 6):
        full = v in (1, 4)
        begin(model, rank, v, full)
        old = {k: t.clone() for k, t in words.items()}
        for n, (name, t) in enumerate(words.items()):
            if (n + v) % 3:
                t.view(-1)[(n + v) % t.numel()] = [0x7FC1, -32768, -63, 0, 0x7FC2][v - 1]
        ref.load_weights([(k, t.view(torch.bfloat16)) for k, t in words.items()])
        meta, raw = lane(words, old, rank, full=full)
        before = raw.clone()
        assert not apply_frame(model, v, meta, raw).any()
        assert finish(model, v).tolist() == [0, 1]
        assert torch.equal(raw, before)
        for (n, p), (nr, r) in zip(model.named_parameters(), ref.named_parameters(), strict=True):
            assert n == nr and torch.equal(p.view(torch.uint8), r.view(torch.uint8))
        if not full:
            m = model._relax_delta_receiver.target_shard_metrics
            assert m["entries"] == 9 and m["words"] == sum(
                meta["tensors"][i]["count"] for i in meta["target_shard"]["entries"]
            )


@pytest.mark.parametrize("fault", ["rank", "ordinal", "duplicate", "full", "limit", "mixed_order", "target", "alias"])
def test_bad_partial_fails_before_any_ordinary_or_target_write(fault, monkeypatch):
    m, experts = model_at(0)
    fixture_admission(monkeypatch)
    words = source_words()
    assert seed(m, 0, words).tolist() == [0, 1]
    begin(m, 0, 2, False)
    old = {k: t.clone() for k, t in words.items()}
    for t in words.values():
        t.view(-1)[:2] = -32768
    meta, raw = lane(words, old, 0, full=False)
    marker = meta["target_shard"]
    entry = meta["tensors"][marker["entries"][-1]]
    if fault == "rank":
        marker["tp_rank"] = 1
    if fault == "ordinal":
        marker["entries"] = [999]
    if fault == "duplicate":
        marker["entries"].append(marker["entries"][-1])
    if fault == "full":
        marker["entries"].insert(3, 3)
    if fault in ("limit", "mixed_order"):
        pos = raw[entry["offset"] : entry["offset"] + 4 * entry["count"]].view(torch.int32)
        pos[-1] = 8 if fault == "limit" else pos[0]
    if fault == "target":
        experts.use_padded_loading = True
    if fault == "alias":
        m.register_buffer("new_alias", torch.from_dlpack(experts.w13_weight.detach()))
    before = [p.clone() for p in m.parameters()]
    with pytest.raises((ValueError, RuntimeError)):
        apply_frame(m, 2, meta, raw)
    assert m._relax_delta_receiver.failed and m._relax_delta_receiver.version == "1"
    assert all(
        torch.equal(p.view(torch.uint8), b.view(torch.uint8)) for p, b in zip(m.parameters(), before, strict=True)
    )
    with pytest.raises(RuntimeError):
        apply_frame(m, 2, meta, raw)


@pytest.mark.parametrize("fault", ["partial_schema", "padding", "alias", "parameter_subclass"])
def test_unsupported_seed_declines_before_partial_delivery(fault, monkeypatch):
    m, ex = model_at(2)
    fixture_admission(monkeypatch)
    words = source_words()
    if fault == "partial_schema":
        words.pop(next(iter(words)))
    if fault == "padding":
        ex.use_padded_loading = True
    if fault == "alias":
        m.register_buffer("alias", torch.from_dlpack(ex.w2_weight.detach()))
    if fault == "parameter_subclass":

        class CustomParameter(torch.nn.Parameter):
            pass

        m.model.extra = CustomParameter(m.model.extra.detach(), requires_grad=False)
    # Ordinary FULL input still follows its loader; enable the safe original
    # CPU implementation before triggering the layout-only negative check.
    begin(m, 2, 1, True)
    if fault == "padding":
        ex.use_padded_loading = False
    assert not apply_frame(m, 1, *lane(words, None, 2, full=True)).any()
    if fault == "padding":
        ex.use_padded_loading = True
    assert finish(m, 1).tolist() == [0, 0]


@pytest.mark.parametrize("status_alias", [False, True])
def test_ordinary_full_with_parameter_subclass_acknowledges_only_safe_status(status_alias):
    class PackedParameter(torch.nn.Parameter):
        pass

    class Ordinary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = PackedParameter(torch.zeros(3, dtype=torch.bfloat16), requires_grad=False)

        def load_weights(self, items):
            for name, tensor in items:
                getattr(self, name).copy_(tensor)

    model = Ordinary()
    words = torch.tensor([0x7FC1, -32768, 42], dtype=torch.int16)
    raw_bf16 = words.view(torch.bfloat16)
    begin(model, 0, 1, True)
    metadata, raw = encode([("weight", (3,), torch.bfloat16, None, raw_bf16)], max_bytes=128)
    assert apply_frame(model, 1, metadata, raw).tolist() == [1]
    receiver = model._relax_delta_receiver
    if status_alias:
        status = torch.from_dlpack(model.weight.detach()).view(torch.uint8)[:2]
        with pytest.raises(ValueError, match="overlap"):
            delta_loader.load_delta_weights(
                model,
                [
                    (json.dumps(dict(op="finalize", version="1", frames=1)), torch.empty(0)),
                    ("__delta_capability__", status),
                ],
            )
        assert receiver.failed and receiver.version is None
    else:
        assert finish(model, 1).tolist() == [1, 0]
        assert not receiver.failed and receiver.version == "1"
    assert torch.equal(model.weight.detach().view(torch.int16), words)


def test_final_status_alias_fails_before_version_and_live_write(monkeypatch):
    model, experts = model_at(0)
    fixture_admission(monkeypatch)
    words = source_words()
    begin(model, 0, 1, True)
    apply_frame(model, 1, *lane(words, None, 0, full=True))
    before = experts.w13_weight.detach().view(torch.uint8).clone()
    status = torch.from_dlpack(experts.w13_weight.detach()).view(torch.uint8).reshape(-1)[:2]
    with pytest.raises(ValueError, match="overlap"):
        delta_loader.load_delta_weights(
            model,
            [
                (json.dumps(dict(op="finalize", version="1", frames=1)), torch.empty(0)),
                ("__delta_capability__", status),
            ],
        )
    assert model._relax_delta_receiver.version is None and model._relax_delta_receiver.failed
    assert torch.equal(before, experts.w13_weight.detach().view(torch.uint8))


def test_native_serializer_selective_proxy_f0_then_distinct_tp_payloads(monkeypatch):
    from sglang.srt.utils import MultiprocessingSerializer

    from relax.backends.sglang import weight_sync_proxy as wire

    models = [model_at(tp)[0] for tp in range(4)]
    refs = [model_at(tp)[0] for tp in range(4)]
    fixture_admission(monkeypatch)
    proxy = wire.WeightSyncProxy(None)
    proxy.children = [None] * 4
    proxy.identities = [dict(uuid=f"cpu-{r}") for r in range(4)]
    payloads = []
    statuses = []
    events = []

    def request(endpoint, packet, *, deadline):
        if endpoint == "update_weight_version":
            assert all(m._relax_delta_receiver.version == packet["new_version"] for m in models)
            events.append("version")
            return
        assert wire._http_bytes(packet) <= wire.MAX_HTTP_BYTES
        for rank, (model, encoded) in enumerate(zip(models, packet["serialized_named_tensors"], strict=True)):
            items = MultiprocessingSerializer.deserialize(encoded)
            ctl = json.loads(items[0][0])
            if ctl["op"] == "consume":
                assert torch.equal(items[0][1], payloads[rank])
                if ctl["version"] == "2":
                    assert "metadata_zlib" in ctl
            delta_loader.load_delta_weights(model, items)
        events.append(ctl["op"])

    def commands(cmds, *, deadline):
        op = cmds[0]["op"]
        events.append(op)
        if op == "export":
            statuses[:] = [torch.full((c["count"],), 255, dtype=torch.uint8) for c in cmds]
            return [
                dict(serialized=wire._serialize(c["control"], raw, status))
                for c, raw, status in zip(cmds, payloads, statuses, strict=True)
            ]
        if op == "status":
            return [dict(full_required=s.tolist()) for s in statuses]
        return [dict(success=True)] * 4

    monkeypatch.setattr(proxy, "_commands", commands)
    monkeypatch.setattr(proxy, "_request", request)
    words = source_words()
    for v in (1, 2):
        full = v == 1
        proxy.begin(
            dict(
                version=str(v),
                base=None if full else "1",
                seed=full,
                staging_budget_bytes=1 << 17,
                receiver_gpu_budget_bytes=1 << 20,
                target_shard_request=True,
            ),
            timeout=30,
        )
        old = {k: t.clone() for k, t in words.items()}
        if not full:
            for n, t in enumerate(words.values()):
                t.view(-1)[: (n % 4) + 1] = -32768
        deliveries = [lane(words, old, r, full=full) for r in range(4)]
        payloads[:] = [raw for _, raw in deliveries]
        assert not proxy.consume(0, deliveries[0][0] if full else [md for md, _ in deliveries], timeout=30)[
            "full_required"
        ]
        assert events[-4:] == ["export", "consume", "status", "release"]
        result = proxy.finalize(str(v), timeout=30)
        assert result["target_shard_capable"] and not result["force_full_seed"]
        if not full:
            assert [r["raw_bytes"] for r in result["metrics"]["target_shards"]] == [p.numel() for p in payloads]
            assert len({p.numel() for p in payloads}) > 1
        for m, ref in zip(models, refs, strict=True):
            ref.load_weights([(k, t.view(torch.bfloat16)) for k, t in words.items()])
            assert all(
                torch.equal(p.view(torch.uint8), q.view(torch.uint8))
                for p, q in zip(m.parameters(), ref.parameters(), strict=True)
            )


def test_batched_triton_tiles_preserve_words_and_detect_cross_tile_duplicate():
    from triton.runtime.interpreter import InterpretedFunction

    from relax.backends.sglang.delta_shard_kernel import shard_tiles

    count = 791
    coords = torch.arange(count, dtype=torch.int32)
    words = torch.arange(count, dtype=torch.int16)
    words[1:4] = torch.tensor([0x7FC1, -32768, -63], dtype=torch.int16)
    meta, raw = encode(
        [("first", (count + 10,), torch.bfloat16, coords, words.view(torch.bfloat16))], max_bytes=1 << 20
    )
    entry = meta["tensors"][0]
    width = 512
    rows = torch.tensor(
        [
            [entry["offset"] + 4 * j, entry["value_offset"] + 2 * j, min(width, count - j), int(j > 0), 7, count]
            for j in range(0, count, width)
        ],
        dtype=torch.int64,
    )
    target = torch.full((count + 20,), 13, dtype=torch.int16)
    bad = torch.zeros(1, dtype=torch.int32)
    kernel = InterpretedFunction(shard_tiles.fn)
    kernel[(2,)](raw, rows, target, bad, False, width)
    assert not bad.item()
    kernel[(2,)](raw, rows, target, bad, True, width)
    assert torch.equal(target[7 : 7 + count], words) and (target[:7] == 13).all() and (target[7 + count :] == 13).all()
    raw[: 4 * count].view(torch.int32)[width] = width - 1
    kernel[(2,)](raw, rows, target, bad, False, width)
    assert bad.item() == 1


@pytest.mark.parametrize("fault", ["none", "fragmented", "capacity", "ack", "bad_last", "legacy_tp"])
def test_ordered_target_consume_fences_each_frame_and_byte(fault, monkeypatch):
    from sglang.srt.utils import MultiprocessingSerializer

    from relax.backends.sglang import weight_sync_proxy as wire

    models = [model_at(tp)[0] for tp in range(4)]
    refs = [model_at(tp)[0] for tp in range(4)]
    fixture_admission(monkeypatch)
    proxy = wire.WeightSyncProxy(None)
    proxy.children = [None] * 4
    proxy.identities = [dict(uuid=str(rank)) for rank in range(4)]
    arenas = [wire._StagingArena(8192, "cpu") for _ in models]
    sizes, packed, receipts = {}, [], []
    http, releases = [], []
    version = "1"

    def install(endpoint, packet, *, deadline):
        assert wire._http_bytes(packet) <= wire.MAX_HTTP_BYTES
        if endpoint == "update_weight_version":
            assert all(model._relax_delta_receiver.version == version for model in models)
            return
        for rank, (model, handle) in enumerate(zip(models, packet["serialized_named_tensors"], strict=True)):
            items = MultiprocessingSerializer.deserialize(handle)
            control = json.loads(items[0][0])
            if fault == "legacy_tp" and rank == 3 and control["op"] == "begin":
                control.pop("ordered_target_consume")
                items[0] = (json.dumps(control), items[0][1])
            delta_loader.load_delta_weights(model, items)
        http.append(control["op"])

    def commands(cmds, *, deadline):
        op = cmds[0]["op"]
        if op == "locations":
            return [
                dict(
                    free_bytes=1 << 20,
                    spans=[(a.active[k][0], sizes[k][r]) for k in range(cmds[r]["round"], cmds[r]["stop"])],
                )
                for r, a in enumerate(arenas)
            ]
        if op == "export":
            receipts.clear()
            replies = []
            for rank, (c, arena) in enumerate(zip(cmds, arenas, strict=True)):
                status = torch.full((c["count"],), 255, dtype=torch.uint8)
                receipts.append(status)
                offset, length = c.get("span", (arena.active[c["round"]][0], sizes[c["round"]][rank]))
                raw = arena.storage.narrow(0, offset, length)
                replies.append(dict(serialized=wire._serialize(c["control"], raw, status)))
            return replies
        if op == "status":
            if fault == "ack" and version == "2":
                receipts[-1][-1] = 255
            return [dict(full_required=s.tolist()) for s in receipts]
        if op == "release":
            for key in range(cmds[0]["round"], cmds[0].get("stop", cmds[0]["round"] + 1)):
                for arena in arenas:
                    arena.release(key)
                releases.append(key)
            return [{}] * 4
        pytest.fail(op)

    def stage(key, deliveries):
        sizes[key] = [raw.numel() for _, raw in deliveries]
        for arena, (_, raw) in zip(arenas, deliveries, strict=True):
            arena.reserve(key, raw.numel()).copy_(raw)
        return [meta for meta, _ in deliveries]

    def finish_receive(key, *args, **kwargs):
        stage(key, [(None, torch.empty(0, dtype=torch.uint8))] * 4)

    monkeypatch.setattr(proxy, "_request", install)
    monkeypatch.setattr(proxy, "_commands", commands)
    monkeypatch.setattr(proxy, "receive", finish_receive)
    monkeypatch.setattr(proxy, "ready", lambda *a, **kw: None)
    words = source_words()
    words.pop("model.extra")
    config = dict(receiver_gpu_budget_bytes=1 << 20, staging_budget_bytes=8192, target_shard_request=True)
    assert "ordered_target_consume" not in proxy.begin(dict(config, version="1", seed=True), timeout=30)
    seed_lanes = stage(0, [lane(words, None, rank, full=True) for rank in range(4)])
    proxy.consume(0, seed_lanes[0], timeout=30)
    assert proxy.finalize("1", timeout=30)["target_shard_capable"]
    version = "2"
    ready = proxy.begin(dict(config, version=version, base="1", seed=False), timeout=30)
    if fault == "legacy_tp":
        assert "ordered_target_consume" not in ready
        return
    assert ready["ordered_target_consume"] is True
    old = {k: t.clone() for k, t in words.items()}
    groups = [dict(list(words.items())[n : n + 3]) for n in (0, 3, 6)]
    # Middle frame is all empty. TP ranks differ in changed counts as well.
    for group in (groups[0], groups[2]):
        for index, words2d in enumerate(group.values()):
            words2d.flatten()[: index + 5] = torch.tensor([0x7FC1, -32768, -63, 0x7FC2, 3, 4, 5][: index + 5])
    for key, group in enumerate(groups):
        if fault == "fragmented" and key == 1:
            arenas[0].reserve(777, 8)
        packed.append(stage(key, [lane(group, old, rank, full=False) for rank in range(4)]))
    before = [arena.storage.clone() for arena in arenas]
    http.clear()
    releases.clear()
    if fault == "bad_last":
        initial = [p.clone() for p in models[0].parameters()]
        offset = arenas[0].active[2][0]
        arenas[0].storage[offset : offset + 4].view(torch.int32).fill_(2**30)
        with pytest.raises(ValueError, match="coordinates"):
            proxy.consume(0, packed[0], following=packed[1:], timeout=30)
        assert proxy._target_failed and releases == []
        assert all(
            torch.equal(p.view(torch.uint8), original.view(torch.uint8))
            for p, original in zip(models[0].parameters(), initial, strict=True)
        )
        return
    if fault == "capacity":
        proxy._workspace_budget = 1
    if fault == "ack":
        with pytest.raises(RuntimeError, match="acknowledgement"):
            proxy.consume(0, packed[0], following=packed[1:], timeout=30)
        assert releases == [] and http == ["consume"] and proxy._target_failed
        with pytest.raises(RuntimeError, match="ambiguous"):
            proxy.finalize("2", timeout=30)
        return
    result = proxy.consume(0, packed[0], following=packed[1:], timeout=30)
    assert result == dict(success=True, full_required=[], ordered_frames_ack=3)
    assert http == ["consume"] * {"none": 1, "fragmented": 2, "capacity": 3}[fault]
    assert releases == [0, 1, 2] and proxy.batch == 3
    assert all(torch.equal(a.storage, initial) for a, initial in zip(arenas, before, strict=True))
    assert all(s["status_acks"] == s["releases"] == 3 for s in proxy.metrics["target_shards"])
    assert proxy.finalize("2", timeout=30)["target_shard_capable"]
    for model, ref in zip(models, refs, strict=True):
        ref.load_weights([(k, t.view(torch.bfloat16)) for k, t in words.items()])
        assert all(
            torch.equal(p.view(torch.uint8), q.view(torch.uint8))
            for p, q in zip(model.parameters(), ref.parameters(), strict=True)
        )


def conditional_at(rank, tp):
    cls = pytest.importorskip("sglang.srt.models.qwen3_5").Qwen3_5MoeForConditionalGeneration

    template, experts = model_at(rank, tp)
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.model, model.config = template.model, template.config
    model.model.start_layer, model.model.end_layer = 0, 1
    model.config.encoder_only = model.config.tie_word_embeddings = False
    model.enable_shared_expert_fusion = False
    return model, experts


@pytest.mark.parametrize("fused", [False, True])
def test_conditional_moe_ordinary_sparse_loader_keeps_raw_tp_words(monkeypatch, fused):
    model, actual = conditional_at(1, 4)
    reference, expected = conditional_at(1, 4)
    fixture_admission(monkeypatch)
    prefix = "model.language_model.layers.0.mlp.experts."
    if fused:
        shapes = {prefix + "gate_up_proj": (3, 16, 4), prefix + "down_proj": (3, 4, 8)}
    else:
        shapes = {
            prefix + "1.gate_proj.weight": (8, 4),
            prefix + "1.up_proj.weight": (8, 4),
            prefix + "1.down_proj.weight": (4, 8),
        }

    words = {name: torch.arange(torch.Size(shape).numel(), dtype=torch.int16) for name, shape in shapes.items()}
    for seed in (True, False):
        version = 1 if seed else 2
        begin(model, 1, version, seed, requested=False)
        entries = []
        dense = []
        for name, shape in shapes.items():
            flat = words[name]
            indices = torch.arange(0, flat.numel(), 2, dtype=torch.int32)
            if not seed:
                flat[indices.long()] = -32768
                flat[indices[::2].long()] = 32705
            values = flat.view(torch.bfloat16)
            entries.append(
                (name, shape, values.dtype, None if seed else indices, values if seed else values[indices.long()])
            )
            dense.append((name, values.view(shape)))
        metadata, raw = encode(entries, max_bytes=1 << 20)
        assert not apply_frame(model, version, metadata, raw).any()
        finish(model, version, requested=False)
        reference.load_weights(dense)
        for weight, oracle in ((actual.w13_weight, expected.w13_weight), (actual.w2_weight, expected.w2_weight)):
            assert torch.equal(weight.view(torch.int16), oracle.view(torch.int16))
        if not seed:
            assert (actual.w13_weight.view(torch.int16) < 0).any()
            assert (actual.w2_weight.view(torch.int16) < 0).any()


@pytest.mark.parametrize(("rank", "fault"), [(0, None), (1, None), (0, "alias"), (1, "bad_coordinate")])
def test_broadcast_global_expert_positions_match_conditional_loader(rank, fault, monkeypatch):
    model, actual = conditional_at(rank, 2)
    reference, _ = conditional_at(rank, 2)
    fixture_admission(monkeypatch)
    # Dense aliases and FP32 occur outside the direct BF16 expert destinations.
    for m in (model, reference):
        m.model.extra_alias = m.model.extra
        m.model.single_precision = torch.nn.Parameter(torch.ones(3), requires_grad=False)
    if fault == "alias":
        model.register_buffer("aliased_expert_words", actual.w13_weight.detach())
    words = {
        name.replace("model.layers.", "model.language_model.layers.") if rank else name: value
        for name, value in source_words().items()
    }
    for version in (1, 2):
        full = version == 1
        begin(model, rank, version, full, requested=False)
        entries = []
        dense = []
        for i, (name, matrix) in enumerate(words.items()):
            pos = None
            flat = matrix.flatten()
            if not full:
                # Include both lane boundaries, the last global word, an empty
                # sparse projection, and a count greater than the local size.
                pos = torch.arange(0 if i == 0 else flat.numel(), dtype=torch.int32)
                flat[pos.long()] = torch.tensor([32705, -32768, -63, 32706], dtype=torch.int16).repeat(8)[
                    : pos.numel()
                ]
                if name == "model.extra":
                    pos = None
            value = flat if pos is None else flat[pos.long()]
            if fault == "bad_coordinate" and not full and i == len(words) - 1:
                pos[-1] += 1
            entries.append((name, tuple(matrix.shape), torch.bfloat16, pos, value.view(torch.bfloat16)))
            dense.append((name, matrix.view(torch.bfloat16)))
        metadata, raw = encode(entries, max_bytes=1 << 20)
        if fault == "bad_coordinate" and not full:
            before = [p.detach().clone() for p in model.parameters()]
            with pytest.raises(ValueError, match="in bounds"):
                apply_frame(model, version, metadata, raw)
            assert all(
                torch.equal(b.view(torch.uint8), p.view(torch.uint8)) for b, p in zip(before, model.parameters())
            )
            return
        prev_lazy = getattr(model, "_routed_experts_weights_of_layer", None)
        apply_frame(model, version, metadata, raw)
        finish(model, version, requested=False)
        reference.load_weights(dense)
        assert model._routed_experts_weights_of_layer is not prev_lazy
        for p, oracle in zip(model.parameters(), reference.parameters(), strict=True):
            assert torch.equal(p.view(torch.uint8), oracle.view(torch.uint8))
        receiver = model._relax_delta_receiver
        assert (receiver._broadcast_plan is None) == (fault == "alias")
        if not full:
            m = receiver.target_shard_metrics
            assert m["entries"] == (0 if fault == "alias" else 9)
            assert m["ordinary_entries"] == (10 if fault == "alias" else 1)
            assert m["ordinary_full_entries"] == 1


@pytest.mark.parametrize("down", [True, False])
def test_triton_maps_global_broadcast_words_to_each_tp_rank(down):
    from triton.runtime.interpreter import InterpretedFunction

    from relax.backends.sglang.delta_shard_kernel import shard_tiles

    hidden, local, width = 53, 11, 512
    elements = hidden * 2 * local
    coords = torch.arange(elements, dtype=torch.int32)
    words = torch.arange(elements, dtype=torch.int16) - 32768
    words[::13] = 32705
    shape = (hidden, 2 * local) if down else (2 * local, hidden)
    meta, raw = encode([("expert", shape, torch.bfloat16, coords, words.view(torch.bfloat16))], max_bytes=1 << 20)
    entry = meta["tensors"][0]
    rows = torch.tensor(
        [
            (4 * j, entry["value_offset"] + 2 * j, min(width, elements - j), int(j > 0), 3, elements)
            for j in range(0, elements, width)
        ]
    )
    kernel = InterpretedFunction(shard_tiles.fn)
    for rank in range(2):
        target = torch.zeros(hidden * local + 6, dtype=torch.int16)
        bad = torch.zeros(1, dtype=torch.int32)
        kernel[(len(rows),)](raw, rows, target, bad, False, width)
        assert not bad.item()
        kernel[(len(rows),)](raw, rows, target, bad, True, width, True, down, hidden, local, 2, rank)
        expected = words.view(shape).chunk(2, dim=int(down))[rank].reshape(-1)
        assert torch.equal(target[3:-3], expected) and not target[:3].any() and not target[-3:].any()
