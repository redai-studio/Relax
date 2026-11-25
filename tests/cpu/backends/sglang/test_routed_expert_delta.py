# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Directed receive, the installed SGLang MoE loaders, and the scheduler
ACK."""

import json
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang.delta_loader import load_delta_weights
from relax.backends.sglang.weight_sync_proxy import WeightSyncProxy, _install_status, _status_buffer
from relax.distributed.weight_sync.codec import WireBucket, encode


moe = pytest.importorskip("sglang.srt.layers.moe.fused_moe_triton.layer", exc_type=ImportError)
qwen = pytest.importorskip("sglang.srt.models.qwen3_5", exc_type=ImportError)
expert_location = pytest.importorskip("sglang.srt.eplb.expert_location", exc_type=ImportError)

PREFIX = "model.language_model.layers.0.mlp.experts"
EXPERTS = 4
HIDDEN = 3
LOCAL_WIDTH = 2
ORDINARY = "model.ordinary"
BUDGET = 1 << 18
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def native_model(tp_size, tp_rank, ep_size, ep_rank, *, up_first):
    model = qwen.Qwen3_5MoeForCausalLM.__new__(qwen.Qwen3_5MoeForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_experts=EXPERTS)
    model.num_fused_shared_experts = 0
    model.enable_shared_expert_fusion = False
    model.model = torch.nn.Module()
    model.model.register_parameter("ordinary", torch.nn.Parameter(torch.empty(3), requires_grad=False))
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([layer])
    experts = moe.FusedMoE.__new__(moe.FusedMoE)
    torch.nn.Module.__init__(experts)
    layer.mlp.experts = experts
    experts.layer_id = 0
    experts.moe_tp_size, experts.moe_tp_rank = tp_size, tp_rank
    experts.moe_ep_size, experts.moe_ep_rank = ep_size, ep_rank
    experts._num_global_routed = EXPERTS
    experts._num_local_routed = EXPERTS // ep_size
    experts._expert_storage_rank = ep_rank
    experts.num_fused_shared_experts = 0
    experts.num_local_experts = experts._num_local_routed
    experts.intermediate_size_per_partition = LOCAL_WIDTH
    experts.moe_runner_config = SimpleNamespace(is_gated=True)
    experts.use_padded_loading = experts.use_presharded_weights = False
    experts.use_triton_kernels = experts.use_flashinfer_trtllm_moe = experts._has_fused_shared = False
    experts.quant_config = None
    experts.quant_method = moe.UnquantizedFusedMoEMethod()
    experts.quant_method.use_flashinfer_cutlass = up_first
    experts.quant_method.create_weights(
        experts, experts.num_local_experts, HIDDEN, LOCAL_WIDTH, torch.bfloat16, weight_loader=experts.weight_loader
    )
    return model


def location(monkeypatch, order, ep_size):
    physical = torch.tensor([order])
    inverse = torch.full((1, EXPERTS, EXPERTS), -1, dtype=physical.dtype)
    inverse[:, :, 0] = torch.argsort(physical, dim=1)
    counts = torch.ones_like(physical)
    cls = expert_location.ExpertLocationMetadata
    layout = {"ep_size": ep_size} if "ep_size" in cls.__dataclass_fields__ else {}
    metadata = cls(physical, physical, inverse, inverse, counts, logical_to_rank_dispatch_physical_map=None, **layout)
    if hasattr(expert_location, "get_resources"):
        monkeypatch.setattr(expert_location.get_resources(), "expert_location_metadata", metadata)
    else:
        monkeypatch.setattr(expert_location, "_global_expert_location_metadata", metadata)


def native_weights(tp_size):
    weights = {}
    width = tp_size * LOCAL_WIDTH
    for expert in range(EXPERTS):
        for projection_id, projection in enumerate(PROJECTIONS):
            shape = (HIDDEN, width) if projection == "down_proj" else (width, HIDDEN)
            data = torch.arange(HIDDEN * width, dtype=torch.bfloat16).reshape(shape) + (expert + 1) * 20
            data.add_(projection_id * 5)
            if expert == 0:
                # An unchanged real NaN payload must survive the sparse install.
                data.view(torch.int16).flatten()[-1] = 0x7FC3
            weights[f"{PREFIX}.{expert}.{projection}.weight"] = data
    weights[ORDINARY] = torch.tensor([1.0, 2.0, 3.0])
    return weights


def changes(baseline):
    result = {name: tensor.clone() for name, tensor in baseline.items()}
    for name, tensor in result.items():
        if name == ORDINARY:
            tensor[2] = 9
        else:
            # Include an exact signed zero among changes across all shards.
            word = tensor.view(torch.int16)
            if ".down_proj." in name:
                word[0, :] = -32768
            else:
                word[:, 0] = -32768
    return result


def packed_dense(weights, *, verify=False):
    meta, raw = encode(
        [(name, tuple(value.shape), value.dtype, None, value.view(-1)) for name, value in weights.items()],
        max_bytes=BUDGET,
    )
    if verify:
        meta["verify"] = True
    return meta, raw


def packed_lane(baseline, updated, tp_rank, ep_rank, ep_size, *, write_ordinary=True):
    expert_start = ep_rank * (EXPERTS // ep_size)
    expert_stop = expert_start + EXPERTS // ep_size
    slots = []
    for name, value in updated.items():
        if name == ORDINARY:
            continue
        expert, projection, _ = name.removeprefix(PREFIX + ".").split(".")
        if expert_start <= int(expert) < expert_stop:
            axis = 1 if projection == "down_proj" else 0
            start = tp_rank * LOCAL_WIDTH
            current = value.narrow(axis, start, LOCAL_WIDTH).contiguous()
            before = baseline[name].narrow(axis, start, LOCAL_WIDTH).contiguous()
            positions = (current.view(-1).view(torch.int16) != before.view(-1).view(torch.int16)).nonzero().flatten()
            slots.append((name, tuple(current.shape), current.dtype, positions.int(), current.view(-1)[positions]))
    expert = WireBucket(BUDGET)
    meta, raw = encode(slots, max_bytes=BUDGET)
    meta["expert_direct"] = True
    for entry in meta["tensors"]:
        entry["expert_direct"] = True
    expert.offer(meta, raw)
    if write_ordinary:
        ordinary = updated[ORDINARY]
        meta, raw = encode(
            [(ORDINARY, tuple(ordinary.shape), ordinary.dtype, torch.tensor([2], dtype=torch.int32), ordinary[2:])],
            max_bytes=BUDGET,
        )
        expert.offer(meta, raw)
    return expert.finish()


class CpuProxy:
    """Only worker process transport and HTTP are replaced.

    The real scheduler consumes every lane.
    """

    def __init__(self, models, monkeypatch):
        self.models = models
        self.frames = []
        self.pending = []
        self.published = []
        p = self.proxy = WeightSyncProxy(None)
        p.children = [None] * len(models)
        p.tp_size = len(models)
        p.identities = [
            dict(rank=rank + 1, tp_rank=rank, pp_rank=0, uuid=f"test-scheduler-{rank}") for rank in range(len(models))
        ]
        monkeypatch.setattr(p, "_controls", self.controls)
        monkeypatch.setattr(p, "_request", self.request)
        monkeypatch.setattr(p, "_commands", self.commands)

    def controls(self, controls, *, deadline, flush=False):
        for model, control in zip(self.models, controls, strict=True):
            load_delta_weights(model, [(json.dumps(control), torch.empty(0))])

    def request(self, endpoint, body, *, deadline):
        if endpoint == "update_weight_version":
            self.published.append(body["new_version"])
            assert all(model._relax_delta_receiver.version == body["new_version"] for model in self.models)
        else:
            for model, (control, value, status) in zip(self.models, self.pending, strict=True):
                load_delta_weights(model, [(json.dumps(control), value), ("__delta_capability__", status)])

    def commands(self, commands, *, deadline):
        operation = commands[0]["op"]
        if operation == "receive":
            self.frames = [torch.empty(c["batch"]["bytes"], dtype=torch.uint8) for c in commands]
        elif operation == "export":
            self.pending = [
                (c["control"], tensor, _status_buffer(c["count"], c.get("directory_tail", b""), "cpu"))
                for c, tensor in zip(commands, self.frames, strict=True)
            ]
            self.commands_sent = commands
            return [dict(serialized="fake CPU IPC handle")] * len(self.models)
        elif operation == "status":
            result = []
            for command, (control, _, ack) in zip(self.commands_sent, self.pending, strict=True):
                count = command["count"]
                item = dict(status=ack[:count])
                if control.get("expert_route_capacity"):
                    item["expert_routes"] = ack[count:]
                if control.get("verify_ack_bytes"):
                    item["verified_bytes"] = ack[count:]
                result.append(_install_status(item))
            return result
        elif operation == "release":
            self.pending = []
        return [dict(success=True)] * len(self.models)

    def begin(self, version, **flags):
        self.proxy.begin(
            dict(version=version, staging_budget_bytes=BUDGET, receiver_gpu_budget_bytes=4 * BUDGET, **flags),
            timeout=30,
        )

    def consume(self, data, frame=0):
        if isinstance(data, list):
            self.frames = [raw for _, raw in data]
            return self.proxy.consume(frame, [meta for meta, _ in data], timeout=30)
        meta, raw = data
        self.frames = [raw] * len(self.models)
        return self.proxy.consume(frame, meta, timeout=30)


@pytest.mark.parametrize(("tp_size", "ep_size", "up_first"), [(2, 1, False), (3, 2, True)])
def test_native_expert_lanes_match_full_words_and_report_each_scheduler(monkeypatch, tp_size, ep_size, up_first):
    location(monkeypatch, list(range(EXPERTS)), ep_size)
    models = [
        native_model(tp_size, tp, ep_size, ep, up_first=up_first) for ep in range(ep_size) for tp in range(tp_size)
    ]
    harness = CpuProxy(models, monkeypatch)
    baseline = native_weights(tp_size)
    harness.begin("1", seed=True, expert_direct=True)
    harness.consume(packed_dense(baseline))
    seed = harness.proxy.finalize("1", timeout=30)
    routes = seed["expert_routes"]
    assert len(routes) == len(models)
    for rank, entry in enumerate(routes):
        assert (entry["rank"], entry["tp_rank"], entry["uuid"]) == (rank + 1, rank, f"test-scheduler-{rank}")
        assert entry["fallbacks"] == {}
        [route] = entry["routes"]
        local = EXPERTS // ep_size
        ep_rank, tp_rank = divmod(rank, tp_size)
        assert route == dict(
            prefix=PREFIX,
            expert_start=ep_rank * local,
            expert_stop=(ep_rank + 1) * local,
            intermediate_start=tp_rank * LOCAL_WIDTH,
            intermediate_stop=(tp_rank + 1) * LOCAL_WIDTH,
            intermediate_size=tp_size * LOCAL_WIDTH,
            hidden_size=HIDDEN,
            dtype="bfloat16",
        )

    new = changes(baseline)
    harness.begin("2", seed=False, base="1", verify=True)
    lanes = [packed_lane(baseline, new, tp, ep, ep_size) for ep in range(ep_size) for tp in range(tp_size)]
    harness.consume(lanes)
    # A later change is routed exclusively to rank 0. All other schedulers
    # consume the empty lane and still advance the common frame ordinal.
    only_one = f"{PREFIX}.0.gate_proj.weight"
    new[only_one][0, 1] = 73
    local_value = new[only_one][0, 1:2]
    one_meta, one_raw = encode(
        [(only_one, (LOCAL_WIDTH, HIDDEN), local_value.dtype, torch.tensor([1], dtype=torch.int32), local_value)],
        max_bytes=BUDGET,
    )
    one_meta["expert_direct"] = one_meta["tensors"][0]["expert_direct"] = True
    empty = (dict(format=2, bytes=0, tensors=[], flushes=[], expert_direct=True), torch.empty(0, dtype=torch.uint8))
    harness.consume([(one_meta, one_raw), *([empty] * (len(models) - 1))], frame=1)

    for rank, model in enumerate(models):
        ep, tp = divmod(rank, tp_size)
        target = native_model(tp_size, tp, ep_size, ep, up_first=up_first)
        target.load_weights(list(new.items()))
        for observed, expected in zip(model.parameters(), target.parameters(), strict=True):
            assert torch.equal(observed.view(torch.uint8), expected.view(torch.uint8))
        experts = model.model.layers[0].mlp.experts
        assert not experts.use_presharded_weights
        expert = ep * (EXPERTS // ep_size)
        start = tp * LOCAL_WIDTH
        gate, up, down = [new[f"{PREFIX}.{expert}.{kind}.weight"] for kind in PROJECTIONS]
        rows = (up, gate) if up_first else (gate, up)
        resident = torch.cat([part[start : start + LOCAL_WIDTH] for part in rows])
        assert torch.equal(experts.w13_weight[0].view(torch.int16), resident.view(torch.int16))
        assert torch.equal(
            experts.w2_weight[0].view(torch.int16), down[:, start : start + LOCAL_WIDTH].view(torch.int16)
        )

    assert harness.published == ["1"]
    harness.consume(packed_dense(new, verify=True), frame=2)
    result = harness.proxy.finalize("2", timeout=30)
    assert harness.published == ["1", "2"]
    for rank, receipt in enumerate(result["verified_bytes"]):
        assert (receipt["rank"], receipt["uuid"], receipt["bytes"]) == (
            rank + 1,
            f"test-scheduler-{rank}",
            sum(p.nbytes for p in models[rank].parameters()),
        )


def test_expert_location_remap_withholds_direct_layout_and_keeps_native_install(monkeypatch):
    location(monkeypatch, [1, 0, 2, 3], 2)
    model = native_model(2, 1, 2, 0, up_first=False)
    harness = CpuProxy([model], monkeypatch)
    baseline = native_weights(2)
    harness.begin("1", seed=True, expert_direct=True)
    harness.consume(packed_dense(baseline))
    [entry] = harness.proxy.finalize("1", timeout=30)["expert_routes"]
    assert entry["routes"] == [] and entry["fallbacks"]["remapped_or_redundant_experts"] == 1
    target = native_model(2, 1, 2, 0, up_first=False)
    target.load_weights(list(baseline.items()))
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(model.parameters(), target.parameters(), strict=True)
    )
    harness.begin("2", seed=False, base="1")
    with pytest.raises(ValueError, match="Unqualified receiver-local expert slice"):
        harness.consume([packed_lane(baseline, changes(baseline), 1, 0, 2)])
    assert harness.published == ["1"] and model._relax_delta_receiver.version == "1"


def test_verify_rejects_incomplete_native_full_and_keeps_published_baseline(monkeypatch):
    model = native_model(2, 0, 1, 0, up_first=False)
    harness = CpuProxy([model], monkeypatch)
    baseline = native_weights(2)
    harness.begin("1", seed=True)
    harness.consume(packed_dense(baseline))
    harness.proxy.finalize("1", timeout=30)

    harness.begin("2", seed=False, base="1", verify=True)
    harness.consume(packed_dense({ORDINARY: baseline[ORDINARY]}, verify=True))
    with pytest.raises(ValueError, match="complete acknowledged schema"):
        harness.proxy.finalize("2", timeout=30)
    assert harness.published == ["1"] and model._relax_delta_receiver.version == "1"
    assert model._relax_delta_receiver.failed and model._relax_delta_receiver.verifier is None
