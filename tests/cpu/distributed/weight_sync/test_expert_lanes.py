# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.distributed.weight_sync.codec import encode, frame_flushes, validate_directory
from relax.distributed.weight_sync.expert_lanes import ExpertLanePlan


def _engine(tp, ep):
    workers = []
    for e in range(ep):
        for t in range(tp):
            workers.append(
                dict(
                    routes=[
                        dict(
                            prefix="layer.experts",
                            expert_start=e,
                            expert_stop=e + 1,
                            intermediate_start=6 * t // tp,
                            intermediate_stop=6 * (t + 1) // tp,
                            intermediate_size=6,
                            hidden_size=5,
                            dtype="bfloat16",
                        )
                    ]
                )
            )
    return workers


@pytest.mark.parametrize("tp,ep", [(1, 2), (2, 1), (3, 2)])
def test_sparse_lanes_match_both_axes_of_native_expert_weights(tp, ep):
    engine = _engine(tp, ep)
    plan = ExpertLanePlan([engine, engine])
    rows = []
    expected = {}
    for expert in range(ep):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"layer.experts.{expert}.{projection}.weight"
            shape = (5, 6) if projection == "down_proj" else (6, 5)
            idx = torch.tensor([29, 3, 7, 22, 0], dtype=torch.int32)
            values = torch.tensor([32704, -32768, 10, 8, 1], dtype=torch.int16).view(torch.bfloat16)
            rows.append((name, shape, values.dtype, idx, values))
            target = torch.zeros(shape, dtype=torch.int16)
            target.view(-1)[idx.long()] = values.view(torch.int16)
            expected[name] = target
    values = torch.tensor([10, -4], dtype=torch.float16)
    rows.insert(1, ("ordinary", (3,), values.dtype, torch.tensor([1, 2], dtype=torch.int32), values))
    directory, original = encode(rows, max_bytes=4096)
    metrics = {}
    metas, slabs = plan.partition(directory, original, 4096, metrics)
    assert metrics["direct_expert_delivery_bytes"] == 2 * metrics["direct_expert_input_bytes"]
    assert len(slabs) == 2 * tp * ep
    for rank, (meta, slab) in enumerate(zip(metas, slabs, strict=True)):
        expert, t = divmod(rank % (tp * ep), tp)
        assert slabs[rank % (tp * ep)] is slab
        validate_directory(meta, max_bytes=4096)
        received = {}
        for spec, pos, vals in frame_flushes(meta, slab):
            for slot in spec["params"]:
                idx = pos[slot["pos_start"] : slot["pos_end"]].view(torch.int32)
                words = vals[slot["val_start"] : slot["val_end"]].view(torch.int16)
                if slot["name"] == "ordinary":
                    assert not spec.get("expert_direct")
                    assert torch.equal(words, values.view(torch.int16))
                    continue
                assert spec["expert_direct"] and f".experts.{expert}." in slot["name"]
                local = torch.zeros(slot["shape"], dtype=torch.int16)
                local.view(-1)[idx.long()] = words
                received[slot["name"]] = local
        for name, original_words in expected.items():
            if f".experts.{expert}." in name:
                dim = int("down_proj" in name)
                wanted = original_words.chunk(tp, dim=dim)[t]
                assert torch.equal(received.get(name, torch.zeros_like(wanted)), wanted)


def test_gap_and_missing_layout_retain_original_broadcast_and_report_actual_bytes():
    engines = [_engine(2, 1)]
    engines[0][0]["routes"][0]["intermediate_stop"] -= 1
    for xs in (engines, [[dict(routes=[]), dict(routes=[])]]):
        plan = ExpertLanePlan(xs)
        directory, original = encode(
            [
                (
                    "layer.experts.0.down_proj.weight",
                    (5, 6),
                    torch.float32,
                    torch.tensor([2], dtype=torch.int32),
                    torch.tensor([1.0]),
                )
            ],
            max_bytes=32,
        )
        metrics = {}
        manifest, slab = plan.partition(directory, original, 32, metrics)
        assert manifest is directory and slab is original
        assert sum(metrics["expert_fallback_delivery_bytes"].values()) == 16


def test_packed_lane_keeps_wire_order_and_offsets_after_omitting_an_expert_slot():
    plan = ExpertLanePlan([_engine(2, 1)])
    prefix = "layer.experts.0."
    dtype = torch.bfloat16
    words = torch.tensor([32705, -32768, 12, 34, 56], dtype=torch.int16).view(dtype)
    source, wire = encode(
        [
            (prefix + "gate_proj.weight", (6, 5), dtype, torch.tensor([18], dtype=torch.int32), words[:1]),
            (prefix + "up_proj.weight", (6, 5), dtype, torch.tensor([6, 17, 3], dtype=torch.int32), words[:3]),
            (prefix + "down_proj.weight", (5, 6), dtype, torch.tensor([12, 1], dtype=torch.int32), words[3:]),
        ],
        max_bytes=512,
    )
    directories, slabs = plan.partition(source, wire, 512, {})
    expected, payload = encode(
        [
            (prefix + "up_proj.weight", (3, 5), dtype, torch.tensor([6, 3], dtype=torch.int32), words[[0, 2]]),
            (prefix + "down_proj.weight", (5, 3), dtype, torch.tensor([6, 1], dtype=torch.int32), words[3:]),
        ],
        max_bytes=512,
    )
    expected["expert_direct"] = True
    for entry in expected["tensors"]:
        entry["expert_direct"] = True
    assert directories[0] == expected
    assert torch.equal(slabs[0], payload)


def test_packed_expert_temporaries_are_released_before_assembling_mixed_frame(monkeypatch):
    from torch.multiprocessing.reductions import StorageWeakRef

    from relax.distributed.weight_sync import expert_lanes

    ordinary_encode = expert_lanes.encode_flushes
    ordinary_finish = expert_lanes.WireBucket.finish
    temporaries = []

    def track_inputs(groups, **kwargs):
        groups = tuple(groups)
        for _, positions, values in groups:
            temporaries.extend(StorageWeakRef(t.untyped_storage()) for t in (positions, values))
        return ordinary_encode(groups, **kwargs)

    def check_lifetime(bucket):
        assert temporaries and all(ref.expired() for ref in temporaries)
        return ordinary_finish(bucket)

    monkeypatch.setattr(expert_lanes, "encode_flushes", track_inputs)
    monkeypatch.setattr(expert_lanes.WireBucket, "finish", check_lifetime)
    indices = torch.tensor([0, 3], dtype=torch.int32)
    values = torch.tensor([1, 2], dtype=torch.bfloat16)
    source, wire = encode(
        [
            ("layer.experts.0.up_proj.weight", (6, 5), values.dtype, indices, values),
            ("ordinary", (30,), values.dtype, indices, values),
        ],
        max_bytes=128,
    )
    directories, slabs = ExpertLanePlan([_engine(1, 1)]).partition(source, wire, 128, {})
    assert len(directories[0]["flushes"]) == 2 and slabs[0].numel() == directories[0]["bytes"]
