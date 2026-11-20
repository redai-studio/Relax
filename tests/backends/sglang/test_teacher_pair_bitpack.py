# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import struct
from itertools import combinations, cycle, islice

import numpy as np
import pytest
import torch

from examples.on_policy_distillation.mopd.pack_teacher_pair import _hash_tensor, _write_package
from relax.backends.sglang.adts.bitpack import TILE, PackedU16Plan, encode_frame, read_package_header
from relax.backends.sglang.adts_teacher_loader import adts_teacher_loader


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for in-place U16 restoration")
def test_packed_pair_roundtrip_on_all_tile_modes(tmp_path):
    left = np.full(TILE * 4 + 1, 65533, dtype="<u2")
    residual = np.concatenate((np.zeros(TILE), np.ones(TILE), np.full(TILE, 70), np.full(TILE, -12345), [-32768]))
    residual[TILE + 3] = 101
    residual[TILE * 2 + 7] = -1000
    right = (left.astype(np.int32) + residual.astype(np.int32)).astype("<u2")
    split = TILE * 4
    frames = [
        encode_frame(a.view(np.uint8), b.view(np.uint8))
        for a, b in ((left[:split], right[:split]), (left[split:], right[split:]))
    ]
    modes = np.frombuffer(frames[0], dtype="<u4", count=4) & 3
    assert modes.tolist() == [0, 1, 2, 3]
    header = {
        "provenance": {"source": "A"},
        "packets": [
            {"name": "w", "offset": 0, "n": left[:split].nbytes, "size": len(frames[0])},
            {"name": "w", "offset": left[:split].nbytes, "n": left[split:].nbytes, "size": len(frames[1])},
        ],
    }
    payload = b"".join(frames)
    raw = json.dumps(header).encode()
    data = b"ADTS\0\0\0\0" + struct.pack("<Q", len(raw)) + raw + payload
    path = tmp_path / "test.adts"
    path.write_bytes(data)
    gpu = torch.from_numpy(left.copy()).cuda()
    with path.open("rb") as stream:
        header = read_package_header(stream)
        plan = PackedU16Plan(stream, {"w": gpu}, header)
    for target, expected in (("B", right), ("A", left), ("B", right)):
        plan.apply(target)
        np.testing.assert_array_equal(gpu.cpu().numpy(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Teacher switching")
def test_three_teachers_restore_weights_and_logits(tmp_path, monkeypatch):
    rng = torch.Generator().manual_seed(17)
    states = {
        name: {
            "weight": torch.randn(32, 32, generator=rng, dtype=torch.bfloat16),
            "buffer:gemma_weight": torch.randn(32, generator=rng, dtype=torch.bfloat16),
        }
        for name in "ABC"
    }
    packages = {}
    for src, dst in combinations(states, 2):
        path = tmp_path / f"{src}-{dst}.adts"
        header = {
            "provenance": {"source": src, "target": dst},
            "tensors": {
                name: {
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "a_sha256": _hash_tensor(t),
                    "b_sha256": _hash_tensor(states[dst][name]),
                }
                for name, t in states[src].items()
            },
        }
        _write_package(path, [{"params": states[src]}, {"params": states[dst]}], header)
        packages[path.name] = {"bytes": path.stat().st_size}
    (tmp_path / "manifest.json").write_text(json.dumps({"initial_teacher": "A", "package_integrity": packages}))
    monkeypatch.setenv("RELAX_ADTS_TEACHER_DIR", str(tmp_path))

    model = torch.nn.Linear(32, 32, bias=False, device="cuda", dtype=torch.bfloat16)
    model.weight.data.copy_(states["A"]["weight"])
    model.register_buffer("gemma_weight", states["A"]["buffer:gemma_weight"].cuda())
    cache = torch.arange(32, device="cuda", dtype=torch.bfloat16)
    model.register_buffer("cache", cache.clone())
    features = torch.randn(4, 32, generator=rng, dtype=torch.bfloat16).cuda()
    labels = torch.tensor([2, 5, 9, 1], device="cuda")
    loss_mask = torch.tensor([False, True, True, False], device="cuda")
    logits = {
        name: torch.nn.functional.linear(features, tensors["weight"].cuda()) * tensors["buffer:gemma_weight"].cuda()
        + cache
        for name, tensors in states.items()
    }
    ptrs = model.weight.data_ptr(), model.gemma_weight.data_ptr()

    def update(operation, teacher):
        adts_teacher_loader(model, [(f"relax_adts:{operation}:{teacher}", torch.empty(0))])

    def compare(teacher):
        assert torch.equal(model.weight.cpu().view(torch.int16), states[teacher]["weight"].view(torch.int16))
        assert torch.equal(
            model.gemma_weight.cpu().view(torch.int16), states[teacher]["buffer:gemma_weight"].view(torch.int16)
        )
        assert (model.weight.data_ptr(), model.gemma_weight.data_ptr()) == ptrs
        actual = model(features) * model.gemma_weight + model.cache
        torch.testing.assert_close(actual, logits[teacher], rtol=0, atol=0)
        torch.testing.assert_close(
            actual[loss_mask].float().log_softmax(-1).gather(1, labels[loss_mask, None]),
            logits[teacher][loss_mask].float().log_softmax(-1).gather(1, labels[loss_mask, None]),
            rtol=0,
            atol=0,
        )

    update("prepare", "A")
    # Every six switches visit all six directed pairs and return to A.
    for teacher in islice(cycle("BCACBA"), 100):
        update("switch", teacher)
        compare(teacher)

    model.weight.data.zero_()
    model.gemma_weight.zero_()
    model.cache.zero_()
    for teacher in ("B", "A"):
        update("restore", teacher)
        compare(teacher)
