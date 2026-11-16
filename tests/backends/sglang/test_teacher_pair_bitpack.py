# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import hashlib
import json
import struct

import numpy as np
import pytest
import torch

from relax.backends.sglang.adp import package_teacher_ids
from relax.backends.sglang.adp_teacher_loader import _package_header


def test_package_has_one_reversible_file_per_teacher_pair_and_tp_rank(tmp_path):
    pytest.importorskip("triton")
    from examples.on_policy_distillation.mopd.pack_teacher_pair import build_rank_captures

    checkpoints = {name: tmp_path / name for name in ("a", "b", "c")}
    captures = {}
    for i, (teacher, checkpoint) in enumerate(checkpoints.items()):
        checkpoint.mkdir()
        captures[teacher] = []
        for rank in range(2):
            path = tmp_path / f"{teacher}-{rank}.pt"
            weight = torch.tensor([i, rank], dtype=torch.bfloat16)
            torch.save(
                dict(
                    checkpoint_path=str(checkpoint),
                    tp_rank=rank,
                    tp_size=2,
                    params={"w": weight},
                    runtime_config=None,
                    metadata={"layout": {"w": {"shape": [2], "dtype": "torch.bfloat16"}}},
                    auxiliary_hashes={},
                ),
                path,
            )
            captures[teacher].append(path)

    root = tmp_path / "bank"
    build_rank_captures(checkpoints, captures, root)
    routes = {f"route-{name}": str(path) for name, path in checkpoints.items()}
    assert package_teacher_ids(str(root), routes, 2) == {f"route-{name}": name for name in checkpoints}
    manifest = json.loads((root / "manifest.json").read_bytes())
    edges = {(edge["source"], edge["target"]): edge for edge in manifest["edges"]}
    assert len(edges) == len(manifest["package_integrity"]) == 6
    for (source, target), edge in edges.items():
        if edge.get("reverse"):
            assert edge["packages"] == edges[target, source]["packages"]
            continue
        for rank, name in enumerate(edge["packages"]):
            header = _package_header(root / name)
            assert header["provenance"] == dict(
                runtime="sglang-dense-v1", source=source, target=target, tp_rank=rank, tp_size=2
            )
            assert header["packets"][0]["mode"] == "U16_PACKED"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for in-place U16 restoration")
def test_packed_pair_roundtrip_on_all_tile_modes(tmp_path):
    from examples.on_policy_distillation.mopd.pack_teacher_pair import encode_frame
    from relax.backends.sglang.adp.bitpack import TILE, PackedU16Plan

    left = np.full(TILE * 4 + 1, 65533, dtype="<u2")
    residual = np.concatenate((np.zeros(TILE), np.ones(TILE), np.full(TILE, 70), np.full(TILE, -12345), [-32768]))
    residual[TILE + 3] = 101
    residual[TILE * 2 + 7] = -1000
    right = (left.astype(np.int32) + residual.astype(np.int32)).astype("<u2")
    payload = encode_frame(left.view(np.uint8), right.view(np.uint8))
    modes = np.frombuffer(payload, dtype="<u4", count=5) & 3
    assert modes[:4].tolist() == [0, 1, 2, 3]
    header = {
        "provenance": {"source": "A"},
        "packets": [
            {
                "name": "w",
                "offset": 0,
                "n": left.nbytes,
                "size": len(payload),
                "payload_offset": 0,
                "mode": "U16_PACKED",
            }
        ],
    }
    raw = json.dumps(header).encode()
    data = b"ADP1\0\0\0\0" + struct.pack("<Q", len(raw)) + raw + payload
    path = tmp_path / "test.adp"
    path.write_bytes(data)
    gpu = torch.from_numpy(left.copy()).cuda()
    plan = PackedU16Plan(path, {"w": gpu}, header, hashlib.sha256(data).hexdigest())
    for target, expected in (("B", right), ("A", left), ("B", right)):
        plan.apply(target)
        np.testing.assert_array_equal(gpu.cpu().numpy(), expected)
    # A trusted SHA of a malformed size table must still be rejected before running a kernel.
    corrupt = bytearray(data)
    corrupt[-len(payload) : -len(payload) + 4] = struct.pack("<I", 4)
    path.write_bytes(corrupt)
    with pytest.raises(ValueError, match="tile offsets"):
        PackedU16Plan(path, {"w": gpu}, header, hashlib.sha256(corrupt).hexdigest())
