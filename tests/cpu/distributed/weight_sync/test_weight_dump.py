# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Audit packing must preserve every raw bit and tied parameter alias."""

import torch

from relax.backends.sglang.disk_weight_dump import dump_model
from relax.distributed.weight_sync.tensor_diff import compare


def test_packed_dump_matches_streamed_dump_without_changing_model(tmp_path):
    model = torch.nn.Module()
    # Distinct NaNs and signed zeros must survive without float comparisons.
    bits = torch.tensor([0x7FC1, 0x7FC2, 0, -0x8000, 1], dtype=torch.int16)
    model.weight = torch.nn.Parameter(bits.view(torch.bfloat16), requires_grad=False)
    model.tied = model.weight
    model.register_buffer("counter", torch.tensor([3, 7], dtype=torch.int32))
    model.register_buffer("cache", torch.ones(2), persistent=False)
    before = {name: (p.data_ptr(), p.view(torch.uint8).numpy().tobytes()) for name, p in model.state_dict().items()}
    streamed = dump_model(model, tmp_path / "streamed", 0)
    packed = dump_model(model, tmp_path / "packed", 0, packed=True)
    assert packed == streamed
    assert packed["parameter_aliases"] == [["tied", "weight"]]
    assert packed["persistent_buffer_count"] == 1
    assert [x["name"] for x in packed["excluded_nonpersistent_buffers"]] == ["cache"]
    assert len(list((tmp_path / "packed/rank-0").glob("*.safetensors"))) == 1
    result = compare(tmp_path / "streamed/rank-0", tmp_path / "packed/rank-0")
    assert result["equal"] and result["mismatch_bytes"] == result["mismatch_bits"] == 0
    assert before == {
        name: (p.data_ptr(), p.view(torch.uint8).numpy().tobytes()) for name, p in model.state_dict().items()
    }
