# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest
import torch
from safetensors.torch import load_file

from relax.backends.sglang.disk_weight_dump import dump_model


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        bits = torch.tensor([0x7FC00001, 0, -0x80000000], dtype=torch.int32).view(torch.float32)
        self.weight = torch.nn.Parameter(bits.clone())
        self.alias = self.weight
        self.register_buffer("persistent", torch.tensor([7], dtype=torch.int64))
        self.register_buffer("scratch", torch.tensor([8], dtype=torch.int64), persistent=False)


def test_disk_weight_dump_preserves_bits_and_coverage(tmp_path):
    model = SmallModel()
    before = model.weight.detach().view(torch.int32).clone()
    coverage = dump_model(model, tmp_path / "attempt", 2)
    rank_dir = tmp_path / "attempt" / "rank-2"
    index = json.loads((rank_dir / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index) == {"weight", "alias", "persistent"}
    assert coverage["parameter_aliases"] == [["alias", "weight"]]
    assert coverage["parameter_count"] == 2
    assert coverage["persistent_buffer_count"] == 1
    assert [item["name"] for item in coverage["excluded_nonpersistent_buffers"]] == ["scratch"]
    assert coverage["bytes"] == 2 * model.weight.numel() * 4 + 8
    assert torch.equal(load_file(rank_dir / index["weight"])["weight"].view(torch.int32), before)
    assert torch.equal(model.weight.detach().view(torch.int32), before)
    with pytest.raises(FileExistsError):
        dump_model(model, tmp_path / "attempt", 2)
