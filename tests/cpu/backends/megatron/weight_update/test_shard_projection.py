# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from relax.backends.megatron.weight_update.verl_projection import McoreParamExport, ShardSpec, mcore_hf_delta_entry
from relax.backends.megatron.weight_update.verl_sparse_gather import shard_delta_indices


mappings = pytest.importorskip("megatron.bridge.models.conversion.param_mapping", exc_type=ImportError)


def test_verl_nan_probe_drops_a_real_nan_after_raw_bit_diff():
    """The fixed upstream probe uses the same NaN encoding for a hole and a
    real changed NaN."""
    name = "model.weight"
    before = torch.tensor([0x3F80, 0x7FC1, 0, 0x4000], dtype=torch.int16).view(torch.bfloat16)
    after = torch.tensor([0x7FC3, 0x7FC2, -32768, 0x4000], dtype=torch.int16).view(torch.bfloat16)
    local_idx, local_vals = shard_delta_indices(after, before, 0)
    assert local_idx.tolist() == [0, 1, 2]
    assert local_vals.view(torch.int16).tolist() == [0x7FC3, 0x7FC2, -32768]

    mapping = mappings.DirectMapping("weight", name)
    rec = McoreParamExport("weight", after, ShardSpec(tuple(after.shape)), mapping, None, [(name, tuple(after.shape))])
    slots, dtype, counts, positions, values = mcore_hf_delta_entry(rec, 0, local_idx, local_vals, {})
    assert slots == [(name, (4,))] and dtype == "bfloat16"
    assert counts.tolist() == [1] and positions.tolist() == [2]
    assert values.view(torch.int16).tolist() == [-32768]
