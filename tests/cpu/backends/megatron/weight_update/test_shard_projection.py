# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.weight_update.verl_projection import (
    McoreParamExport,
    ShardSpec,
    _local_hf_views,
    mcore_hf_delta_entry,
)
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


def test_numeric_bridge_projection_omits_changes_rounded_back_to_the_confirmed_hf_word():
    class Layout(mappings.DirectMapping):
        def local_hf_param_specs(self, name):
            return (SimpleNamespace(name=self.hf_param),)

    class Offset(Layout):
        def megatron_to_hf(self, weight, module):
            return {self.hf_param: weight + 1}

    name = "model.weight"
    mapping = Offset("source", name)
    assert _local_hf_views(Layout("source", name), "source") == {name}
    assert not _local_hf_views(mapping, "source")
    before = torch.zeros(3, dtype=torch.bfloat16)
    after = torch.tensor([2**-10, 2**-7, 0], dtype=torch.bfloat16)
    local_idx, local_vals = shard_delta_indices(after, before, 0)
    assert local_idx.tolist() == [0, 1]
    rec = McoreParamExport("source", after, ShardSpec((3,)), mapping, None, [(name, (3,))])
    _, _, counts, positions, values = mcore_hf_delta_entry(
        rec, 0, local_idx, local_vals, {}, previous_shard=before, wire_dtype=torch.bfloat16
    )
    assert counts.tolist() == [1] and positions.tolist() == [1]
    expected = mapping.megatron_to_hf(after, None)[name]
    assert values.view(torch.int16).tolist() == expected[1:2].view(torch.int16).tolist()


def test_native_wire_cast_preserves_only_changed_words_including_signed_zero_and_old_nan():
    name = "model.weight"
    before = torch.tensor([1.001, 1.001, float("nan"), 0.0, 3.0])
    after = torch.tensor([1.002, 1.016, 2.0, -0.0, float("nan")])
    rec = McoreParamExport(
        "source",
        after,
        ShardSpec((5,)),
        mappings.DirectMapping("source", name),
        None,
        [(name, (5,))],
        local_hf_views=frozenset({name}),
    )
    local_idx, local_vals = shard_delta_indices(after, before, 0)
    _, dtype, counts, positions, values = mcore_hf_delta_entry(
        rec, 0, local_idx, local_vals, {}, previous_shard=before, wire_dtype=torch.bfloat16
    )
    assert dtype == "bfloat16" and counts.tolist() == [3]
    assert positions.tolist() == [1, 2, 3]
    assert values.view(torch.int16).tolist() == after[[1, 2, 3]].bfloat16().view(torch.int16).tolist()


def test_only_named_native_views_bypass_comparison_in_a_mixed_output_mapping():
    class Mixed(mappings.DirectMapping):
        def local_hf_param_specs(self, name):
            return (SimpleNamespace(name="view"),)

        def megatron_to_hf(self, weight, module):
            return {"view": weight[:2], "offset": weight[2:] + 1}

    before = torch.zeros(4, dtype=torch.bfloat16)
    after = torch.tensor([0, 2**-10, 2**-10, 2**-7], dtype=torch.bfloat16)
    mapping = Mixed("source", "view")
    rec = McoreParamExport(
        "source",
        after,
        ShardSpec((4,)),
        mapping,
        None,
        [("view", (2,)), ("offset", (2,))],
        local_hf_views=_local_hf_views(mapping, "source"),
    )
    indices, changed = shard_delta_indices(after, before, 0)
    _, _, counts, positions, values = mcore_hf_delta_entry(
        rec, 0, indices, changed, {}, previous_shard=before, wire_dtype=torch.bfloat16
    )
    expected = mapping.megatron_to_hf(after, None)
    assert counts.tolist() == [1, 1] and positions.tolist() == [1, 1]
    assert torch.equal(
        values.view(torch.int16), torch.stack([expected["view"][1], expected["offset"][1]]).view(torch.int16)
    )
