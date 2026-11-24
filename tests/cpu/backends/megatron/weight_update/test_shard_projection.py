# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from math import prod
from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("megatron.bridge.models.conversion.param_mapping", exc_type=ImportError)

from megatron.bridge.models.conversion import param_mapping as mappings

from relax.backends.megatron.weight_update.bridge_converter import BridgeConverter
from relax.backends.megatron.weight_update.shard_projection import ShardProjector, UnsupportedShardProjection
from relax.utils.types import ParamInfo


def _bridge(mapping, config=None, *, vocab_size=32):
    converter = BridgeConverter(Namespace(vocab_size=vocab_size), [], None)
    for child in converter.collect_all_mappings(mapping):
        if isinstance(child, mappings.AutoMapping) and child._mapping is None:
            child._mapping = child._get_or_create_mapping("replicated")
    converter._bridge_task_map = {
        mapping.megatron_param: SimpleNamespace(
            mapping=mapping,
            megatron_module=SimpleNamespace(config=config),
            param_weight=torch.empty(0),
        )
    }
    return converter


def _info(mapping, shape, axis, dtype=torch.bfloat16, **attrs):
    return ParamInfo(
        name="module.module." + mapping.megatron_param,
        shape=torch.Size(shape),
        dtype=dtype,
        attrs={"tensor_model_parallel": True, "partition_dim": axis, **attrs},
        size=prod(shape) * dtype.itemsize,
        src_rank=0,
    )


def _round_trip(converter, info, restore, changes, *, splits=None):
    """Patch an actual FULL baseline and compare every output word against FULL
    current."""
    dtype = info.dtype
    word_dtype = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[dtype]
    count = prod(info.shape)
    shards = [torch.arange(1 + i * count, 1 + (i + 1) * count, dtype=dtype).reshape(info.shape) for i in range(2)]
    original = [(key, t.contiguous().clone()) for key, t in converter.convert(info.name, restore(shards))]
    initial = [t.clone() for _, t in original]
    projector = ShardProjector(converter, splits)
    layouts = []
    seen = [set() for _ in original]
    for rank, shard in enumerate(shards):
        idx, words = changes[rank]
        shard.view(word_dtype).view(-1)[idx] = torch.tensor(words, dtype=word_dtype)
        layout = projector.layout(info, rank, len(shards))
        layouts.append(layout)
        entries = layout.project(torch.tensor(idx, dtype=torch.long), shard)
        assert [x[0] for x in entries] == [x[0] for x in original]
        for slot, ((_, baseline), (_, shape, positions, values)) in enumerate(zip(original, entries, strict=True)):
            assert (tuple(baseline.shape), baseline.dtype, torch.int32) == (shape, values.dtype, positions.dtype)
            assert positions.tolist() == sorted(set(positions.tolist()))
            assert not seen[slot].intersection(positions.tolist())
            seen[slot].update(positions.tolist())
            # Copy raw words so signed zeros and the NaN payload never pass through arithmetic here.
            baseline.reshape(-1).view(word_dtype).index_copy_(
                0, positions.long(), values.contiguous().view(word_dtype)
            )
    updated = converter.convert(info.name, restore(shards))
    for slot, ((_, patched), (_, expected)) in enumerate(zip(original, updated, strict=True)):
        assert torch.equal(patched.contiguous().view(word_dtype), expected.contiguous().view(word_dtype))
        expected_positions = (initial[slot].view(word_dtype) != expected.contiguous().view(word_dtype)).view(-1)
        assert sorted(seen[slot]) == expected_positions.nonzero().flatten().tolist()
    return layouts


def test_fused_expert_slots_preserve_native_bits_and_local_fast_slices():
    name = "decoder.layers.0.mlp.experts.linear_fc1.weight7"
    mapping = mappings.FusedGatedExpertMapping(name, "model.layers.0.mlp.experts.gate_up_proj")
    converter = _bridge(mapping)
    info = _info(mapping, (4, 3), 0)

    def restore(shards):
        gate0, up0 = shards[0].chunk(2, dim=0)
        gate1, up1 = shards[1].chunk(2, dim=0)
        return torch.cat((gate0, gate1, up0, up1))

    layouts = _round_trip(
        converter,
        info,
        restore,
        [([0, 2, 9], [0x7FC2, -32768, -61]), ([0, 7, 9], [0x7FC5, -32768, 0])],
    )
    assert layouts[1].local_slices == ((0, 6, 6), (6, 12, 6))
    assert not layouts[1].transforms_values
    assert [slot[0] for slot in layouts[0].slots] == [
        "model.layers.0.mlp.experts.7.gate_proj.weight",
        "model.layers.0.mlp.experts.7.up_proj.weight",
    ]


def test_square_expert_down_uses_actual_full_transpose_and_corrected_etp_axis():
    name = "decoder.layers.0.mlp.experts.linear_fc2.weight7"
    mapping = mappings.AutoMapping(name, "model.layers.0.mlp.experts.down_proj")
    converter = _bridge(mapping)
    converter._bridge_expert_transposes_down = True
    info = _info(mapping, (4, 2), 0)
    layouts = _round_trip(
        converter,
        info,
        lambda shards: torch.cat(shards, 1),
        [([0, 1, 4, 6], [1, -32768, 0x7FC5, 0]), ([0, 3, 6, 7], [0x7FC2, -62, 3, 0])],
    )
    assert layouts[0].local_slices is None


@pytest.mark.parametrize("kind", ["qkv", "vision_qkv", "gdn", "gdn_conv"])
def test_attention_rows_match_full_gated_qkv_and_component_major_gdn(kind):
    config = SimpleNamespace(
        hidden_size=6,
        kv_channels=2,
        num_attention_heads=4,
        num_query_groups=2,
        attention_output_gate=True,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    if kind in ("qkv", "vision_qkv"):
        if kind == "vision_qkv":
            name = "vision_model.decoder.layers.0.self_attention.linear_qkv.weight"
            config.attention_output_gate = False
            mapping = mappings.ConcatenatedQKVMapping(name, "visual.attn.qkv")
            local_rows = 8
        else:
            name = "decoder.layers.0.self_attention.linear_qkv.weight"
            mapping = mappings.QKVMapping(name, "attn.q", "attn.k", "attn.v")
            local_rows = 12
        splits = None
    elif kind == "gdn":
        name = "decoder.layers.0.self_attention.in_proj.weight"
        mapping = mappings.GDNLinearMappingSeparate(name, "linear.qkv", "linear.z", "linear.b", "linear.a")
        splits = (2, 2, 4, 4, 2, 2)
        local_rows = sum(splits)
    else:
        name = "decoder.layers.0.self_attention.conv1d.weight"
        mapping = mappings.GDNConv1dMapping(name, "linear.conv1d")
        splits = (2, 2, 4)
        local_rows = sum(splits)
    converter = _bridge(mapping, config)
    shape = (local_rows, 1, 3) if kind == "gdn_conv" else (local_rows, 6)
    info = _info(mapping, shape, 0, torch.float32)

    def restore(shards):
        if splits is None:
            return torch.cat(shards)
        components = zip(*(s.split(splits) for s in shards), strict=True)
        return torch.cat([torch.cat(pair) for pair in components])

    changes = []
    width = prod(shape[1:])
    for rank in range(2):
        positions = [row * width + (row + rank) % width for row in range(local_rows)]
        words = [0x3F7F0000 + rank * local_rows + row for row in range(local_rows)]
        words[:3] = [0x7FA00005, -(2**31), 0]
        changes.append((positions, words))
    _round_trip(converter, info, restore, changes, splits={name: splits})


def test_vocab_padding_keeps_only_visible_rows_without_allocating_an_embedding():
    name = "embedding.word_embeddings.weight"
    mapping = mappings.ColumnParallelMapping(name, "model.embed_tokens.weight")
    converter = _bridge(mapping, vocab_size=7)
    info = _info(mapping, (4, 2), 0)
    layouts = _round_trip(
        converter,
        info,
        lambda shards: torch.cat(shards),
        [([0, 1, 5], [0x7FC2, -32768, 3]), ([1, 5, 6, 7], [-32768, 0x7FC8, 1, 2])],
    )
    assert layouts[1].local_slices == ((0, 6, 8),)
    converter._args.vocab_size = 250_003
    big = ShardProjector(converter).layout(_info(mapping, (65_536, 2048), 0), 3, 4)
    assert big.local_slices == ((0, (250_003 - 3 * 65_536) * 2048, 3 * 65_536 * 2048),)


def test_strided_boolean_buffer_preserves_noncanonical_true_bytes_across_column_shards():
    mapping = mappings.ReplicatedMapping("decoder.layers.0.buffer", "model.buffer")
    converter = _bridge(mapping)
    source = torch.tensor([[17, 128, 1], [241, 2, 255]], dtype=torch.uint8).T.view(torch.bool)
    layout = ShardProjector(converter).layout(_info(mapping, source.shape, 1, torch.bool), 1, 2)
    [(name, shape, positions, values)] = layout.project(torch.tensor([1, 2, 5]), source)
    assert name == "model.buffer" and shape == (3, 4)
    assert positions.tolist() == [3, 6, 11]
    assert values.dtype == torch.bool
    assert values.view(torch.uint8).tolist() == [241, 128, 255]


def test_replicated_norm_runs_full_value_arithmetic_and_checkpoint_alias():
    name = "decoder.layers.0.self_attention.out_norm.weight"
    alias = "model.layers.0.indexer.weights_proj.weight"
    mapping = mappings.RMSNorm2ZeroCenteredRMSNormMapping(
        name, alias.replace(".weights_proj.", ".scorer.weights_proj.")
    )
    converter = _bridge(mapping)
    converter._hf_alias_checkpoint_keys = {alias}
    info = _info(mapping, (5,), 0, torch.float32, parallel_mode="duplicated")
    layout = ShardProjector(converter).layout(info, 3, 4)
    assert layout.local_slices is None and layout.transforms_values
    shard = torch.tensor([1.25, -1.0, 2.5, 3, 0])
    shard.view(torch.int32)[2] = 0x7FA12345
    positions = torch.tensor([0, 1, 2])
    [(actual_name, shape, indices, values)] = layout.project(positions, shard)
    [(expected_name, expected)] = converter.convert(info.name, shard)
    assert expected_name == actual_name == alias
    assert torch.equal(indices, positions.int()) and shape == (5,)
    assert torch.equal(values.view(torch.int32), expected[positions].view(torch.int32))
    [(empty_name, empty_shape, empty_idx, empty_val)] = layout.project(positions[:0], shard)
    assert (empty_name, empty_shape, empty_idx.numel(), empty_val.dtype) == (alias, shape, 0, torch.float32)


def test_coupled_mapping_and_buffered_group_require_full_before_running_mapping():
    class NeighborSum(mappings.DirectMapping):
        def megatron_to_hf(self, value, module):
            raise AssertionError("The unsupported mapping must never run")

    name = "decoder.layers.0.mlp.weight"
    unknown = NeighborSum(name, "model.mlp.weight")
    with pytest.raises(UnsupportedShardProjection, match="NeighborSum"):
        ShardProjector(_bridge(unknown)).layout(_info(unknown, (6, 2), 0), 0, 2)
    alpha = mappings.DirectMapping("decoder.layers.0.hc.alpha_pre", "hc_scale")
    converter = _bridge(alpha)
    with pytest.raises(UnsupportedShardProjection, match="stateful"):
        ShardProjector(converter).layout(_info(alpha, (1,), 0), 0, 1)
    assert not converter._hc_alpha_buf
