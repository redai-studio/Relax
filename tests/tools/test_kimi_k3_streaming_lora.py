# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_SPEC = importlib.util.spec_from_file_location(
    "kimi_k3_streaming_lora",
    Path(__file__).resolve().parents[2] / "examples/models/kimi-k3/tools/kimi_k3_streaming_lora.py",
)
export = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export)


def test_grouped_mapping_keeps_checkpoint_keys_and_splits_canonical_fc1() -> None:
    prefix = "language_model.decoder.layers.1.mlp.experts.experts.linear_fc1"
    a_key, b_key = (prefix + f".adapter.linear_{side}.weight" for side in ("in", "out"))
    tensors = {a_key: {"shape": [2, 2, 3]}, b_key: {"shape": [2, 8, 2]}}
    seen = []

    def lookup(target):
        seen.append(target)
        return SimpleNamespace(hf_param={"gate": "model.experts.0.w1.weight", "up": "model.experts.0.w3.weight"})

    weight_map = {f"model.experts.{idx}.w{proj}.weight_packed": "x" for idx in (0, 1) for proj in (1, 3)}
    weight_map.update({key.removesuffix("_packed") + "_scale": "x" for key in list(weight_map)})
    actual = export.build_mappings(tensors, SimpleNamespace(megatron_to_hf_lookup=lookup), weight_map, 2)
    assert seen == ["decoder.layers.1.mlp.experts.linear_fc1.weight0"]
    assert len(actual) == 4
    assert actual["model.experts.1.w3.weight"] == {
        "a_key": a_key,
        "b_key": b_key,
        "expert_index": 1,
        "b_slice": [4, 8],
    }


def test_merge_selects_expert_up_projection_and_accumulates_fp32() -> None:
    a = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3).to(torch.bfloat16)
    b = torch.arange(32, dtype=torch.float32).reshape(2, 8, 2).to(torch.bfloat16)
    base = torch.ones(4, 3, dtype=torch.bfloat16)
    entry = {"expert_index": 1, "b_slice": [4, 8]}
    actual = export.merge_weight(base, a, b, entry, 4, 2)
    expected = (base.float() + (b[1, 4:8].float() @ a[1].float()) * 2).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(base, torch.ones_like(base))


def test_dense_fc1_dcp_is_canonical_not_tp_interleaved() -> None:
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from megatron.core.transformer.mlp import apply_swiglu_sharded_factory

    # Model live TP shards: [gate_rank, up_rank]. DCP offsets assemble all gates
    # first, then all ups, so full-DCP readers must not perform TP reordering.
    assembled = torch.empty(16, 2)
    for rank in range(4):
        live = torch.cat((torch.full((2, 2), float(rank)), torch.full((2, 2), float(rank + 10))))
        sharded = ShardedTensor.from_rank_offsets("fc1", live, (0, rank, 4))
        factory = apply_swiglu_sharded_factory(sharded, ())
        for part in factory.build():
            start = part.global_offset[0]
            assembled[start : start + 2] = part.data
    assert assembled[:8, 0].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert assembled[8:, 0].tolist() == [10, 10, 11, 11, 12, 12, 13, 13]


def test_mapping_rejects_incomplete_or_wrong_rank_pairs() -> None:
    prefix = "language_model.decoder.x.adapter.linear_"
    with pytest.raises(ValueError, match="Incomplete"):
        export.build_mappings({prefix + "in.weight": {"shape": [2, 3]}}, None, {}, 2)
    with pytest.raises(ValueError, match="rank differs"):
        export.build_mappings(
            {prefix + "in.weight": {"shape": [2, 3]}, prefix + "out.weight": {"shape": [4, 2]}}, None, {}, 8
        )


def test_common_state_singleton_list_reads_only_requested_record(tmp_path: Path) -> None:
    path = tmp_path / "part.distcp"
    args = SimpleNamespace(lora_rank=16, lora_alpha=32)
    with path.open("wb") as handle:
        handle.write(b"not optimizer payload")
        offset = handle.tell()
        torch.save([{"args": args, "iteration": 0}], handle)
        length = handle.tell() - offset
        handle.write(b"unrelated optimizer payload")
    from torch.distributed.checkpoint.metadata import MetadataIndex

    metadata = SimpleNamespace(
        storage_data={
            MetadataIndex("common_state/shard_0_1"): SimpleNamespace(
                relative_path=path.name, offset=offset, length=length
            )
        }
    )
    common = export.read_common_state(str(tmp_path), metadata)
    assert common["args"].lora_rank == 16
    assert common["iteration"] == 0
