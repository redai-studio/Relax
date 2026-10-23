# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regression for the real Bridge chunk entry point and fused-pair
matcher."""

from argparse import Namespace
from collections.abc import Iterator
from unittest.mock import Mock

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")

from relax.backends.megatron.weight_update import hf_weight_iterator_bridge as implementation  # noqa: E402


def _iterator(monkeypatch, tensors: list[tuple[str, torch.Tensor]], chunk_size: int):
    iterator = implementation.HfWeightIteratorBridge.__new__(implementation.HfWeightIteratorBridge)
    iterator.args = Namespace(update_weight_buffer_size=chunk_size)
    iterator._expert_router = None
    seen = []

    def parameters(weights, *, include_experts: bool = True) -> Iterator[tuple[str, torch.Tensor]]:
        assert weights == {"cpu_backup": "sentinel"}
        assert include_experts
        for item in tensors:
            seen.append(item[0])
            yield item

    iterator._iter_hf_params = parameters
    warnings = Mock()
    monkeypatch.setattr(implementation.logger, "warning", warnings)
    monkeypatch.setattr(implementation.dist, "get_rank", lambda group=None: 0)
    chunks = iterator.get_hf_weight_chunks({"cpu_backup": "sentinel"})
    return chunks, seen, warnings


@pytest.mark.parametrize(
    "name",
    [
        "layers.1.engram.wkv.weight",
        "layers.14.engram.wkv.scale",
        "model.layers.1.engram.wkv.weight_scale_inv",
        "language_model.model.layers.14.engram.wkv.weight",
    ],
)
def test_bridge_fused_pair_key_excludes_standalone_engram(name: str) -> None:
    assert implementation._fused_pair_key(name) is None


def test_bridge_chunks_emit_engram_immediately_and_count_its_bytes(monkeypatch) -> None:
    tensors = [
        ("layers.1.engram.wkv.weight", torch.ones(4, 8, dtype=torch.bfloat16)),
        ("layers.2.attn_norm.weight", torch.ones(32, dtype=torch.bfloat16)),
        ("layers.3.attn_norm.weight", torch.ones(32, dtype=torch.bfloat16)),
    ]
    chunks, seen, warnings = _iterator(monkeypatch, tensors, chunk_size=96)
    first = next(chunks)
    assert [name for name, _ in first] == [tensors[0][0]]
    assert first[0][1] is tensors[0][1]
    assert seen == [tensors[0][0], tensors[1][0]]
    remaining = list(chunks)
    all_chunks = [first, *remaining]
    assert [name for chunk in all_chunks for name, _ in chunk] == [name for name, _ in tensors]
    assert [sum(tensor.nbytes for _, tensor in chunk) for chunk in all_chunks] == [64, 64, 64]
    warnings.assert_not_called()


def test_bridge_chunks_count_engram_weight_and_scale_in_normal_bucket(monkeypatch) -> None:
    tensors = [
        ("layers.14.engram.wkv.weight", torch.ones(4, 8, dtype=torch.bfloat16)),
        ("layers.14.engram.wkv.scale", torch.ones(2, 2, dtype=torch.float32)),
        ("layers.15.attn_norm.weight", torch.ones(16, dtype=torch.bfloat16)),
    ]
    chunks, _, warnings = _iterator(monkeypatch, tensors, chunk_size=100)
    result = list(chunks)
    assert [[name for name, _ in chunk] for chunk in result] == [
        [tensors[0][0], tensors[1][0]],
        [tensors[2][0]],
    ]
    assert [sum(tensor.nbytes for _, tensor in chunk) for chunk in result] == [80, 32]
    warnings.assert_not_called()


@pytest.mark.parametrize(
    "left,right",
    [
        ("model.layers.0.self_attn.q_a_proj", "model.layers.0.self_attn.kv_a_proj_with_mqa"),
        ("layers.0.attn.wq_a", "layers.0.attn.wkv"),
        ("layers.0.attn.compressor.wkv", "layers.0.attn.compressor.wgate"),
    ],
)
@pytest.mark.parametrize("suffix", ["weight", "weight_scale_inv"])
@pytest.mark.parametrize("reverse", [False, True])
def test_bridge_chunks_keep_existing_attention_and_compressor_pairs(left, right, suffix, reverse, monkeypatch) -> None:
    names = [f"{left}.{suffix}", f"{right}.{suffix}"]
    if reverse:
        names.reverse()
    tensors = [
        (names[0], torch.ones(4, dtype=torch.bfloat16)),
        ("layers.0.attn_norm.weight", torch.ones(4, dtype=torch.bfloat16)),
        (names[1], torch.ones(4, dtype=torch.bfloat16)),
    ]
    chunks, _, warnings = _iterator(monkeypatch, tensors, chunk_size=16)
    result = list(chunks)
    assert [[name for name, _ in chunk] for chunk in result] == [[tensors[1][0]], names]
    assert [sum(tensor.nbytes for _, tensor in chunk) for chunk in result] == [8, 16]
    warnings.assert_not_called()


def test_bridge_chunks_keep_pair_family_and_weight_scale_keys_distinct(monkeypatch) -> None:
    names = [
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.compressor.wkv.weight",
        "layers.0.attn.wq_a.weight_scale_inv",
        "layers.0.attn.wkv.weight",
        "layers.0.attn.compressor.wgate.weight",
        "layers.0.attn.wkv.weight_scale_inv",
    ]
    tensors = [(name, torch.ones(4, dtype=torch.bfloat16)) for name in names]
    chunks, _, warnings = _iterator(monkeypatch, tensors, chunk_size=16)
    result = list(chunks)
    assert [[name for name, _ in chunk] for chunk in result] == [
        [names[0], names[3]],
        [names[1], names[4]],
        [names[2], names[5]],
    ]
    warnings.assert_not_called()


def test_bridge_chunks_still_warn_for_a_genuine_unpaired_attention_weight(monkeypatch) -> None:
    name = "layers.0.attn.wq_a.weight"
    chunks, _, warnings = _iterator(monkeypatch, [(name, torch.ones(4, dtype=torch.bfloat16))], chunk_size=16)
    result = list(chunks)
    assert [[key for key, _ in chunk] for chunk in result] == [[name]]
    warnings.assert_called_once()
    assert warnings.call_args.args[1] == name
