# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only Engram tests; run in the remote Megatron environment."""

import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file


pytest.importorskip("megatron.core.extensions.transformer_engine")

from relax.models.deepseekv41.engram import (  # noqa: E402
    DeepSeekV41Engram,
    _engram_gate,
    _mmap_rows,
    _table_shard,
    build_compressed_token_map,
    build_engram_layout,
    compute_hash_multipliers,
    engram_gate,
)


def _layout_config(**overrides: Any) -> SimpleNamespace:
    values = dict(
        v41_engram_layer_ids=(1, 2),
        v41_engram_num_embeddings=(36, 88),
        v41_engram_max_ngram_size=3,
        v41_engram_vocab_size=5,
        v41_engram_n_heads=2,
        v41_engram_head_dim=32,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _bare_engram() -> DeepSeekV41Engram:
    """Initialize only CPU helpers, without TE construction or CUDA."""
    module = DeepSeekV41Engram.__new__(DeepSeekV41Engram)
    torch.nn.Module.__init__(module)
    module.head_dim = 32
    module.rows = 4
    module.row_start = 0
    module.rows_avail = 4
    module.tp_size = 1
    module.tp_group = None
    module.cp_size = 1
    module.cp_group = None
    module.table = None
    module.table_scale = None
    module._stage = None
    module._stage_device = None
    module.max_ngram_size = 3
    module.pad_id = 0
    module.register_buffer("token_map", torch.tensor([0, 1, 2, 1]), persistent=False)
    module.register_buffer("multipliers", torch.tensor([3, 5, 7]), persistent=False)
    module.register_buffer("primes", torch.tensor([[5, 7], [11, 13]]), persistent=False)
    module.register_buffer("offsets", torch.tensor([0, 5, 12, 23]), persistent=False)
    return module


class _TokenizerStub:
    texts = ["A", "a", "Á", " ", "\t", "\ufffd", "\ufffd", "ß", ""]
    backend_tokenizer: Any

    def __init__(self) -> None:
        self.backend_tokenizer = self

    def __len__(self) -> int:
        return len(self.texts)

    def decode(self, ids: list[int], skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return self.texts[ids[0]]

    def id_to_token(self, token_id: int) -> str:
        return f"byte_{token_id}"


def test_compressed_token_map_normalizes_text_but_keeps_distinct_bytes() -> None:
    lookup, size = build_compressed_token_map(_TokenizerStub())
    assert lookup == [0, 0, 0, 1, 1, 2, 3, 4, 5]
    assert size == 6


def test_build_engram_layout_matches_unique_prime_order() -> None:
    layout = build_engram_layout(_layout_config())
    assert layout.layer_ids == (1, 2)
    assert layout.primes == (((5, 7), (11, 13)), ((17, 19), (23, 29)))
    assert layout.num_embeddings == (36, 88)


def test_build_engram_layout_allows_no_engram_layers() -> None:
    assert build_engram_layout(_layout_config(v41_engram_layer_ids=())) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"v41_engram_layer_ids": (1, 1)},
        {"v41_engram_num_embeddings": (36,)},
        {"v41_engram_num_embeddings": (35, 88)},
        {"v41_engram_max_ngram_size": 1},
        {"v41_engram_n_heads": 0},
        {"v41_engram_head_dim": 31},
        {"v41_engram_vocab_size": 0},
        {"v41_engram_layer_ids": (-1, 2)},
    ],
)
def test_build_engram_layout_rejects_invalid_metadata(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        build_engram_layout(_layout_config(**overrides))


def test_hash_multipliers_match_checkpoint_numpy_seed_and_odd_values() -> None:
    actual = compute_hash_multipliers((1, 14), 4, 99092)
    bound = (np.iinfo(np.int64).max // 99092) // 2
    expected = np.stack(
        [np.random.default_rng(10007 * layer).integers(0, bound, size=4, dtype=np.int64) * 2 + 1 for layer in (1, 14)]
    )
    torch.testing.assert_close(actual, torch.from_numpy(expected))
    assert (actual % 2 == 1).all()


@pytest.mark.parametrize("arguments", [((), 4, 10), ((1,), 1, 10), ((1,), 4, 0)])
def test_hash_multipliers_reject_invalid_geometry(arguments: tuple[Any, ...]) -> None:
    with pytest.raises(ValueError):
        compute_hash_multipliers(*arguments)


def test_hash_ids_match_independent_ngram_reference_and_sample_padding() -> None:
    module = _bare_engram()
    input_ids = torch.tensor([[1, 2, 3, 2], [2, 1, 0, 3]])
    actual = module.hash_ids(input_ids)
    expected = []
    for sample in input_ids.tolist():
        compressed = [module.token_map[token].item() for token in sample]
        sample_hashes = []
        for position in range(len(sample)):
            hashes, rolling = [], compressed[position] * 3
            for order, multiplier in enumerate((5, 7), start=1):
                token = compressed[position - order] if position >= order else module.pad_id
                rolling ^= token * multiplier
                primes = (5, 7) if order == 1 else (11, 13)
                offsets = (0, 5) if order == 1 else (12, 23)
                hashes.extend(rolling % prime + offset for prime, offset in zip(primes, offsets))
            sample_hashes.append(hashes)
        expected.append(sample_hashes)
    torch.testing.assert_close(actual, torch.tensor(expected))
    # History resets for the second batch item rather than crossing the sample boundary.
    torch.testing.assert_close(actual[1, :1], module.hash_ids(input_ids[1:2, :1])[0])


@pytest.mark.parametrize("ids", [torch.zeros(2), torch.zeros(2, 2), torch.zeros(1, 2, 3, dtype=torch.long)])
def test_hash_ids_reject_non_integer_or_non_matrix_input(ids: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        _bare_engram().hash_ids(ids)


def test_mmap_reads_only_requested_rows_and_owns_mapping_lifetime(tmp_path: Path) -> None:
    path = tmp_path / "table.safetensors"
    original = torch.arange(160, dtype=torch.uint8).view(5, 32)
    save_file({"table": original}, path)
    mapped = _mmap_rows(str(path), "table", 1, 4, 32, expected_rows=5, expected_dtypes=("U8",))
    assert mapped.device.type == "cpu"
    torch.testing.assert_close(mapped, original[1:4])
    alias = mapped[1:]
    del mapped
    gc.collect()
    # The tensor view must keep the mmap live after the original tensor is released.
    torch.testing.assert_close(alias, original[2:4])
    torch.testing.assert_close(
        _mmap_rows(str(path), "table", 5, 5, 32, expected_rows=5, expected_dtypes=("U8",)),
        torch.empty(0, 32, dtype=torch.uint8),
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "missing"},
        {"row_start": -1},
        {"row_end": 6},
        {"row_start": 3, "row_end": 2},
        {"row_bytes": 16},
        {"expected_rows": 4},
        {"expected_dtypes": ("F8_E4M3",)},
    ],
)
def test_mmap_rejects_invalid_metadata_or_range(tmp_path: Path, overrides: dict[str, Any]) -> None:
    path = tmp_path / "table.safetensors"
    save_file({"table": torch.zeros(5, 32, dtype=torch.uint8)}, path)
    arguments = dict(
        path=str(path), name="table", row_start=0, row_end=2, row_bytes=32, expected_rows=5, expected_dtypes=("U8",)
    )
    arguments.update(overrides)
    with pytest.raises(ValueError):
        _mmap_rows(**arguments)


@pytest.mark.parametrize("payload", [b"", b"bad", (4096).to_bytes(8, "little") + b"{}"])
def test_mmap_rejects_truncated_headers(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        _mmap_rows(str(path), "table", 0, 1, 32, expected_rows=1, expected_dtypes=("U8",))


@pytest.mark.parametrize("rows,tp_size", [(10, 3), (3, 8)])
def test_table_shards_cover_every_row_exactly_once(rows: int, tp_size: int) -> None:
    shards = [_table_shard(rows, tp_size, rank) for rank in range(tp_size)]
    owned = [row for start, end in shards for row in range(start, end)]
    assert owned == list(range(rows))
    assert all(0 <= start <= end <= rows for start, end in shards)


def test_table_shard_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError):
        _table_shard(10, 2, 2)


def test_frozen_tables_and_hash_buffers_are_not_checkpoint_or_optimizer_state() -> None:
    module = _bare_engram()
    module.table = torch.zeros(4, 32, dtype=torch.uint8)
    module.table_scale = torch.zeros(4, 1, dtype=torch.uint8)
    module.linear_wkv = torch.nn.Linear(32, 4, bias=False)
    module.q_weight = torch.nn.Parameter(torch.ones(1, 2))
    module.k_weight = torch.nn.Parameter(torch.ones(1, 2))
    expected = {"linear_wkv.weight", "q_weight", "k_weight"}
    assert set(dict(module.named_parameters())) == expected
    assert set(module.state_dict()) == expected
    table, scale = module.table, module.table_scale
    module.to(dtype=torch.float64)
    assert module.table is table and module.table_scale is scale
    assert module.table.dtype == torch.uint8 and module.table.device.type == "cpu"


def test_load_table_uses_checkpoint_index_and_only_local_rows(tmp_path: Path) -> None:
    module = _bare_engram()
    module.layer_id = 1
    module.row_start = 1
    module.rows_avail = 2
    module.config = SimpleNamespace(v41_hf_checkpoint=str(tmp_path))
    weight_name = "layers.1.engram.embed.weight"
    scale_name = "layers.1.engram.embed.scale"
    weight = torch.arange(128, dtype=torch.uint8).view(4, 32)
    scale = torch.full((4, 1), 127, dtype=torch.uint8)
    save_file({weight_name: weight, scale_name: scale}, tmp_path / "table.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {weight_name: "table.safetensors", scale_name: "table.safetensors"}})
    )
    module._load_table()
    torch.testing.assert_close(module.table, weight[1:3])
    torch.testing.assert_close(module.table_scale, scale[1:3])


def test_lookup_masks_unowned_rows_and_decodes_fp8_e8m0_on_cpu() -> None:
    module = _bare_engram()
    module.row_start = 1
    module.rows_avail = 2
    source = torch.tensor([[1.0] * 32, [2.0] * 32]).to(torch.float8_e4m3fn)
    module.table = source.view(torch.uint8)
    module.table_scale = torch.tensor([[127], [128]], dtype=torch.uint8)
    actual = module.lookup(torch.tensor([0, 1, 2, 3]))
    expected = torch.tensor([[0.0] * 32, [1.0] * 32, [4.0] * 32, [0.0] * 32], dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected)


def test_lookup_empty_tp_shard_still_participates_in_explicit_group_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _bare_engram()
    module.row_start = 4
    module.rows_avail = 0
    module.table = torch.empty(0, 32, dtype=torch.uint8)
    module.table_scale = torch.empty(0, 1, dtype=torch.uint8)
    module.tp_size = 8
    module.tp_group = object()
    groups = []

    def reduce(tensor: torch.Tensor, *, group: Any) -> None:
        groups.append(group)
        assert torch.count_nonzero(tensor) == 0

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    actual = module.lookup(torch.tensor([1, 3]))
    assert groups == [module.tp_group]
    assert torch.count_nonzero(actual) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_gate_matches_unchunked_forward_and_gradients(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(7, 2, 2, 4, generator=generator, dtype=dtype, requires_grad=True)
    kv = torch.randn(7, 2, 12, generator=generator, dtype=dtype, requires_grad=True)
    q = torch.randn(2, 4, generator=generator, requires_grad=True)
    k = torch.randn(2, 4, generator=generator, requires_grad=True)
    inputs = (x, kv, q, k)
    reference = _engram_gate(*inputs, 1e-6, 1e-6)
    expected_gradients = torch.autograd.grad(reference.float().sum(), inputs)
    actual = engram_gate(*inputs, 1e-6, 1e-6, chunk=3)
    actual_gradients = torch.autograd.grad(actual.float().sum(), inputs)
    torch.testing.assert_close(actual, reference)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    with torch.no_grad():
        torch.testing.assert_close(engram_gate(*inputs, 1e-6, 1e-6, chunk=3), reference)
