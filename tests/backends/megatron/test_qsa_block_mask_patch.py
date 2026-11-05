# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math

import pytest
import torch


def _selection(lengths: tuple[int, ...], pattern: str):
    from megatron.core.transformer.experimental_attention_variant import qsa

    sequence_length = sum(lengths)
    docs = []
    positions = []
    for doc, length in enumerate(lengths):
        docs.extend([doc] * length)
        positions.extend(range(length))
    blocks = math.ceil(max(lengths) / 4)
    bytes_per_row = math.ceil((blocks + 1) / 8)
    bits = torch.zeros((sequence_length, bytes_per_row), dtype=torch.uint8)
    for row, position in enumerate(positions):
        for block in range((position + 1) // 4):
            if pattern == "all" or (pattern == "sparse" and (block * 7 + row * 3) % 5 < 2):
                bits[row, block // 8] |= 1 << (block % 8)
    device = torch.device("cuda")
    return qsa.QSASelection(
        doc_ids=torch.tensor(docs, dtype=torch.int32, device=device).view(1, -1),
        positions=torch.tensor(positions, dtype=torch.int32, device=device).view(1, -1),
        selected_bits=bits.to(device).flatten(),
        bits_per_row=bytes_per_row,
        bits_per_row_t=torch.tensor(bytes_per_row, device=device),
        compress_ratio=4,
        all_selected=pattern == "all",
    )


def _block_presence(counts: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    rows, columns = indices.shape[-2:]
    present = torch.zeros((rows, columns), dtype=torch.bool)
    counts = counts.cpu().reshape(-1)
    indices = indices.cpu().reshape(rows, columns)
    for row in range(rows):
        present[row, indices[row, : counts[row]]] = True
    return present


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton BlockMask construction requires CUDA")
@pytest.mark.parametrize(
    "lengths,pattern",
    [
        ((256,), "sparse"),
        ((7, 129, 4, 116), "sparse"),
        ((3, 125, 1, 255, 127), "none"),
        ((512,), "all"),
        ((1, 1, 1, 253), "sparse"),
    ],
)
def test_qsa_block_mask_covers_exact_token_mask_without_false_full_blocks(lengths, pattern):
    from megatron.core.transformer.experimental_attention_variant import qsa

    from relax.models.qwen3_8.qsa_block_mask import _build_sparse_block_mask

    selection = _selection(lengths, pattern)
    sequence_length = sum(lengths)
    block_mask = _build_sparse_block_mask(selection, qsa._qsa_mask_mod_factory(selection), sequence_length)
    dense = qsa.build_qsa_dense_mask(selection, sequence_length).cpu()[0]
    partial = _block_presence(block_mask.kv_num_blocks, block_mask.kv_indices)
    full = _block_presence(block_mask.full_kv_num_blocks, block_mask.full_kv_indices)
    assert not (partial & full).any()
    if pattern == "all" and sequence_length >= 256:
        assert full.any()
    assert block_mask.kv_indices.shape[-1] == math.ceil(sequence_length / 128)
    for query_block in range(partial.shape[0]):
        for key_block in range(partial.shape[1]):
            tile = dense[
                query_block * 128 : (query_block + 1) * 128,
                key_block * 128 : (key_block + 1) * 128,
            ]
            if tile.any():
                assert partial[query_block, key_block] or full[query_block, key_block]
            if full[query_block, key_block]:
                assert tile.all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention forward/backward requires CUDA")
def test_qsa_block_mask_matches_reference_flex_forward_backward():
    from megatron.core.transformer.experimental_attention_variant import qsa

    from relax.models.qwen3_8.qsa_block_mask import _build_sparse_block_mask

    selection = _selection((5, 251), "sparse")
    mask_mod = qsa._qsa_mask_mod_factory(selection)
    sparse_mask = _build_sparse_block_mask(selection, mask_mod, 256)
    reference_mask = qsa.create_block_mask(mask_mod, 1, None, 256, 256, device="cuda", _compile=True)
    torch.manual_seed(13)
    inputs = [torch.randn((1, 2, 256, 32), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    upstream = torch.randn_like(inputs[0])

    def run(block_mask):
        q, k, v = (value.detach().requires_grad_(True) for value in inputs)
        output = qsa._compiled_flex_attention()(q, k, v, block_mask=block_mask)
        output.backward(upstream)
        return output.detach(), tuple(value.grad.detach() for value in (q, k, v))

    expected_output, expected_grads = run(reference_mask)
    actual_output, actual_grads = run(sparse_mask)
    torch.testing.assert_close(actual_output, expected_output, rtol=1e-2, atol=1e-2)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton BlockMask construction requires CUDA")
def test_qsa_block_mask_keeps_batch_rows_independent():
    from megatron.core.transformer.experimental_attention_variant import qsa

    from relax.models.qwen3_8.qsa_block_mask import _build_sparse_block_mask

    first = _selection((256,), "all")
    second = _selection((256,), "none")
    selection = qsa.QSASelection(
        doc_ids=torch.cat((first.doc_ids, second.doc_ids + 1), dim=0),
        positions=torch.cat((first.positions, second.positions), dim=0),
        selected_bits=torch.cat((first.selected_bits, second.selected_bits), dim=0),
        bits_per_row=first.bits_per_row,
        bits_per_row_t=first.bits_per_row_t,
        compress_ratio=4,
        all_selected=False,
    )
    block_mask = _build_sparse_block_mask(selection, qsa._qsa_mask_mod_factory(selection), 256)
    expected = qsa.build_qsa_dense_mask(selection, 256).cpu()
    for batch in range(2):
        partial = _block_presence(block_mask.kv_num_blocks[batch], block_mask.kv_indices[batch])
        full = _block_presence(block_mask.full_kv_num_blocks[batch], block_mask.full_kv_indices[batch])
        for query_block in range(2):
            for key_block in range(2):
                tile = expected[
                    batch,
                    query_block * 128 : (query_block + 1) * 128,
                    key_block * 128 : (key_block + 1) * 128,
                ]
                if tile.any():
                    assert partial[query_block, key_block] or full[query_block, key_block]
                if full[query_block, key_block]:
                    assert tile.all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention BlockMask construction requires CUDA")
@pytest.mark.parametrize("sequence_length", [257, 383])
def test_qsa_block_mask_tail_tiles_match_create_block_mask_classification(sequence_length):
    from megatron.core.transformer.experimental_attention_variant import qsa

    from relax.models.qwen3_8.qsa_block_mask import _build_sparse_block_mask

    selection = _selection((sequence_length,), "all")
    mask_mod = qsa._qsa_mask_mod_factory(selection)
    actual = _build_sparse_block_mask(selection, mask_mod, sequence_length)
    expected = qsa.create_block_mask(mask_mod, 1, None, sequence_length, sequence_length, device="cuda", _compile=True)
    assert torch.equal(
        _block_presence(actual.kv_num_blocks, actual.kv_indices),
        _block_presence(expected.kv_num_blocks, expected.kv_indices),
    )
    assert torch.equal(
        _block_presence(actual.full_kv_num_blocks, actual.full_kv_indices),
        _block_presence(expected.full_kv_num_blocks, expected.full_kv_indices),
    )
